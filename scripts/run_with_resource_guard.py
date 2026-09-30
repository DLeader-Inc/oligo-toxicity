#!/usr/bin/env python3
"""Run one command with host-memory, GPU-memory, and disk safeguards."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--breach-samples", type=int, default=2)
    parser.add_argument("--grace-seconds", type=float, default=30.0)
    parser.add_argument("--min-host-available-gib", type=float, default=20.0)
    parser.add_argument("--max-process-rss-gib", type=float, default=48.0)
    parser.add_argument("--max-gpu-used-mib", type=float, default=34_816.0)
    parser.add_argument("--min-disk-free-gib", type=float, default=20.0)
    parser.add_argument("--disk-path", type=Path, default=Path.cwd())
    parser.add_argument("--max-elapsed-seconds", type=float, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.breach_samples <= 0:
        parser.error("--breach-samples must be positive")
    if args.max_elapsed_seconds is not None and args.max_elapsed_seconds <= 0:
        parser.error("--max-elapsed-seconds must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_meminfo() -> tuple[int, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            name, value = line.split(":", maxsplit=1)
            values[name] = int(value.strip().split()[0]) * 1024
    return values["MemTotal"], values["MemAvailable"]


def read_process_table() -> dict[int, tuple[int, int]]:
    table: dict[int, tuple[int, int]] = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields: dict[str, str] = {}
            with status_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith(("Pid:", "PPid:", "VmRSS:")):
                        key, value = line.split(":", maxsplit=1)
                        fields[key] = value.strip()
            pid = int(fields["Pid"])
            ppid = int(fields["PPid"])
            rss_kib = int(fields.get("VmRSS", "0 kB").split()[0])
            table[pid] = (ppid, rss_kib * 1024)
        except (FileNotFoundError, KeyError, PermissionError, ProcessLookupError):
            continue
    return table


def process_tree_rss(root_pid: int) -> tuple[int, int]:
    table = read_process_table()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _rss) in table.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    rss = sum(table.get(pid, (0, 0))[1] for pid in descendants)
    return rss, len(descendants)


def query_gpu() -> tuple[float | None, float | None, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None, None, None
    used = 0.0
    total = 0.0
    utilization: list[float] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [float(field.strip()) for field in line.split(",")]
        used += fields[0]
        total += fields[1]
        utilization.append(fields[2])
    return used, total, max(utilization) if utilization else None


def disk_free_bytes(path: Path) -> int:
    return os.statvfs(path).f_bavail * os.statvfs(path).f_frsize


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def terminate_process_group(process: subprocess.Popen[bytes], grace: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def main() -> int:
    args = parse_args()
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    command = [str(value) for value in args.command]
    started_at = utc_now()
    process = subprocess.Popen(command, start_new_session=True)
    fieldnames = [
        "timestamp_utc",
        "elapsed_seconds",
        "pid",
        "process_count",
        "process_tree_rss_gib",
        "host_total_gib",
        "host_available_gib",
        "gpu_used_mib",
        "gpu_total_mib",
        "gpu_utilization_percent",
        "disk_free_gib",
        "breach_reasons",
    ]
    peak_rss = 0.0
    minimum_available = float("inf")
    peak_gpu = 0.0
    minimum_disk = float("inf")
    samples = 0
    consecutive_breaches = 0
    termination_reasons: list[str] = []
    status = "running"
    started_monotonic = time.monotonic()
    print(
        "resource guard started: "
        f"pid={process.pid}, host_available>={args.min_host_available_gib:.1f} GiB, "
        f"process_rss<={args.max_process_rss_gib:.1f} GiB, "
        f"gpu_used<={args.max_gpu_used_mib:.0f} MiB",
        flush=True,
    )
    try:
        with args.log_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            while process.poll() is None:
                total, available = read_meminfo()
                rss, process_count = process_tree_rss(process.pid)
                gpu_used, gpu_total, gpu_utilization = query_gpu()
                disk_free = disk_free_bytes(args.disk_path)
                rss_gib = rss / GIB
                available_gib = available / GIB
                total_gib = total / GIB
                disk_gib = disk_free / GIB
                peak_rss = max(peak_rss, rss_gib)
                minimum_available = min(minimum_available, available_gib)
                minimum_disk = min(minimum_disk, disk_gib)
                if gpu_used is not None:
                    peak_gpu = max(peak_gpu, gpu_used)
                reasons: list[str] = []
                budget_exhausted = (args.max_elapsed_seconds is not None and
                    time.monotonic() - started_monotonic >= args.max_elapsed_seconds)
                if budget_exhausted:
                    reasons.append("wall_time_budget_exhausted")
                if available_gib < args.min_host_available_gib:
                    reasons.append("host_available_below_limit")
                if rss_gib > args.max_process_rss_gib:
                    reasons.append("process_tree_rss_above_limit")
                if gpu_used is not None and gpu_used > args.max_gpu_used_mib:
                    reasons.append("gpu_memory_above_limit")
                if disk_gib < args.min_disk_free_gib:
                    reasons.append("disk_free_below_limit")
                consecutive_breaches = (
                    consecutive_breaches + 1 if reasons else 0
                )
                samples += 1
                writer.writerow(
                    {
                        "timestamp_utc": utc_now(),
                        "elapsed_seconds": round(
                            time.monotonic() - started_monotonic, 3
                        ),
                        "pid": process.pid,
                        "process_count": process_count,
                        "process_tree_rss_gib": round(rss_gib, 4),
                        "host_total_gib": round(total_gib, 4),
                        "host_available_gib": round(available_gib, 4),
                        "gpu_used_mib": (
                            "" if gpu_used is None else round(gpu_used, 1)
                        ),
                        "gpu_total_mib": (
                            "" if gpu_total is None else round(gpu_total, 1)
                        ),
                        "gpu_utilization_percent": (
                            ""
                            if gpu_utilization is None
                            else round(gpu_utilization, 1)
                        ),
                        "disk_free_gib": round(disk_gib, 4),
                        "breach_reasons": ";".join(reasons),
                    }
                )
                handle.flush()
                if samples == 1 or samples % max(
                    1, round(60.0 / args.poll_seconds)
                ) == 0:
                    print(
                        "resource sample: "
                        f"rss={rss_gib:.1f} GiB, "
                        f"host_available={available_gib:.1f} GiB, "
                        f"gpu_used={gpu_used if gpu_used is not None else 'NA'} MiB, "
                        f"disk_free={disk_gib:.1f} GiB",
                        flush=True,
                    )
                if budget_exhausted or consecutive_breaches >= args.breach_samples:
                    termination_reasons = reasons
                    status = "resource_guard_terminated"
                    print(
                        "resource guard terminating command: "
                        + ", ".join(reasons),
                        file=sys.stderr,
                        flush=True,
                    )
                    terminate_process_group(process, args.grace_seconds)
                    break
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        status = "interrupted"
        termination_reasons = ["keyboard_interrupt"]
        terminate_process_group(process, args.grace_seconds)
    return_code = process.poll()
    if return_code is None:
        return_code = process.wait()
    if status == "running":
        status = "completed" if return_code == 0 else "command_failed"
    finished_at = utc_now()
    summary = {
        "schema_version": "resource_guard_summary_v1",
        "status": status,
        "command": command,
        "pid": process.pid,
        "return_code": int(return_code),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "samples": samples,
        "peak_process_tree_rss_gib": round(peak_rss, 4),
        "minimum_host_available_gib": (
            None
            if minimum_available == float("inf")
            else round(minimum_available, 4)
        ),
        "peak_gpu_used_mib": round(peak_gpu, 1),
        "minimum_disk_free_gib": (
            None if minimum_disk == float("inf") else round(minimum_disk, 4)
        ),
        "termination_reasons": termination_reasons,
        "limits": {
            "min_host_available_gib": args.min_host_available_gib,
            "max_process_rss_gib": args.max_process_rss_gib,
            "max_gpu_used_mib": args.max_gpu_used_mib,
            "min_disk_free_gib": args.min_disk_free_gib,
            "breach_samples": args.breach_samples,
        },
        "telemetry_csv": str(args.log_csv.resolve()),
    }
    write_json_atomic(args.summary_json, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if status == "resource_guard_terminated":
        return 99
    if status == "interrupted":
        return 130
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
