"""Print or run resource-guarded training stages without altering the recipe."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_feature_automl_nested import POLICIES


def command(stage, device):
    python = sys.executable
    if stage == "source_oligogym":
        return [python, "scripts/compare_oligogym_neurips2025_same_splits.py", "--official-root", "external/OligoGym",
                "--out-dir", "outputs/oligogym_neurips2025_same_split", "--split-modes", "nucleobase",
                "--routes", "oligogym_published_overall", "oligogym_published_xgb", "--n-jobs", "1", "--device", device]
    if stage == "source_atlas":
        return [python, "scripts/benchmark_aso_atlas2_four_stage_factorial.py", "--out-dir", "outputs/aso_atlas2_four_stage_factorial_v1/full",
                "--device", device, "--parallel-jobs", "1"]
    policy = "expert_rule_scale" if stage == "primary" else stage
    if policy in {"expert_rule_scale", "expert_nested_scale"}:
        parent = "expert_rule_scale_v1"
    elif policy.startswith("expert_rule_"):
        parent = "expert_rule_capacity_v1"
    else:
        parent = "mechanism_informed_automl_nested_v1"
    directory = {"expert_rule_fixed_ultra": "fixed_ultra", "expert_rule_fixed_compact": "fixed_compact",
                 "expert_rule_fixed_conservative": "fixed_conservative", "expert_rule_fixed_deep": "fixed_deep",
                 "expert_rule_capacity_best": "capacity_inner_best", "expert_rule_capacity_guarded": "capacity_guarded"}.get(policy, policy)
    return [python, "scripts/benchmark_feature_automl_nested.py", "--policy", policy,
            "--out-dir", f"outputs/{parent}/{directory}", "--outer-grouping", "source_protocol",
            "--seed-count", "5", "--inner-folds", "3", "--device", device, "--parallel-folds", "1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["primary", "source_oligogym", "source_atlas", *POLICIES])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    formal = command(args.stage, args.device)
    print(json.dumps({"probe": formal + ["--probe"], "formal": formal}, indent=2))
    if not args.run:
        return
    mem = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2 and line.split()[1].isdigit()}
    if mem["MemAvailable"] < 20 * 1024**3 or shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise SystemExit("Resource preflight failed: require >=20 GiB host available and >=20 GiB disk free")
    gpu_limit = 34816
    if args.device == "cuda":
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], text=True)
        totals = [tuple(map(float, line.split(","))) for line in raw.strip().splitlines()]
        if any(total-used < 6144 for total,used in totals):
            raise SystemExit("Resource preflight failed: require >=6 GiB unused GPU memory")
        gpu_limit = min(total for total,used in totals)-6144
    logs = ROOT / "rerun/resources" / args.stage
    logs.mkdir(parents=True, exist_ok=True)
    for phase, cmd in (("probe", formal + ["--probe"]), ("formal", formal)):
        guard = [sys.executable, "scripts/run_with_resource_guard.py", "--log-csv", str(logs / (phase + ".csv")),
                 "--summary-json", str(logs / (phase + ".json")), "--min-host-available-gib", "20",
                 "--max-process-rss-gib", "48", "--max-gpu-used-mib", str(gpu_limit), "--", *cmd]
        subprocess.run(guard, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
