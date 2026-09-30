from __future__ import annotations

from pathlib import Path

from scripts.run_with_resource_guard import (
    disk_free_bytes,
    process_tree_rss,
    read_meminfo,
)


def test_resource_guard_reads_current_host_state() -> None:
    total, available = read_meminfo()
    assert total > 0
    assert 0 < available <= total
    assert disk_free_bytes(Path.cwd()) > 0


def test_resource_guard_counts_current_process() -> None:
    import os

    rss, process_count = process_tree_rss(os.getpid())
    assert rss > 0
    assert process_count >= 1


def test_guard_enforces_wall_clock_budget(tmp_path):
    import json
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1]/'scripts/run_with_resource_guard.py'
    result = subprocess.run([sys.executable,str(script),'--log-csv',str(tmp_path/'telemetry.csv'),
        '--summary-json',str(tmp_path/'summary.json'),'--max-elapsed-seconds','0.1',
        '--poll-seconds','0.05','--grace-seconds','0.1','--min-host-available-gib','0',
        '--min-disk-free-gib','0','--',sys.executable,'-c','import time; time.sleep(60)'],
        capture_output=True,text=True,timeout=20)
    assert result.returncode != 0
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['status']=='resource_guard_terminated'
    assert 'wall_time_budget_exhausted' in summary['termination_reasons']
    assert (tmp_path/'telemetry.csv').stat().st_size > 0
