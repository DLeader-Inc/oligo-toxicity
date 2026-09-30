#!/usr/bin/env python3
"""Reconstruct and verify OligoGym NeurIPS 2025 Table 3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarking.oligogym_neurips2025 import (  # noqa: E402
    load_public_benchmarks,
    reconstruct_table3,
    sha256_file,
    table3_display,
    toxicity_best_models,
    verify_table3,
)


DEFAULT_OFFICIAL_ROOT = ROOT / "external/OligoGym"
DEFAULT_OUT = ROOT / "outputs/oligogym_neurips2025_reproduction"


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    benchmark_path = args.official_root / "benchmarks/oligogym_benchmarks.csv"
    if not benchmark_path.exists():
        raise SystemExit(f"missing official benchmark CSV: {benchmark_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    public = load_public_benchmarks(benchmark_path)
    best = reconstruct_table3(public)
    verification = verify_table3(best)
    display = table3_display(best)
    toxicity = toxicity_best_models(best)

    best.to_csv(args.out_dir / "table3_best_configurations.csv", index=False)
    verification.to_csv(args.out_dir / "table3_rounding_verification.csv", index=False)
    display.to_csv(args.out_dir / "table3_reconstructed_display.csv")
    toxicity.to_csv(args.out_dir / "toxicity_table3_best_models.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "OligoGym: Curated Datasets and Benchmarks for Oligonucleotide Drug Discovery",
        "paper_url": "https://proceedings.neurips.cc/paper_files/paper/2025/file/4927dd556ff6e49671e3a77ac1f12487-Paper-Datasets_and_Benchmarks_Track.pdf",
        "official_repository": "https://github.com/Roche/OligoGym",
        "official_repository_commit": _git_commit(args.official_root),
        "benchmark_csv": str(benchmark_path),
        "benchmark_csv_sha256": sha256_file(benchmark_path),
        "benchmark_rows": int(len(public)),
        "nucleobase_rows": int(public["cross_validation"].eq("nucleobase").sum()),
        "verified_cells": int(verification["cell_match"].sum()),
        "total_table3_cells": int(len(verification)),
        "all_rounded_cells_match": bool(verification["cell_match"].all()),
        "selection_rule": "highest mean test PCC across five public nucleobase holdouts within each dataset/model class",
    }
    (args.out_dir / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(display.to_string())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not manifest["all_rounded_cells_match"]:
        mismatches = verification[~verification["cell_match"]]
        print(mismatches.to_string(index=False), file=sys.stderr)
        raise SystemExit("reconstructed Table 3 does not match the published rounded values")


if __name__ == "__main__":
    main()

