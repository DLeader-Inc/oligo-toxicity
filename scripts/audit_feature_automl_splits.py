#!/usr/bin/env python3
"""Audit outer/inner grouping and selection isolation for feature AutoML v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_aso_atlas2_project_representation import canonical_aso_sequence, load_task  # noqa: E402
from scripts.benchmark_feature_automl_nested import ATLAS, EARLIER, _outer_partitions  # noqa: E402
from scripts.benchmark_registry_nine_endpoints import earlier_frame  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/integrated_feature_automl_v2/formal")
    args = parser.parse_args()
    rows = []
    for dataset in EARLIER + ATLAS:
        frame = earlier_frame(dataset) if dataset in EARLIER else load_task(dataset).reset_index(drop=True)
        helms = frame["HELM Annotation"].astype(str).to_numpy()
        sequences = frame.x.map(canonical_aso_sequence).astype(str).to_numpy()
        for partition in _outer_partitions(dataset, frame):
            train = np.asarray(partition["train"], dtype=int)
            test = np.asarray(partition["test"], dtype=int)
            groups = np.asarray(partition["groups"]).astype(str)
            rows.append({
                "dataset": dataset,
                "outer_partition": int(partition["partition"]),
                "grouping_contract": "earliest_patent" if dataset in ATLAS else "base_1_2_3mer_kmeans_group",
                "train_rows": len(train),
                "test_rows": len(test),
                "assigned_group_overlap": len(set(groups[train]).intersection(groups[test])),
                "exact_helm_overlap": len(set(helms[train]).intersection(helms[test])),
                "canonical_sequence_overlap": len(set(sequences[train]).intersection(sequences[test])),
            })
    outer = pd.DataFrame(rows)
    outer.to_csv(args.run_dir / "outer_split_overlap_audit.csv", index=False)

    inner_paths = sorted((args.run_dir / "fold_checkpoints").glob("*__inner_scores.csv.gz"))
    inner = pd.concat([pd.read_csv(path) for path in inner_paths], ignore_index=True) if inner_paths else pd.DataFrame()
    selections = [json.loads(path.read_text()) for path in sorted((args.run_dir / "fold_checkpoints").glob("*__selection.json"))]
    summary = {
        "schema_version": "feature_automl_split_audit_v1",
        "outer_folds": len(outer),
        "assigned_group_overlap_total": int(outer.assigned_group_overlap.sum()),
        "exact_helm_overlap_total": int(outer.exact_helm_overlap.sum()),
        "canonical_sequence_overlap_total": int(outer.canonical_sequence_overlap.sum()),
        "canonical_sequence_overlap_folds": int(outer.canonical_sequence_overlap.gt(0).sum()),
        "inner_rows": len(inner),
        "inner_group_overlap_total": int(inner.group_overlap.sum()) if len(inner) else None,
        "selection_manifests": len(selections),
        "selection_using_outer_test": sum(bool(row.get("selection_uses_outer_test")) for row in selections),
        "interpretation": (
            "The declared grouping unit and exact HELM are leakage controls. Canonical-sequence overlap with different chemistry/source is reported separately as an applicability limitation, not hidden."
        ),
    }
    (args.run_dir / "split_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    run_manifest_path = args.run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        manifest = json.loads(run_manifest_path.read_text())
        implementation = [
            ROOT / "scripts/benchmark_feature_automl_nested.py",
            ROOT / "src/features/automl_descriptor_routes.py",
            ROOT / "scripts/benchmark_xgb_robust_nested.py",
            ROOT / "scripts/benchmark_aso_atlas2_project_representation.py",
        ]
        manifest.update({
            "schema_version": "integrated_feature_automl_strict_nested_v4_audited",
            "local_compatibility_patches": "none; the formal route is a project-native implementation",
            "task_checkpoint_usage": "none; XGBoost models were trained from scratch",
            "fold_checkpoint_usage": "per-partition compressed inner scores, predictions, selection JSON, model files, and membership manifest; v4-complete folds only are reusable",
            "intentional_deviations": [
                "source-native protocols remain contextual project-recomputed artifacts and are not refitted by this training entry point",
                "five seed members are fitted after selection; grouped inner selection uses a fixed deterministic seed per inner fold",
                "an ultra-compact fifth capacity was added as a conservative challenger; it is governed by the same inner-only safety gate",
            ],
            "implementation_hashes": {str(path.relative_to(ROOT)): _sha256(path) for path in implementation},
            "split_audit_path": str((args.run_dir / "split_audit_summary.json").resolve()),
            "split_audit": summary,
            "resource_guard_summary_path": str((args.run_dir.parent / "resource_guard/summary_formal_v4_p5.json").resolve()),
        })
        run_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
