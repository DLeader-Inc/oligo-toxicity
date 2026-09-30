#!/usr/bin/env python3
"""Outer-held-out Registry/motif/chemistry ablation of selected full-AutoML recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_aso_atlas2_project_representation import load_task  # noqa: E402
from scripts.benchmark_feature_automl_nested import (  # noqa: E402
    ATLAS, EARLIER, _model, _outer_partitions, _seed_values, inverse_target,
    metrics, transform_target,
)
from scripts.benchmark_registry_nine_endpoints import earlier_frame  # noqa: E402
from src.features.automl_descriptor_routes import build_route_matrices, descriptor_family  # noqa: E402

FAMILIES = ("registry", "motif", "chemical_modification")


def component_after_ablation(
    matrix: np.ndarray, names: list[str], family: str,
) -> tuple[np.ndarray, list[str], bool]:
    molecular = [name for name in names if not name.startswith("assay__")]
    pure_removed_branch = bool(molecular) and all(descriptor_family(name) == family for name in molecular)
    keep = [
        index for index, name in enumerate(names)
        if name.startswith("assay__") or descriptor_family(name) != family
    ]
    if keep:
        return matrix[:, keep], [names[index] for index in keep], pure_removed_branch
    return np.zeros((len(matrix), 1), dtype=np.float32), ["empty_intercept_column"], pure_removed_branch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/mechanism_informed_automl_nested_v1/full_automl")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--datasets", nargs="*", choices=EARLIER + ATLAS)
    parser.add_argument("--partitions", nargs="*", type=int, choices=range(1, 6))
    parser.add_argument("--families", nargs="*", choices=FAMILIES)
    parser.add_argument("--checkpoint-only", action="store_true")
    args = parser.parse_args()
    out = args.run_dir / "family_ablation"
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    audit_rows = []
    selected_datasets = args.datasets or (EARLIER + ATLAS)
    for dataset in selected_datasets:
        frame = earlier_frame(dataset) if dataset in EARLIER else load_task(dataset).reset_index(drop=True)
        matrices, schemas, _manifest = build_route_matrices(frame)
        y = frame.y.to_numpy(float)
        for partition in _outer_partitions(dataset, frame):
            fold = int(partition["partition"])
            if args.partitions and fold not in args.partitions:
                continue
            train = np.asarray(partition["train"], dtype=int)
            test = np.asarray(partition["test"], dtype=int)
            selection = json.loads((args.run_dir / "fold_checkpoints" / f"{dataset}__partition{fold}__selection.json").read_text())
            transform = selection["selected_transform"]
            fit_y = transform_target(y, transform)
            seeds = _seed_values(dataset, fold, args.seed_count)
            components = list(selection["selected_components"])
            base_weights = np.asarray(selection["selected_weights"], dtype=float)
            for family in (args.families or FAMILIES):
                path = checkpoints / f"{dataset}__partition{fold}__remove_{family}.csv.gz"
                kept_components = []
                kept_matrices = []
                kept_names = []
                kept_weights = []
                removed_branches = []
                for component, weight in zip(components, base_weights, strict=True):
                    matrix, names, pure = component_after_ablation(matrices[component], schemas[component], family)
                    if len(components) > 1 and pure:
                        removed_branches.append(component)
                        continue
                    kept_components.append(component)
                    kept_matrices.append(matrix)
                    kept_names.append(names)
                    kept_weights.append(weight)
                if not kept_components:
                    kept_components = ["empty_intercept"]
                    kept_matrices = [np.zeros((len(frame), 1), dtype=np.float32)]
                    kept_names = [["empty_intercept_column"]]
                    kept_weights = [1.0]
                weights = np.asarray(kept_weights, dtype=float)
                weights /= weights.sum()
                audit_rows.append({
                    "dataset": dataset, "outer_partition": fold, "ablation_family": family,
                    "selected_components": json.dumps(components), "removed_late_fusion_branches": json.dumps(removed_branches),
                    "refit_components": json.dumps(kept_components), "renormalized_weights": json.dumps(weights.tolist()),
                    "remaining_feature_counts": json.dumps([len(names) for names in kept_names]),
                    "transform": transform, "outer_test_labels_used_for_refit_or_selection": False,
                })
                if path.exists():
                    current = pd.read_csv(path)
                    prediction_rows.append(current)
                    continue
                members = []
                for seed in seeds:
                    component_predictions = []
                    for offset, matrix in enumerate(kept_matrices):
                        estimator = _model(selection["selected_xgboost_params"], seed + 499 * offset, args.device)
                        estimator.fit(matrix[train], fit_y[train])
                        component_predictions.append(np.asarray(estimator.predict(matrix[test]), dtype=float))
                    model_prediction = sum(weight * pred for weight, pred in zip(weights, component_predictions, strict=True))
                    members.append(inverse_target(model_prediction, transform))
                values = np.column_stack(members)
                current = pd.DataFrame({
                    "dataset": dataset, "outer_partition": fold, "row_index": test, "y": y[test],
                    "ablation_family": family, "prediction": values.mean(axis=1),
                    **{f"pred_member_{index + 1}": values[:, index] for index in range(values.shape[1])},
                })
                current.to_csv(path, index=False, compression="gzip")
                prediction_rows.append(current)
    if args.checkpoint_only:
        print(json.dumps({"status": "checkpoint_only_complete", "datasets": list(selected_datasets)}))
        return
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(out / "family_ablation_oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(audit_rows).to_csv(out / "family_ablation_refit_audit.csv", index=False)
    metric_rows = []
    for (dataset, fold, family), part in predictions.groupby(["dataset", "outer_partition", "ablation_family"]):
        metric_rows.append({"dataset": dataset, "outer_partition": fold, "ablation_family": family, **metrics(part.y.to_numpy(), part.prediction.to_numpy())})
    fold_metrics = pd.DataFrame(metric_rows)
    fold_metrics.to_csv(out / "family_ablation_fold_metrics.csv", index=False)
    endpoint = fold_metrics.groupby(["dataset", "ablation_family"], as_index=False).agg(
        spearman=("spearman", "mean"), raw_r2=("r2", "mean"), rmse=("rmse", "mean"),
        rmse_over_y_std=("rmse_over_y_std", "mean"), outer_partitions=("outer_partition", "nunique"),
    )
    endpoint.to_csv(out / "family_ablation_endpoint_metrics.csv", index=False)
    policy_values = []
    for selection_file in sorted((args.run_dir / "fold_checkpoints").glob("*__selection.json")):
        policy_values.append(json.loads(selection_file.read_text())["policy"])
    manifest = {
        "schema_version": "outer_heldout_descriptor_family_ablation_v1",
        "policy": sorted(set(policy_values)),
        "families": list(FAMILIES), "outer_partitions": int(fold_metrics[["dataset", "outer_partition"]].drop_duplicates().shape[0]),
        "seed_count": args.seed_count,
        "claim_boundary": "functional dependence of refitted models on descriptor families; not a biological causal effect",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(endpoint.to_string(index=False))


if __name__ == "__main__":
    main()
