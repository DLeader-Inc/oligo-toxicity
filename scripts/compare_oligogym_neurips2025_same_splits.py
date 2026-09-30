#!/usr/bin/env python3
"""Run OligoGym published recipes and the current baseline on identical splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarking.oligogym_neurips2025 import (  # noqa: E402
    load_public_benchmarks,
    reconstruct_table3,
    sha256_file,
)
from src.benchmarking.oligogym_grouped_splits import (  # noqa: E402
    exact_sequence_groups,
    grouping_feasibility,
    joint_target_sequence_component_groups,
    load_similarity_groups,
    make_groupkfold_splits,
    split_overlap_counts,
    target_groups,
)
from src.benchmarking.oligogym_same_split import (  # noqa: E402
    ENDPOINT_SOURCES,
    clone_recipe_row,
    current_baseline_matrix,
    fit_oligogym_features,
    fit_predict_current_baseline,
    fit_predict_project_xgb_features,
    fit_predict_published_recipe,
    load_endpoint_frame,
    make_exact_sequence_splits,
    make_nucleobase_splits,
    regression_metrics,
)


DEFAULT_OFFICIAL_ROOT = ROOT / "external/OligoGym"
DEFAULT_OUT = ROOT / "outputs/oligogym_neurips2025_same_split"
ROUTE_ORDER = (
    "current_core_baseline_xgb",
    "oligogym_published_xgb",
    "oligogym_published_features_project_xgb",
    "oligogym_published_overall",
)
TARGET_ELIGIBLE = {"neurotoxicity_moe", "tlr7_immunotox", "tlr8_immunotox"}
SPLIT_MODE_CHOICES = (
    "nucleobase",
    "sequence",
    "similarity90",
    "target",
    "joint_target_similarity90",
)


def _git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _published_recipes(best: pd.DataFrame, benchmark_dataset: str) -> dict[str, dict[str, object]]:
    endpoint = best[best["dataset"].eq(benchmark_dataset)].copy()
    xgb = endpoint[endpoint["paper_model"].eq("XGB")].iloc[0]
    overall = endpoint.sort_values(["pcc_mean", "pcc_std"], ascending=[False, True]).iloc[0]
    return {
        "oligogym_published_xgb": clone_recipe_row(xgb),
        "oligogym_published_features_project_xgb": clone_recipe_row(xgb),
        "oligogym_published_overall": clone_recipe_row(overall),
    }


def _aggregate(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    endpoint_stats: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_columns = ["project_dataset", "paper_dataset", "split_mode", "route"]
    for keys, group in fold_metrics.groupby(group_columns, sort=False):
        record = dict(zip(group_columns, keys, strict=True))
        for metric in ["pearson", "spearman", "r2", "rmse", "y_std", "rmse_over_y_std"]:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1))
        record["n_splits"] = int(len(group))
        record["test_rows_mean"] = float(group["n_test"].mean())
        route_predictions = predictions[
            predictions["project_dataset"].eq(record["project_dataset"])
            & predictions["split_mode"].eq(record["split_mode"])
            & predictions["route"].eq(record["route"])
        ]
        averaged = route_predictions.groupby("row_index", as_index=False).agg(
            y=("y", "first"), prediction=("prediction", "mean")
        )
        y_std = float(
            endpoint_stats.loc[
                endpoint_stats["project_dataset"].eq(record["project_dataset"]), "y_std"
            ].iloc[0]
        )
        pooled = regression_metrics(
            averaged["y"].to_numpy(dtype=float),
            averaged["prediction"].to_numpy(dtype=float),
            normalization_std=y_std,
        )
        for metric, value in pooled.items():
            record[f"covered_rows_{metric}"] = value
        n_total = int(
            endpoint_stats.loc[
                endpoint_stats["project_dataset"].eq(record["project_dataset"]), "rows"
            ].iloc[0]
        )
        record["covered_unique_rows"] = int(len(averaged))
        record["coverage_fraction"] = float(len(averaged) / n_total)
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[source.project_key for source in ENDPOINT_SOURCES],
    )
    parser.add_argument(
        "--split-modes",
        nargs="+",
        choices=list(SPLIT_MODE_CHOICES),
        default=["nucleobase", "sequence"],
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=list(ROUTE_ORDER),
        default=list(ROUTE_ORDER),
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = args.official_root / "benchmarks/oligogym_benchmarks.csv"
    public = load_public_benchmarks(benchmark_path)
    best = reconstruct_table3(public)
    selected_sources = [
        source for source in ENDPOINT_SOURCES if source.project_key in set(args.datasets)
    ]
    if len(selected_sources) != len(set(args.datasets)):
        known = {source.project_key for source in ENDPOINT_SOURCES}
        raise SystemExit(f"unknown dataset requested; valid values: {sorted(known)}")

    fold_records: list[dict[str, object]] = []
    prediction_records: list[pd.DataFrame] = []
    split_records: list[pd.DataFrame] = []
    endpoint_records: list[dict[str, object]] = []
    feasibility_records: list[dict[str, object]] = []
    resource_hashes: dict[str, str] = {}

    for source in selected_sources:
        frame, resource_path = load_endpoint_frame(args.official_root, source)
        resource_hashes[source.project_key] = sha256_file(resource_path)
        y = frame["y"].to_numpy(dtype=np.float32)
        helms = frame["x"].astype(str).to_numpy()
        current_features, parsed = current_baseline_matrix(frame)
        exact_groups = exact_sequence_groups(parsed, dataset_key=source.project_key)
        similarity_groups = load_similarity_groups(
            parsed, dataset_key=source.project_key, threshold=0.90
        )
        endpoint_target_groups = target_groups(parsed)
        joint_groups = joint_target_sequence_component_groups(
            endpoint_target_groups, similarity_groups
        )
        groups_by_dimension = {
            "exact_sequence": exact_groups,
            "similarity90": similarity_groups,
            "target": endpoint_target_groups,
            "joint_target_similarity90": joint_groups,
        }
        for mode, labels in groups_by_dimension.items():
            feasibility = grouping_feasibility(labels, split_mode=mode)
            feasibility_records.append(
                {
                    "project_dataset": source.project_key,
                    "paper_dataset": source.paper_dataset,
                    **feasibility.__dict__,
                    "benchmark_enabled": mode not in {"target", "joint_target_similarity90"}
                    or source.project_key in TARGET_ELIGIBLE,
                }
            )
        y_std = float(np.std(y, ddof=0))
        endpoint_records.append(
            {
                "project_dataset": source.project_key,
                "paper_dataset": source.paper_dataset,
                "rows": len(frame),
                "y_mean": float(np.mean(y)),
                "y_median": float(np.median(y)),
                "y_std": y_std,
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
                "current_baseline_features": int(current_features.shape[1]),
                "resource_file": str(resource_path),
                "resource_sha256": resource_hashes[source.project_key],
            }
        )
        recipes = _published_recipes(best, source.benchmark_dataset)
        splits = []
        if "nucleobase" in args.split_modes:
            splits.extend(make_nucleobase_splits(helms, y))
        if "sequence" in args.split_modes:
            splits.extend(
                make_exact_sequence_splits(parsed, y, dataset_key=source.project_key)
            )
        if "similarity90" in args.split_modes:
            splits.extend(
                make_groupkfold_splits(
                    similarity_groups,
                    y,
                    split_mode="sequence_identity90_groupkfold",
                    n_splits=5,
                )
            )
        if "target" in args.split_modes and source.project_key in TARGET_ELIGIBLE:
            splits.extend(
                make_groupkfold_splits(
                    endpoint_target_groups,
                    y,
                    split_mode="target_groupkfold",
                    n_splits=5,
                )
            )
        if (
            "joint_target_similarity90" in args.split_modes
            and source.project_key in TARGET_ELIGIBLE
        ):
            splits.extend(
                make_groupkfold_splits(
                    joint_groups,
                    y,
                    split_mode="joint_target_sequence_identity90_groupkfold",
                    n_splits=5,
                )
            )

        if args.probe:
            splits = splits[:1]

        prediction_cache: dict[
            tuple[str, tuple[int, ...], tuple[int, ...]],
            tuple[np.ndarray, dict[str, object]],
        ] = {}
        for split in splits:
            train_groups = set(split.group_labels[split.train_indices].tolist())
            test_groups = set(split.group_labels[split.test_indices].tolist())
            overlap = train_groups & test_groups
            if overlap:
                raise RuntimeError(
                    f"group leakage for {source.project_key}/{split.split_mode}/{split.split_id}"
                )
            dimension_overlaps = split_overlap_counts(
                split.train_indices,
                split.test_indices,
                groups_by_dimension,
            )
            if split.split_mode == "sequence_identity90_groupkfold":
                if dimension_overlaps["similarity90_overlap"]:
                    raise RuntimeError("similarity90 leakage detected")
            if split.split_mode == "target_groupkfold":
                if dimension_overlaps["target_overlap"]:
                    raise RuntimeError("target leakage detected")
            if split.split_mode == "joint_target_sequence_identity90_groupkfold":
                if dimension_overlaps["target_overlap"] or dimension_overlaps["similarity90_overlap"]:
                    raise RuntimeError("joint target/sequence leakage detected")
            assignment = np.full(len(frame), "not_tested", dtype=object)
            assignment[split.train_indices] = "train"
            assignment[split.test_indices] = "test"
            split_records.append(
                pd.DataFrame(
                    {
                        "project_dataset": source.project_key,
                        "paper_dataset": source.paper_dataset,
                        "split_mode": split.split_mode,
                        "split_id": split.split_id,
                        "row_index": np.arange(len(frame)),
                        "group": split.group_labels.astype(str),
                        "assignment": assignment,
                        **dimension_overlaps,
                    }
                )
            )
            train_idx = split.train_indices
            test_idx = split.test_indices
            for route in args.routes:
                run_seed = 42 + split.split_id
                metadata: dict[str, object] = {}
                cache_key = (
                    route,
                    tuple(train_idx.tolist()),
                    tuple(test_idx.tolist()),
                )
                if cache_key in prediction_cache:
                    prediction, cached_metadata = prediction_cache[cache_key]
                    metadata = dict(cached_metadata)
                    metadata["reused_identical_partition"] = True
                else:
                    if route == "current_core_baseline_xgb":
                        prediction = fit_predict_current_baseline(
                            current_features[train_idx],
                            y[train_idx],
                            current_features[test_idx],
                            seed=run_seed,
                            n_jobs=args.n_jobs,
                        )
                    elif route == "oligogym_published_features_project_xgb":
                        recipe = recipes[route]
                        x_train, x_test = fit_oligogym_features(
                            helms[train_idx],
                            helms[test_idx],
                            featurizer_name=str(recipe["featurizer"]),
                            featurizer_args=dict(recipe["featurizer_args"]),
                        )
                        metadata["feature_shape"] = "x".join(
                            str(value) for value in x_train.shape[1:]
                        )
                        prediction = fit_predict_project_xgb_features(
                            x_train,
                            y[train_idx],
                            x_test,
                            seed=run_seed,
                            n_jobs=args.n_jobs,
                            device=args.device,
                        )
                    else:
                        recipe = recipes[route]
                        prediction, metadata = fit_predict_published_recipe(
                            helms[train_idx],
                            y[train_idx],
                            helms[test_idx],
                            model_name=str(recipe["model"]),
                            model_args=dict(recipe["model_args"]),
                            featurizer_name=str(recipe["featurizer"]),
                            featurizer_args=dict(recipe["featurizer_args"]),
                            seed=run_seed,
                            n_jobs=args.n_jobs,
                            device=args.device,
                        )
                    prediction_cache[cache_key] = (prediction, dict(metadata))
                if route == "current_core_baseline_xgb":
                    model_class = "XGBoostModel"
                    featurizer = "project_core_baseline"
                    published_pcc_mean = np.nan
                    published_pcc_std = np.nan
                elif route == "oligogym_published_features_project_xgb":
                    recipe = recipes[route]
                    model_class = "ProjectXGBoost"
                    featurizer = str(recipe["featurizer"])
                    published_pcc_mean = np.nan
                    published_pcc_std = np.nan
                else:
                    recipe = recipes[route]
                    model_class = str(recipe["model"])
                    featurizer = str(recipe["featurizer"])
                    published_pcc_mean = float(recipe["published_pcc_mean"])
                    published_pcc_std = float(recipe["published_pcc_std"])
                metrics = regression_metrics(
                    y[test_idx], prediction, normalization_std=y_std
                )
                fold_records.append(
                    {
                        "project_dataset": source.project_key,
                        "paper_dataset": source.paper_dataset,
                        "split_mode": split.split_mode,
                        "split_id": split.split_id,
                        "route": route,
                        "model_class": model_class,
                        "featurizer": featurizer,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "train_groups": len(train_groups),
                        "test_groups": len(test_groups),
                        "group_overlap": len(overlap),
                        **dimension_overlaps,
                        "global_y_std": y_std,
                        "published_pcc_mean": published_pcc_mean,
                        "published_pcc_std": published_pcc_std,
                        **metrics,
                        **metadata,
                    }
                )
                prediction_records.append(
                    pd.DataFrame(
                        {
                            "project_dataset": source.project_key,
                            "paper_dataset": source.paper_dataset,
                            "split_mode": split.split_mode,
                            "split_id": split.split_id,
                            "route": route,
                            "row_index": test_idx,
                            "group": split.group_labels[test_idx].astype(str),
                            "y": y[test_idx],
                            "prediction": prediction,
                        }
                    )
                )
                print(
                    f"{source.project_key} {split.split_mode} {split.split_id} {route}: "
                    f"PCC={metrics['pearson']:.3f} SCC={metrics['spearman']:.3f} "
                    f"R2={metrics['r2']:.3f} RMSE/y_std={metrics['rmse_over_y_std']:.3f}"
                )

    fold_metrics = pd.DataFrame(fold_records)
    predictions = pd.concat(prediction_records, ignore_index=True)
    split_manifest = pd.concat(split_records, ignore_index=True)
    endpoint_stats = pd.DataFrame(endpoint_records)
    feasibility = pd.DataFrame(feasibility_records)
    aggregate = _aggregate(fold_metrics, predictions, endpoint_stats)

    fold_metrics.to_csv(args.out_dir / "same_split_fold_metrics.csv", index=False)
    predictions.to_csv(args.out_dir / "same_split_predictions.csv", index=False)
    split_manifest.to_csv(args.out_dir / "same_split_manifest.csv", index=False)
    endpoint_stats.to_csv(args.out_dir / "endpoint_data_audit.csv", index=False)
    feasibility.to_csv(args.out_dir / "grouping_feasibility_audit.csv", index=False)
    aggregate.to_csv(args.out_dir / "same_split_aggregate_metrics.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "official_repository": "https://github.com/Roche/OligoGym",
        "official_repository_commit": _git_commit(args.official_root),
        "benchmark_csv_sha256": sha256_file(benchmark_path),
        "resource_hashes": resource_hashes,
        "datasets": [source.project_key for source in selected_sources],
        "split_modes": args.split_modes,
        "routes": args.routes,
        "n_jobs": args.n_jobs,
        "requested_device": args.device,
        "probe": args.probe,
        "metric_policy": {
            "required": ["spearman", "r2", "rmse", "y_std", "rmse_over_y_std"],
            "fold_normalization": "endpoint population y_std shared across routes",
            "interpretation": "Spearman measures ranking; R2 and RMSE/y_std measure numerical prediction",
        },
        "nucleobase_protocol": {
            "kmer": [1, 2, 3],
            "normalization": "L2",
            "kmeans_clusters": 8,
            "seeds": [0, 1, 2, 3, 4],
            "test_size_groups": 0.2,
            "splitter": "GroupShuffleSplit",
        },
        "sequence_protocol": {
            "group": "HELM-derived exact molecular base sequence",
            "splitter": "GroupKFold",
            "folds": 5,
        },
        "similarity_protocol": {
            "identity": "1 - Levenshtein edit distance / max sequence length",
            "threshold": 0.90,
            "cluster": "connected components of all pairs at or above threshold",
            "splitter": "GroupKFold",
            "folds": 5,
        },
        "target_protocol": {
            "group": "source target annotation",
            "eligible_datasets": sorted(TARGET_ELIGIBLE),
            "splitter": "GroupKFold",
        },
        "joint_protocol": {
            "group": "connected components of target-sequence_identity90 bipartite graph",
            "guarantees": ["zero target overlap", "zero sequence_identity90 overlap"],
            "splitter": "GroupKFold",
        },
        "all_group_overlaps_zero": bool(fold_metrics["group_overlap"].eq(0).all()),
    }
    (args.out_dir / "same_split_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
