#!/usr/bin/env python3
"""Compare registry-complete One-Hot and descriptors on nine toxicity endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_aso_atlas2_project_representation import (  # noqa: E402
    PROJECT_PARAMS,
    experimental_conditions,
    fit_predict,
    load_task,
    project_features,
    regression_metrics,
    similarity90_groups,
)
from src.benchmarking.oligogym_same_split import make_nucleobase_splits  # noqa: E402
from src.data.datasets import get_dataset, load_oligogym  # noqa: E402
from src.features.chemistry_registry import registry_manifest  # noqa: E402
from src.features.universal_oligo_representation import (  # noqa: E402
    pre_cet_ohe_feature_names,
    schema_manifest,
    universal_context_names,
    universal_core_names,
    universal_ohe_feature_names,
)


EARLIER = (
    "cytotoxicity",
    "neurotoxicity_lna",
    "neurotoxicity_moe",
    "tlr7_immunotox",
    "tlr8_immunotox",
)
ATLAS = ("mouse_hepatic", "rat_hepatic", "mouse_neuro", "rat_neuro")
DEFAULT_SPLITS = ROOT / "outputs/chemistry_position_subblock_ablation_v1/oof_predictions.csv.gz"


def matrices(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    features = project_features(frame)
    conditions = experimental_conditions(frame)
    raw = np.column_stack([features["raw"], conditions]).astype(np.float32)
    raw_names = universal_ohe_feature_names()
    legacy_names = pre_cet_ohe_feature_names()
    legacy_indices = [raw_names.index(name) for name in legacy_names]
    pre_cet_raw = np.column_stack(
        [features["raw"][:, legacy_indices], conditions]
    ).astype(np.float32)
    branch_a = np.column_stack([features["branch_a"], conditions]).astype(np.float32)
    branch_b = np.column_stack([features["branch_b"], conditions]).astype(np.float32)
    n_core = len(universal_core_names())
    n_context = len(universal_context_names())
    molecular_branch_b = features["branch_b"]
    core = molecular_branch_b[:, :n_core]
    context = molecular_branch_b[:, n_core : n_core + n_context]
    seqchem = molecular_branch_b[:, n_core + n_context :]
    branch_b_no_core = np.column_stack([context, seqchem, conditions]).astype(np.float32)
    branch_b_no_context = np.column_stack([core, seqchem, conditions]).astype(np.float32)
    branch_b_no_seqchem = np.column_stack([core, context, conditions]).astype(np.float32)
    return {
        "pre_cet_onehot": pre_cet_raw,
        "onehot": raw,
        "branch_a": branch_a,
        "branch_b": branch_b,
        "branch_b_no_core": branch_b_no_core,
        "branch_b_no_context": branch_b_no_context,
        "branch_b_no_seqchem": branch_b_no_seqchem,
    }, {
        "pre_cet_onehot": pre_cet_raw.shape[1],
        "onehot": raw.shape[1],
        "branch_a": branch_a.shape[1],
        "branch_b": branch_b.shape[1],
        "branch_b_no_core": branch_b_no_core.shape[1],
        "branch_b_no_context": branch_b_no_context.shape[1],
        "branch_b_no_seqchem": branch_b_no_seqchem.shape[1],
        "conditions": conditions.shape[1],
    }


def earlier_frame(dataset: str) -> pd.DataFrame:
    source = get_dataset(dataset)
    frame = load_oligogym(source.oligogym_key).reset_index(drop=True).copy()
    frame["HELM Annotation"] = frame["x"].astype(str)
    return frame


def earlier_assignments(dataset: str, split_path: Path) -> pd.DataFrame:
    usecols = ["repeat", "dataset", "outer_mode", "outer_fold", "row_index"]
    frame = pd.read_csv(split_path, usecols=usecols)
    return (
        frame[frame.dataset.eq(dataset) & frame.outer_mode.eq("similarity90")]
        .drop_duplicates()
        .sort_values(["repeat", "outer_fold", "row_index"])
        .reset_index(drop=True)
    )


def earlier_nucleobase_assignments(
    dataset: str,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate OligoGym's public k-mer/KMeans cluster holdout with fixed seeds."""

    y = frame["y"].to_numpy(dtype=float)
    helms = frame["x"].astype(str).to_numpy()
    rows = []
    audits = []
    for split in make_nucleobase_splits(helms, y):
        train_groups = set(split.group_labels[split.train_indices].tolist())
        test_groups = set(split.group_labels[split.test_indices].tolist())
        exact_overlap = set(helms[split.train_indices]).intersection(helms[split.test_indices])
        audits.append(
            {
                "dataset": dataset,
                "split_mode": "oligogym_nucleobase_cluster_holdout",
                "repeat": split.split_id,
                "outer_fold": 1,
                "train_rows": len(split.train_indices),
                "test_rows": len(split.test_indices),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "group_overlap": len(train_groups.intersection(test_groups)),
                "exact_helm_overlap": len(exact_overlap),
            }
        )
        rows.extend(
            {
                "repeat": split.split_id,
                "outer_fold": 1,
                "row_index": int(index),
            }
            for index in split.test_indices
        )
    return pd.DataFrame(rows), pd.DataFrame(audits)


def atlas_assignments(
    task: str,
    frame: pd.DataFrame,
    out_dir: Path,
    split_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_mode == "patent":
        groups = frame["cv_group"].astype(str).to_numpy()
    else:
        groups, _ = similarity90_groups(frame, task, out_dir)
    rows = []
    audit_rows = []
    for fold, (_, test) in enumerate(
        GroupKFold(n_splits=5).split(frame, frame["y"], groups), start=1
    ):
        train = np.setdiff1d(np.arange(len(frame), dtype=int), test, assume_unique=True)
        overlap = set(groups[train]).intersection(groups[test])
        exact_overlap = set(frame.iloc[train]["HELM Annotation"]).intersection(
            frame.iloc[test]["HELM Annotation"]
        )
        audit_rows.append(
            {
                "dataset": task,
                "split_mode": split_mode,
                "outer_fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_groups": len(set(groups[train])),
                "test_groups": len(set(groups[test])),
                "group_overlap": len(overlap),
                "exact_helm_overlap": len(exact_overlap),
            }
        )
        rows.extend(
            {"repeat": 1, "outer_fold": fold, "row_index": int(index)}
            for index in test
        )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def run_fold(
    dataset: str,
    repeat: int,
    fold: int,
    test: np.ndarray,
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    device: str,
    checkpoint: Path,
    log_target: bool,
) -> Path:
    if checkpoint.exists():
        return checkpoint
    all_rows = np.arange(len(y), dtype=int)
    train = np.setdiff1d(all_rows, test, assume_unique=True)
    fit_y = np.log1p(y) if log_target else y
    seed = int(100_000 * repeat + 1_000 * (EARLIER + ATLAS).index(dataset) + fold)
    predictions = {
        name: fit_predict(matrix, fit_y, train, test, PROJECT_PARAMS, seed, device)
        for name, matrix in blocks.items()
        if name != "conditions"
    }
    predictions.update(
        {
            "descriptor": 0.5 * (predictions["branch_a"] + predictions["branch_b"]),
            "remove_position": 0.5 * (predictions["onehot"] + predictions["branch_b"]),
            "remove_core": 0.5
            * (predictions["branch_a"] + predictions["branch_b_no_core"]),
            "remove_context": 0.5
            * (predictions["branch_a"] + predictions["branch_b_no_context"]),
            "remove_seqchem": 0.5
            * (predictions["branch_a"] + predictions["branch_b_no_seqchem"]),
        }
    )
    if log_target:
        predictions = {
            name: np.maximum(np.expm1(values), 0.0)
            for name, values in predictions.items()
        }
    result_payload: dict[str, object] = {
        "dataset": dataset,
        "repeat": repeat,
        "outer_fold": fold,
        "row_index": test,
        "y": y[test],
    }
    result_payload.update({f"pred_{name}": values for name, values in predictions.items()})
    result = pd.DataFrame(result_payload)
    temporary = checkpoint.with_suffix(".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(checkpoint)
    return checkpoint


def summarize(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_rows = []
    routes = [column.removeprefix("pred_") for column in oof if column.startswith("pred_")]
    for (dataset, repeat), group in oof.groupby(["dataset", "repeat"]):
        for route in routes:
            metrics = regression_metrics(group.y.to_numpy(), group[f"pred_{route}"].to_numpy())
            repeat_rows.append({"dataset": dataset, "repeat": repeat, "route": route, **metrics})
    repeats = pd.DataFrame(repeat_rows)
    summary = (
        repeats.groupby(["dataset", "route"], as_index=False)
        .agg(
            repeats=("repeat", "nunique"),
            mean_spearman=("spearman", "mean"),
            sd_spearman=("spearman", "std"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
        )
    )
    return repeats, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--alt-target",
        choices=("identity", "log1p"),
        default="log1p",
        help="ALT fitting scale; FOB and OligoGym endpoints are unchanged.",
    )
    parser.add_argument("--datasets", nargs="+", choices=EARLIER + ATLAS, default=list(EARLIER + ATLAS))
    parser.add_argument("--earlier-splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--earlier-split",
        choices=("similarity90", "nucleobase"),
        default="similarity90",
    )
    parser.add_argument(
        "--atlas-split",
        choices=("similarity90", "patent"),
        default="similarity90",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out_dir / "fold_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    jobs = []
    dimensions = []
    split_audits = []
    started = time.time()
    for dataset in args.datasets:
        if dataset in EARLIER:
            frame = earlier_frame(dataset)
            if args.earlier_split == "nucleobase":
                assignments, split_audit = earlier_nucleobase_assignments(dataset, frame)
                split_audits.append(split_audit)
            else:
                assignments = earlier_assignments(dataset, args.earlier_splits)
        else:
            frame = load_task(dataset).reset_index(drop=True)
            assignments, split_audit = atlas_assignments(
                dataset, frame, args.out_dir, args.atlas_split
            )
            split_audits.append(split_audit)
        blocks, dims = matrices(frame)
        dimensions.append({"dataset": dataset, "rows": len(frame), **dims})
        y = frame["y"].to_numpy(dtype=float)
        fold_keys = assignments[["repeat", "outer_fold"]].drop_duplicates()
        if args.probe:
            fold_keys = fold_keys.iloc[:1]
        for key in fold_keys.itertuples(index=False):
            test = assignments[
                assignments.repeat.eq(key.repeat)
                & assignments.outer_fold.eq(key.outer_fold)
            ].row_index.to_numpy(dtype=int)
            checkpoint = checkpoint_dir / (
                f"{dataset}__{args.earlier_split if dataset in EARLIER else args.atlas_split}"
                f"__repeat{key.repeat}__fold{key.outer_fold}.csv"
            )
            jobs.append(
                delayed(run_fold)(
                    dataset,
                    int(key.repeat),
                    int(key.outer_fold),
                    test,
                    blocks,
                    y,
                    device=args.device,
                    checkpoint=checkpoint,
                    log_target=(
                        dataset in ("mouse_hepatic", "rat_hepatic")
                        and args.alt_target == "log1p"
                    ),
                )
            )
    paths = Parallel(n_jobs=args.parallel_jobs, backend="loky", verbose=10)(jobs)
    oof = pd.concat([pd.read_csv(path) for path in sorted(set(paths))], ignore_index=True)
    repeats, summary = summarize(oof)
    oof.to_csv(args.out_dir / "oof_predictions.csv.gz", index=False)
    repeats.to_csv(args.out_dir / "repeat_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    pd.DataFrame(dimensions).to_csv(args.out_dir / "matrix_dimensions.csv", index=False)
    if split_audits:
        pd.concat(split_audits, ignore_index=True).to_csv(
            args.out_dir / "split_audit.csv", index=False
        )
    manifest = {
        "schema_version": "registry_nine_endpoint_benchmark_v1",
        "status": "development_grouped_oof",
        "datasets": args.datasets,
        "probe": args.probe,
        "parallel_jobs": args.parallel_jobs,
        "device": args.device,
        "model_params": PROJECT_PARAMS,
        "onehot_definition": "registry-complete per-position base/sugar/linkage One-Hot",
        "pre_cet_onehot_definition": (
            "controlled 557D molecular One-Hot obtained only by removing the "
            "30 position-specific cEt sugar channels from the 587D registry schema"
        ),
        "registry_position_definition": (
            "registry-complete One-Hot plus named position-interaction descriptors"
        ),
        "descriptor_definition": "50:50 mean of registry-aware Raw+Position and Core+Context+SeqChem branches",
        "atlas_split_mode": args.atlas_split,
        "alt_target": args.alt_target,
        "oligogym_split_mode": args.earlier_split,
        "split_policy": (
            "OligoGym public k-mer/KMeans cluster holdout with fixed seeds 0..4"
            if args.earlier_split == "nucleobase"
            else "reuse earlier similarity90 assignments"
        )
        + (
            "; ASO Atlas 2 uses earliest-patent assignment per exact HELM"
            if args.atlas_split == "patent"
            else "; ASO Atlas 2 uses similarity90 components"
        ),
        "chemistry_registry": registry_manifest(),
        "universal_schema": schema_manifest(),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
