#!/usr/bin/env python3
"""Four-stage ASO Atlas 2 attribution benchmark on author patent folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_aso_atlas2_project_representation import (  # noqa: E402
    PROJECT_PARAMS,
    SOURCE_SPEARMAN,
    TASKS,
    author_onehot,
    chemistry_counts,
    experimental_conditions,
    fit_predict,
    fit_predict_author,
    load_task,
    project_features,
)


ROUTES = (
    "author_ohe_author_xgb",
    "author_ohe_project_xgb",
    "registry_ohe_project_xgb",
    "registry_descriptor_project_xgb",
)


def matrices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    helms = frame["x"].astype(str).tolist()
    conditions = experimental_conditions(frame)
    author = np.column_stack(
        [author_onehot(helms), chemistry_counts(helms), conditions]
    ).astype(np.float32)
    project = project_features(frame)
    registry_ohe = np.column_stack([project["raw"], conditions]).astype(np.float32)
    branch_a = np.column_stack([project["branch_a"], conditions]).astype(np.float32)
    branch_b = np.column_stack([project["branch_b"], conditions]).astype(np.float32)
    return {
        "author": author,
        "registry_ohe": registry_ohe,
        "branch_a": branch_a,
        "branch_b": branch_b,
    }


def assignments(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = frame["cv_group"].astype(str).to_numpy()
    rows = []
    audits = []
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=5).split(frame, frame["y"], groups), start=1
    ):
        rows.extend(
            {"outer_fold": fold, "row_index": int(index)} for index in test
        )
        audits.append(
            {
                "outer_fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_patent_groups": len(set(groups[train])),
                "test_patent_groups": len(set(groups[test])),
                "patent_group_overlap": len(set(groups[train]).intersection(groups[test])),
                "exact_helm_overlap": len(
                    set(frame.iloc[train]["HELM Annotation"]).intersection(
                        frame.iloc[test]["HELM Annotation"]
                    )
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audits)


def run_fold(
    task: str,
    fold: int,
    test: np.ndarray,
    matrix: dict[str, np.ndarray],
    y: np.ndarray,
    device: str,
    checkpoint: Path,
    alt_target: str,
) -> Path:
    if checkpoint.exists():
        return checkpoint
    train = np.setdiff1d(np.arange(len(y), dtype=int), test, assume_unique=True)
    seed = 1000 * TASKS.index(task) + fold
    use_log_target = task.endswith("hepatic") and alt_target == "log1p"
    fit_y = np.log1p(y) if use_log_target else y
    author_author = fit_predict_author(matrix["author"], fit_y, train, test)
    author_project = fit_predict(
        matrix["author"], fit_y, train, test, PROJECT_PARAMS, seed, device
    )
    registry_ohe = fit_predict(
        matrix["registry_ohe"], fit_y, train, test, PROJECT_PARAMS, seed, device
    )
    branch_a = fit_predict(
        matrix["branch_a"], fit_y, train, test, PROJECT_PARAMS, seed, device
    )
    branch_b = fit_predict(
        matrix["branch_b"], fit_y, train, test, PROJECT_PARAMS, seed, device
    )
    predictions = [author_author, author_project, registry_ohe, 0.5 * (branch_a + branch_b)]
    if use_log_target:
        predictions = [np.maximum(np.expm1(prediction), 0.0) for prediction in predictions]
    result = pd.DataFrame(
        {
            "task": task,
            "outer_fold": fold,
            "row_index": test,
            "y": y[test],
            "target_transform": "log1p" if use_log_target else "identity",
            "pred_author_ohe_author_xgb": predictions[0],
            "pred_author_ohe_project_xgb": predictions[1],
            "pred_registry_ohe_project_xgb": predictions[2],
            "pred_registry_descriptor_project_xgb": predictions[3],
        }
    )
    temporary = checkpoint.with_suffix(".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(checkpoint)
    return checkpoint


def summarize(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (task, fold), group in oof.groupby(["task", "outer_fold"]):
        for route in ROUTES:
            prediction = group[f"pred_{route}"].to_numpy()
            rows.append(
                {
                    "task": task,
                    "outer_fold": fold,
                    "route": route,
                    "spearman": float(spearmanr(group["y"], prediction).statistic),
                    "r2": float(r2_score(group["y"], prediction)),
                    "log1p_r2": float(
                        r2_score(
                            np.log1p(group["y"].to_numpy()),
                            np.log1p(np.maximum(prediction, 0.0)),
                        )
                    ),
                }
            )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["task", "route"], as_index=False)
        .agg(
            folds=("outer_fold", "nunique"),
            mean_spearman=("spearman", "mean"),
            sd_spearman=("spearman", "std"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_log1p_r2=("log1p_r2", "mean"),
            sd_log1p_r2=("log1p_r2", "std"),
        )
    )
    summary["source_reported_spearman"] = summary["task"].map(
        lambda task: SOURCE_SPEARMAN[task][0]
    )
    return folds, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument(
        "--alt-target", choices=("identity", "log1p"), default="identity"
    )
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.out_dir / "fold_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    jobs = []
    audits = []
    dimensions = []
    started = time.time()
    for task in args.tasks:
        frame = load_task(task).reset_index(drop=True)
        matrix = matrices(frame)
        assignment, audit = assignments(frame)
        audit.insert(0, "task", task)
        audits.append(audit)
        dimensions.append(
            {
                "task": task,
                "rows": len(frame),
                "author_ohe": matrix["author"].shape[1],
                "registry_ohe": matrix["registry_ohe"].shape[1],
                "registry_branch_a": matrix["branch_a"].shape[1],
                "registry_branch_b": matrix["branch_b"].shape[1],
            }
        )
        fold_ids = sorted(assignment.outer_fold.unique())
        if args.probe:
            fold_ids = fold_ids[:1]
        for fold in fold_ids:
            test = assignment[assignment.outer_fold.eq(fold)].row_index.to_numpy(dtype=int)
            jobs.append(
                delayed(run_fold)(
                    task,
                    int(fold),
                    test,
                    matrix,
                    frame["y"].to_numpy(dtype=float),
                    args.device,
                    checkpoints / f"{task}__fold{fold}.csv",
                    args.alt_target,
                )
            )
    paths = Parallel(n_jobs=args.parallel_jobs, backend="loky", verbose=10)(jobs)
    oof = pd.concat([pd.read_csv(path) for path in sorted(set(paths))], ignore_index=True)
    folds, summary = summarize(oof)
    oof.to_csv(args.out_dir / "oof_predictions.csv.gz", index=False)
    folds.to_csv(args.out_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    pd.concat(audits, ignore_index=True).to_csv(args.out_dir / "split_audit.csv", index=False)
    pd.DataFrame(dimensions).to_csv(args.out_dir / "matrix_dimensions.csv", index=False)
    manifest = {
        "schema_version": "aso_atlas2_four_stage_factorial_v1",
        "split": "author earliest-patent GroupKFold",
        "target": (
            "log1p ALT for fitting with expm1 predictions; raw FOB"
            if args.alt_target == "log1p"
            else "author raw ALT/FOB target; no log transform"
        ),
        "alt_target": args.alt_target,
        "routes": list(ROUTES),
        "device": args.device,
        "parallel_jobs": args.parallel_jobs,
        "probe": args.probe,
        "project_xgb_params": PROJECT_PARAMS,
        "author_xgb_contract": "XGBRegressor(n_estimators=500, max_depth=10) with package defaults",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
