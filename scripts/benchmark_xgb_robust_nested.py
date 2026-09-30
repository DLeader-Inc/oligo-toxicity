#!/usr/bin/env python3
"""Strict nested joint target-scale/capacity selection for the two-branch system."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_registry_nine_endpoints import (  # noqa: E402
    ATLAS,
    EARLIER,
    atlas_assignments,
    earlier_frame,
    matrices,
)
from scripts.benchmark_aso_atlas2_project_representation import load_task  # noqa: E402
from src.benchmarking.oligogym_same_split import make_nucleobase_splits  # noqa: E402


CANDIDATES: tuple[dict[str, object], ...] = (
    {
        "name": "compact_regularized",
        "complexity_rank": 1,
        "params": {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "reg_alpha": 0.1,
        },
    },
    {
        "name": "conservative_shallow",
        "complexity_rank": 2,
        "params": {
            "n_estimators": 600,
            "max_depth": 3,
            "learning_rate": 0.02,
            "min_child_weight": 8,
            "subsample": 0.75,
            "colsample_bytree": 0.7,
            "reg_lambda": 8.0,
            "reg_alpha": 0.5,
        },
    },
    {
        "name": "current_balanced",
        "complexity_rank": 3,
        "params": {
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.02,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "reg_alpha": 0.1,
        },
    },
    {
        "name": "deep_regularized_control",
        "complexity_rank": 4,
        "params": {
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.02,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 5.0,
            "reg_alpha": 0.2,
        },
    },
)
ROUTES = (
    "fixed_current_single",
    "fixed_current_seed_mean",
    "fixed_compact_seed_mean",
    "nested_one_se_seed_mean",
    "fixed_current_identity",
    "fixed_current_log1p",
)
TRANSFORMS = ("identity", "log1p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def _model(params: dict[str, object], seed: int, device: str) -> XGBRegressor:
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",
        device=device,
        n_jobs=1,
        verbosity=0,
    )


def _fit_branch(
    matrix: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    params: dict[str, object],
    seed: int,
    device: str,
) -> np.ndarray:
    estimator = _model(params, seed, device)
    estimator.fit(matrix[train], y[train])
    return np.asarray(estimator.predict(matrix[test]), dtype=float)


def _fit_two_branch(
    branch_a: np.ndarray,
    branch_b: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    params: dict[str, object],
    seed: int,
    device: str,
) -> np.ndarray:
    pred_a = _fit_branch(branch_a, y, train, test, params, seed, device)
    pred_b = _fit_branch(branch_b, y, train, test, params, seed + 499, device)
    return 0.5 * (pred_a + pred_b)


def _safe_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    value = float(r2_score(y, prediction))
    return value if np.isfinite(value) else -1.0


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    spearman = float(spearmanr(y, prediction).statistic)
    if not np.isfinite(spearman):
        spearman = -1.0
    y_std = float(np.std(y, ddof=0))
    rmse = float(np.sqrt(mean_squared_error(y, prediction)))
    return {
        "spearman": spearman,
        "r2": _safe_r2(y, prediction),
        "rmse": rmse,
        "y_std": y_std,
        "rmse_over_y_std": rmse / max(y_std, np.finfo(float).eps),
    }


def _checkpoint_signature(
    dataset: str,
    partition: dict[str, object],
    branch_a: np.ndarray,
    branch_b: np.ndarray,
    y_raw: np.ndarray,
    *,
    inner_folds: int,
    seed_count: int,
    target_policy: str,
) -> str:
    """Hash every input/policy element that makes a fold checkpoint reusable."""

    digest = hashlib.sha256()
    payload = {
        "dataset": dataset,
        "partition": int(partition["partition"]),
        "inner_folds": inner_folds,
        "seed_count": seed_count,
        "target_policy": target_policy,
        "candidates": CANDIDATES,
        "script_sha256": _sha256(Path(__file__)),
    }
    digest.update(json.dumps(payload, sort_keys=True).encode())
    for values in (
        np.asarray(partition["train"], dtype=np.int64),
        np.asarray(partition["test"], dtype=np.int64),
        np.asarray(y_raw, dtype=np.float64),
        np.asarray(branch_a, dtype=np.float32),
        np.asarray(branch_b, dtype=np.float32),
    ):
        digest.update(values.tobytes())
    digest.update("\0".join(np.asarray(partition["groups"]).astype(str)).encode())
    return digest.hexdigest()


def _selection_score(metrics: dict[str, float]) -> float:
    return 0.5 * metrics["spearman"] + 0.5 * np.clip(metrics["r2"], -1.0, 1.0)


def allowed_transforms(y: np.ndarray) -> tuple[str, ...]:
    """Return transforms legal for a training partition."""

    values = np.asarray(y, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("target contains non-finite values")
    return TRANSFORMS if np.min(values) >= 0.0 else ("identity",)


def transform_target(y: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if transform == "identity":
        return values
    if transform == "log1p":
        if np.min(values) < 0.0:
            raise ValueError("log1p target transform requires nonnegative targets")
        return np.log1p(values)
    raise ValueError(f"unsupported target transform: {transform}")


def inverse_target(prediction: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(prediction, dtype=float)
    if transform == "identity":
        return values
    if transform == "log1p":
        return np.expm1(values)
    raise ValueError(f"unsupported target transform: {transform}")


def select_one_standard_error(
    aggregate: pd.DataFrame,
) -> tuple[pd.Series, float, pd.DataFrame]:
    """Choose the simplest candidate statistically tied with the inner-CV best."""

    required = {
        "candidate",
        "complexity_rank",
        "transform",
        "transform_rank",
        "mean_score",
        "se_score",
    }
    missing = required.difference(aggregate.columns)
    if missing:
        raise ValueError(f"missing selection columns: {sorted(missing)}")
    if aggregate.empty:
        raise ValueError("cannot select from an empty candidate table")
    best = aggregate.sort_values(
        ["mean_score", "complexity_rank", "transform_rank"],
        ascending=[False, True, True],
    ).iloc[0]
    threshold = float(best.mean_score - best.se_score)
    eligible = aggregate[aggregate.mean_score.ge(threshold)].sort_values(
        ["complexity_rank", "transform_rank", "mean_score"],
        ascending=[True, True, False],
    )
    return best, threshold, eligible


def _outer_partitions(
    dataset: str, frame: pd.DataFrame
) -> list[dict[str, object]]:
    y = frame["y"].to_numpy(dtype=float)
    if dataset in EARLIER:
        helms = frame["x"].astype(str).to_numpy()
        return [
            {
                "partition": split.split_id,
                "train": split.train_indices,
                "test": split.test_indices,
                "groups": split.group_labels.astype(str),
                "split_mode": "oligogym_nucleobase_cluster_holdout",
            }
            for split in make_nucleobase_splits(helms, y)
        ]
    groups = frame["cv_group"].astype(str).to_numpy()
    result = []
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=5).split(frame, y, groups), start=1
    ):
        result.append(
            {
                "partition": fold,
                "train": np.asarray(train),
                "test": np.asarray(test),
                "groups": groups,
                "split_mode": "aso_atlas2_earliest_patent_groupkfold",
            }
        )
    return result


def _seed_values(dataset: str, partition: int, count: int) -> tuple[int, ...]:
    base = 100_003 + 10_007 * (EARLIER + ATLAS).index(dataset) + 101 * partition
    return tuple(base + 7_919 * index for index in range(count))


def _run_partition(
    dataset: str,
    partition: dict[str, object],
    branch_a: np.ndarray,
    branch_b: np.ndarray,
    y_raw: np.ndarray,
    *,
    inner_folds: int,
    seed_count: int,
    device: str,
    target_policy: str,
    checkpoint_dir: Path,
) -> tuple[Path, Path, Path]:
    number = int(partition["partition"])
    prediction_path = checkpoint_dir / f"{dataset}__partition{number}__predictions.csv"
    inner_path = checkpoint_dir / f"{dataset}__partition{number}__inner_scores.csv"
    selection_path = checkpoint_dir / f"{dataset}__partition{number}__selection.json"
    resume_signature = _checkpoint_signature(
        dataset, partition, branch_a, branch_b, y_raw,
        inner_folds=inner_folds, seed_count=seed_count, target_policy=target_policy,
    )
    if prediction_path.exists() and inner_path.exists() and selection_path.exists():
        previous = json.loads(selection_path.read_text())
        if previous.get("resume_signature") == resume_signature:
            return prediction_path, inner_path, selection_path

    outer_train = np.asarray(partition["train"], dtype=int)
    outer_test = np.asarray(partition["test"], dtype=int)
    all_groups = np.asarray(partition["groups"]).astype(str)
    train_groups = all_groups[outer_train]
    if set(train_groups).intersection(all_groups[outer_test]):
        raise RuntimeError(f"outer group leakage: {dataset}/{number}")
    unique_train_groups = np.unique(train_groups)
    n_inner = min(inner_folds, len(unique_train_groups))
    if n_inner < 2:
        raise RuntimeError(f"too few outer-train groups: {dataset}/{number}")
    splitter = GroupKFold(n_splits=n_inner)
    inner_rows: list[dict[str, object]] = []
    transforms = allowed_transforms(y_raw[outer_train])
    if target_policy != "auto":
        if target_policy not in transforms:
            raise ValueError(f"illegal target policy {target_policy} for {dataset}/{number}")
        transforms = (target_policy,)
    for transform in transforms:
      fit_y = np.full_like(y_raw, np.nan, dtype=float)
      fit_y[outer_train] = transform_target(y_raw[outer_train], transform)
      for candidate in CANDIDATES:
        params = dict(candidate["params"])
        for inner_fold, (inner_train_local, inner_valid_local) in enumerate(
            splitter.split(outer_train, y_raw[outer_train], train_groups), start=1
        ):
            inner_train = outer_train[inner_train_local]
            inner_valid = outer_train[inner_valid_local]
            if set(all_groups[inner_train]).intersection(all_groups[inner_valid]):
                raise RuntimeError("inner group leakage")
            seed = _seed_values(dataset, number * 10 + inner_fold, 1)[0]
            prediction = _fit_two_branch(
                branch_a,
                branch_b,
                fit_y,
                inner_train,
                inner_valid,
                params,
                seed,
                device,
            )
            prediction_raw = inverse_target(prediction, transform)
            metric = _metrics(y_raw[inner_valid], prediction_raw)
            model_scale_r2 = _safe_r2(fit_y[inner_valid], prediction)
            inner_rows.append(
                {
                    "dataset": dataset,
                    "outer_partition": number,
                    "inner_fold": inner_fold,
                    "candidate": candidate["name"],
                    "complexity_rank": candidate["complexity_rank"],
                    "transform": transform,
                    "transform_rank": 0 if transform == "identity" else 1,
                    "train_rows": len(inner_train),
                    "valid_rows": len(inner_valid),
                    "train_groups": len(np.unique(all_groups[inner_train])),
                    "valid_groups": len(np.unique(all_groups[inner_valid])),
                    "group_overlap": 0,
                    **metric,
                    "model_scale_r2": model_scale_r2,
                    "selection_score": _selection_score(metric),
                }
            )
    inner = pd.DataFrame(inner_rows)
    aggregate = (
        inner.groupby(
            ["candidate", "complexity_rank", "transform", "transform_rank"],
            as_index=False,
        )
        .agg(
            mean_score=("selection_score", "mean"),
            sd_score=("selection_score", "std"),
            folds=("inner_fold", "nunique"),
            mean_spearman=("spearman", "mean"),
            mean_r2=("r2", "mean"),
            mean_model_scale_r2=("model_scale_r2", "mean"),
        )
    )
    aggregate["se_score"] = aggregate.sd_score.fillna(0.0) / np.sqrt(aggregate.folds)
    best, threshold, eligible = select_one_standard_error(aggregate)
    selected_name = str(eligible.iloc[0].candidate)
    selected_transform = str(eligible.iloc[0]["transform"])
    selected = next(item for item in CANDIDATES if item["name"] == selected_name)
    fit_y = np.full_like(y_raw, np.nan, dtype=float)
    fit_y[outer_train] = transform_target(y_raw[outer_train], selected_transform)

    seeds = _seed_values(dataset, number, seed_count)
    current = next(item for item in CANDIDATES if item["name"] == "current_balanced")
    compact = next(item for item in CANDIDATES if item["name"] == "compact_regularized")

    ensemble_cache: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}

    def ensemble(candidate: dict[str, object]) -> tuple[np.ndarray, list[np.ndarray]]:
        name = str(candidate["name"])
        if name in ensemble_cache:
            return ensemble_cache[name]
        members = [
            _fit_two_branch(
                branch_a,
                branch_b,
                fit_y,
                outer_train,
                outer_test,
                dict(candidate["params"]),
                seed,
                device,
            )
            for seed in seeds
        ]
        result = (np.mean(members, axis=0), members)
        ensemble_cache[name] = result
        return result

    current_mean, current_members = ensemble(current)
    compact_mean, _ = ensemble(compact)
    selected_mean, selected_members = ensemble(selected)
    predictions = {
        "fixed_current_single": current_members[0],
        "fixed_current_seed_mean": current_mean,
        "fixed_compact_seed_mean": compact_mean,
        "nested_one_se_seed_mean": selected_mean,
    }
    predictions = {
        name: inverse_target(values, selected_transform)
        for name, values in predictions.items()
    }
    explicit_transform_predictions: dict[str, np.ndarray] = {}
    for explicit_transform in allowed_transforms(y_raw[outer_train]):
        explicit_fit_y = np.full_like(y_raw, np.nan, dtype=float)
        explicit_fit_y[outer_train] = transform_target(
            y_raw[outer_train], explicit_transform
        )
        explicit_members = [
            _fit_two_branch(
                branch_a,
                branch_b,
                explicit_fit_y,
                outer_train,
                outer_test,
                dict(current["params"]),
                seed,
                device,
            )
            for seed in seeds
        ]
        explicit_transform_predictions[f"fixed_current_{explicit_transform}"] = (
            inverse_target(np.mean(explicit_members, axis=0), explicit_transform)
        )
    predictions.update(explicit_transform_predictions)
    selected_members_eval = [
        inverse_target(values, selected_transform) for values in selected_members
    ]
    member_sd = np.std(np.column_stack(selected_members_eval), axis=1, ddof=0)
    output = pd.DataFrame(
        {
            "dataset": dataset,
            "split_mode": partition["split_mode"],
            "outer_partition": number,
            "row_index": outer_test,
            "y": y_raw[outer_test],
            "selected_candidate": selected_name,
            "selected_transform": selected_transform,
            "selected_complexity_rank": selected["complexity_rank"],
            "selected_seed_member_sd": member_sd,
            **{f"pred_{name}": values for name, values in predictions.items()},
        }
    )
    inner.to_csv(inner_path.with_suffix(".tmp"), index=False)
    inner_path.with_suffix(".tmp").replace(inner_path)
    output.to_csv(prediction_path.with_suffix(".tmp"), index=False)
    prediction_path.with_suffix(".tmp").replace(prediction_path)
    selection_payload = {
        "dataset": dataset,
        "outer_partition": number,
        "split_mode": partition["split_mode"],
        "outer_train_rows": len(outer_train),
        "outer_test_rows": len(outer_test),
        "outer_train_groups": len(unique_train_groups),
        "outer_test_groups": len(np.unique(all_groups[outer_test])),
        "outer_group_overlap": 0,
        "inner_folds": n_inner,
        "best_mean_candidate": str(best.candidate),
        "best_mean_transform": str(best["transform"]),
        "best_mean_score": float(best.mean_score),
        "best_mean_se": float(best.se_score),
        "one_se_threshold": threshold,
        "eligible_routes": [
            f"{row.transform}::{row.candidate}" for row in eligible.itertuples()
        ],
        "selected_candidate": selected_name,
        "selected_transform": selected_transform,
        "inverse_transform": "identity" if selected_transform == "identity" else "expm1",
        "model_output_scale": "y" if selected_transform == "identity" else "log1p(y)",
        "display_scale": "original y scale",
        "treeshap_scale": "y" if selected_transform == "identity" else "log1p(y)",
        "selected_xgboost_params": selected["params"],
        "selected_complexity_rank": int(selected["complexity_rank"]),
        "seed_values": list(seeds),
        "selection_uses_outer_test": False,
        "resume_signature": resume_signature,
    }
    selection_tmp = selection_path.with_suffix(".tmp")
    selection_tmp.write_text(json.dumps(selection_payload, indent=2) + "\n")
    selection_tmp.replace(selection_path)
    return prediction_path, inner_path, selection_path


def _fit_deployment(
    dataset: str,
    branch_a: np.ndarray,
    branch_b: np.ndarray,
    y_raw: np.ndarray,
    groups: np.ndarray,
    *,
    inner_folds: int,
    seed_count: int,
    device: str,
    target_policy: str,
    out_dir: Path,
) -> dict[str, object]:
    """Repeat grouped selection on all rows and persist deployable branch models."""

    model_dir = out_dir / "deployment_models"
    model_dir.mkdir(exist_ok=True)
    manifest_path = model_dir / f"{dataset}__manifest.json"
    deployment_digest = hashlib.sha256()
    deployment_digest.update(json.dumps({
        "dataset": dataset, "inner_folds": inner_folds, "seed_count": seed_count,
        "target_policy": target_policy, "candidates": CANDIDATES,
        "script_sha256": _sha256(Path(__file__)),
    }, sort_keys=True).encode())
    for values in (
        np.asarray(branch_a, dtype=np.float32), np.asarray(branch_b, dtype=np.float32),
        np.asarray(y_raw, dtype=np.float64),
    ):
        deployment_digest.update(values.tobytes())
    deployment_digest.update("\0".join(np.asarray(groups).astype(str)).encode())
    resume_signature = deployment_digest.hexdigest()
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("resume_signature") == resume_signature and all(
            Path(model["path"]).exists() for model in previous.get("models", [])
        ):
            return previous
    rows = np.arange(len(y_raw), dtype=int)
    unique_groups = np.unique(groups)
    n_inner = min(inner_folds, len(unique_groups))
    if n_inner < 2:
        raise RuntimeError(f"too few deployment groups: {dataset}")
    transforms = allowed_transforms(y_raw)
    if target_policy != "auto":
        transforms = (target_policy,)
    splitter = GroupKFold(n_splits=n_inner)
    score_rows: list[dict[str, object]] = []
    for transform in transforms:
        fit_y = transform_target(y_raw, transform)
        for candidate in CANDIDATES:
            for fold, (train, valid) in enumerate(
                splitter.split(rows, y_raw, groups), start=1
            ):
                prediction = _fit_two_branch(
                    branch_a, branch_b, fit_y, train, valid,
                    dict(candidate["params"]), _seed_values(dataset, 900 + fold, 1)[0], device,
                )
                raw_prediction = inverse_target(prediction, transform)
                metric = _metrics(y_raw[valid], raw_prediction)
                score_rows.append({
                    "candidate": candidate["name"],
                    "complexity_rank": candidate["complexity_rank"],
                    "transform": transform,
                    "transform_rank": 0 if transform == "identity" else 1,
                    "inner_fold": fold,
                    "selection_score": _selection_score(metric),
                    **metric,
                    "model_scale_r2": _safe_r2(fit_y[valid], prediction),
                    "train_groups": len(np.unique(groups[train])),
                    "valid_groups": len(np.unique(groups[valid])),
                    "group_overlap": len(set(groups[train]).intersection(groups[valid])),
                })
    scores = pd.DataFrame(score_rows)
    aggregate = scores.groupby(
        ["candidate", "complexity_rank", "transform", "transform_rank"], as_index=False
    ).agg(
        mean_score=("selection_score", "mean"),
        sd_score=("selection_score", "std"),
        folds=("inner_fold", "nunique"),
        mean_spearman=("spearman", "mean"),
        mean_r2=("r2", "mean"),
        mean_model_scale_r2=("model_scale_r2", "mean"),
    )
    aggregate["se_score"] = aggregate.sd_score.fillna(0.0) / np.sqrt(aggregate.folds)
    best, threshold, eligible = select_one_standard_error(aggregate)
    choice = eligible.iloc[0]
    selected = next(item for item in CANDIDATES if item["name"] == choice.candidate)
    transform = str(choice["transform"])
    fit_y = transform_target(y_raw, transform)
    seeds = _seed_values(dataset, 999, seed_count)
    model_paths: list[dict[str, object]] = []
    reconciliations: list[dict[str, object]] = []
    for member, seed in enumerate(seeds, start=1):
        branch_models = []
        for branch_name, matrix, branch_seed in (
            ("branch_a", branch_a, seed), ("branch_b", branch_b, seed + 499)
        ):
            estimator = _model(dict(selected["params"]), branch_seed, device)
            estimator.fit(matrix, fit_y)
            path = model_dir / f"{dataset}__member{member}__{branch_name}.joblib"
            joblib.dump(estimator, path)
            branch_models.append((branch_name, matrix, estimator, path))
            contribution = estimator.get_booster().predict(
                __import__("xgboost").DMatrix(matrix[:1]), pred_contribs=True
            )[0]
            prediction = float(estimator.predict(matrix[:1])[0])
            reconciliations.append({
                "dataset": dataset,
                "member": member,
                "branch": branch_name,
                "model_scale": "y" if transform == "identity" else "log1p(y)",
                "prediction_model_scale": prediction,
                "treeshap_sum_model_scale": float(contribution.sum()),
                "absolute_residual": abs(prediction - float(contribution.sum())),
            })
            model_paths.append({"member": member, "branch": branch_name, "path": str(path.resolve())})
        pred_a = float(branch_models[0][2].predict(branch_a[:1])[0])
        pred_b = float(branch_models[1][2].predict(branch_b[:1])[0])
        fused_model = 0.5 * (pred_a + pred_b)
        reconciliations.append({
            "dataset": dataset, "member": member, "branch": "fixed_50_50_fusion",
            "model_scale": "y" if transform == "identity" else "log1p(y)",
            "prediction_model_scale": fused_model,
            "treeshap_sum_model_scale": 0.5 * (
                reconciliations[-2]["treeshap_sum_model_scale"]
                + reconciliations[-1]["treeshap_sum_model_scale"]
            ),
            "absolute_residual": abs(
                fused_model - 0.5 * (
                    reconciliations[-2]["treeshap_sum_model_scale"]
                    + reconciliations[-1]["treeshap_sum_model_scale"]
                )
            ),
        })
    scores.to_csv(model_dir / f"{dataset}__inner_scores.csv", index=False)
    payload = {
        "dataset": dataset,
        "training_rows": len(y_raw),
        "training_groups": len(unique_groups),
        "selection_uses_external_or_outer_test": False,
        "selected_transform": transform,
        "inverse_transform": "identity" if transform == "identity" else "expm1",
        "model_output_scale": "y" if transform == "identity" else "log1p(y)",
        "treeshap_scale": "y" if transform == "identity" else "log1p(y)",
        "original_scale_attribution_note": (
            "additive" if transform == "identity" else
            "expm1 display is not claimed to preserve strict additive SHAP attribution"
        ),
        "selected_candidate": str(choice.candidate),
        "selected_xgboost_params": selected["params"],
        "best_mean_route": f"{best['transform']}::{best.candidate}",
        "one_se_threshold": threshold,
        "eligible_routes": [f"{r.transform}::{r.candidate}" for r in eligible.itertuples()],
        "seed_values": list(seeds),
        "fixed_fusion_weights": {"branch_a": 0.5, "branch_b": 0.5},
        "resume_signature": resume_signature,
        "models": model_paths,
        "max_treeshap_reconciliation_residual": max(r["absolute_residual"] for r in reconciliations),
    }
    (model_dir / f"{dataset}__reconciliation.csv").write_text(
        pd.DataFrame(reconciliations).to_csv(index=False), encoding="utf-8"
    )
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(manifest_path)
    return payload


def _summarize(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (dataset, partition), group in oof.groupby(["dataset", "outer_partition"]):
        for route in ROUTES:
            prediction = group[f"pred_{route}"].to_numpy(dtype=float)
            metric = _metrics(group.y.to_numpy(dtype=float), prediction)
            rows.append(
                {
                    "dataset": dataset,
                    "outer_partition": partition,
                    "route": route,
                    "test_rows": len(group),
                    **metric,
                    "log1p_r2": (
                        _safe_r2(
                            np.log1p(group.y.to_numpy(dtype=float)),
                            np.log1p(prediction),
                        )
                        if np.min(group.y.to_numpy(dtype=float)) >= 0.0
                        and np.min(prediction) > -1.0
                        else np.nan
                    ),
                }
            )
    partitions = pd.DataFrame(rows)
    summary = (
        partitions.groupby(["dataset", "route"], as_index=False)
        .agg(
            partitions=("outer_partition", "nunique"),
            mean_spearman=("spearman", "mean"),
            sd_spearman=("spearman", "std"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
            mean_y_std=("y_std", "mean"),
            mean_rmse_over_y_std=("rmse_over_y_std", "mean"),
            mean_log1p_r2=("log1p_r2", "mean"),
        )
    )
    return partitions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=EARLIER + ATLAS, default=list(EARLIER + ATLAS)
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument(
        "--target-policy",
        choices=("auto", "identity", "log1p"),
        default="auto",
        help="auto jointly selects target scale and capacity inside grouped inner CV",
    )
    parser.add_argument(
        "--alt-target",
        choices=("identity", "log1p"),
        default=None,
        help="deprecated compatibility alias; overrides --target-policy when supplied",
    )
    parser.add_argument("--skip-deployment", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.alt_target is not None:
        args.target_policy = args.alt_target
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out_dir / "fold_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    started = time.time()
    prediction_paths: list[Path] = []
    inner_paths: list[Path] = []
    selection_paths: list[Path] = []
    dimensions = []
    deployment_inputs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for dataset in args.datasets:
        frame = earlier_frame(dataset) if dataset in EARLIER else load_task(dataset).reset_index(drop=True)
        blocks, dims = matrices(frame)
        dimensions.append({"dataset": dataset, "rows": len(frame), **dims})
        partitions = _outer_partitions(dataset, frame)
        deployment_inputs.append(
            (
                dataset,
                blocks["branch_a"],
                blocks["branch_b"],
                frame["y"].to_numpy(dtype=float),
                np.asarray(partitions[0]["groups"]).astype(str),
            )
        )
        if args.probe:
            partitions = partitions[:1]
        results = Parallel(n_jobs=args.parallel_jobs, backend="loky", verbose=10)(
            delayed(_run_partition)(
                dataset,
                partition,
                blocks["branch_a"],
                blocks["branch_b"],
                frame["y"].to_numpy(dtype=float),
                inner_folds=args.inner_folds,
                seed_count=args.seed_count,
                device=args.device,
                target_policy=args.target_policy,
                checkpoint_dir=checkpoint_dir,
            )
            for partition in partitions
        )
        for prediction_path, inner_path, selection_path in results:
            prediction_paths.append(prediction_path)
            inner_paths.append(inner_path)
            selection_paths.append(selection_path)

    oof = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True)
    inner = pd.concat([pd.read_csv(path) for path in inner_paths], ignore_index=True)
    selections = pd.DataFrame([json.loads(path.read_text()) for path in selection_paths])
    partitions, summary = _summarize(oof)
    oof.to_csv(args.out_dir / "oof_predictions.csv.gz", index=False)
    inner.to_csv(args.out_dir / "inner_scores.csv.gz", index=False)
    selections.to_csv(args.out_dir / "selections.csv", index=False)
    partitions.to_csv(args.out_dir / "partition_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    pd.DataFrame(dimensions).to_csv(args.out_dir / "matrix_dimensions.csv", index=False)
    deployment_manifests = []
    if not args.probe and not args.skip_deployment:
        for dataset, branch_a, branch_b, y_raw, groups in deployment_inputs:
            deployment_manifests.append(
                _fit_deployment(
                    dataset,
                    branch_a,
                    branch_b,
                    y_raw,
                    groups,
                    inner_folds=args.inner_folds,
                    seed_count=args.seed_count,
                    device=args.device,
                    target_policy=args.target_policy,
                    out_dir=args.out_dir,
                )
            )
        pd.DataFrame(
            [
                {
                    "dataset": item["dataset"],
                    "training_rows": item["training_rows"],
                    "training_groups": item["training_groups"],
                    "selected_transform": item["selected_transform"],
                    "selected_candidate": item["selected_candidate"],
                    "model_output_scale": item["model_output_scale"],
                    "inverse_transform": item["inverse_transform"],
                    "treeshap_scale": item["treeshap_scale"],
                    "max_treeshap_reconciliation_residual": item[
                        "max_treeshap_reconciliation_residual"
                    ],
                }
                for item in deployment_manifests
            ]
        ).to_csv(args.out_dir / "deployment_summary.csv", index=False)
    manifest = {
        "schema_version": "integrated_auto_target_xgb_strict_nested_v2",
        "datasets": args.datasets,
        "outer_protocol": {
            "oligogym": "base 1/2/3-mer KMeans groups, fixed seeds 0..4, 80:20 GroupShuffleSplit",
            "aso_atlas2": "earliest-patent five-fold GroupKFold",
        },
        "inner_protocol": f"GroupKFold({args.inner_folds}) on outer-training groups only",
        "selection_rule": "one-standard-error rule; choose lowest complexity rank within best mean minus best SE",
        "selection_score": "0.5 * Spearman + 0.5 * clip(raw-scale R2, -1, 1), after inverse transform",
        "seed_ensemble": args.seed_count,
        "candidates": CANDIDATES,
        "routes": ROUTES,
        "target_policy": args.target_policy,
        "candidate_transforms": list(TRANSFORMS),
        "nonnegative_target_rule": "identity and log1p are eligible; predictions are inverse-transformed before inner scoring",
        "negative_target_rule": "identity only",
        "one_se_tie_break": "lower model complexity first, then identity",
        "inverse_transform": {"identity": "identity", "log1p": "expm1"},
        "treeshap_contract": "TreeSHAP remains on the fitted model scale; expm1 display is not claimed as strictly additive attribution",
        "deployment_selection_repeated_on_all_training_rows": bool(
            not args.probe and not args.skip_deployment
        ),
        "source_repository": str(ROOT.resolve()),
        "source_commit": _git_commit(),
        "training_entry_point": str(Path(__file__).resolve()),
        "training_entry_point_sha256": _sha256(Path(__file__).resolve()),
        "external_task_checkpoint_usage": "none",
        "fold_checkpoint_usage": "atomic per-partition CSV/JSON; complete partitions reused on resume",
        "local_compatibility_patches": "none required for XGBoost route",
        "intentional_deviations": [
            "one seeded member in this formal automatic-system run; target scale and capacity remain strictly inner-selected",
            "source-native models are contextual comparison artifacts and are not fitted by this entry point",
        ],
        "leakage_safeguards": {
            "outer_group_overlap_required": 0,
            "inner_group_overlap_required": 0,
            "target_transform_fit_scope": "outer-train only",
            "capacity_selection_scope": "outer-train grouped inner CV only",
            "outer_test_labels_used_for_selection": False,
        },
        "output_hashes": {
            "oof_predictions.csv.gz": _sha256(args.out_dir / "oof_predictions.csv.gz"),
            "inner_scores.csv.gz": _sha256(args.out_dir / "inner_scores.csv.gz"),
            "selections.csv": _sha256(args.out_dir / "selections.csv"),
        },
        "device": args.device,
        "parallel_jobs": args.parallel_jobs,
        "probe": args.probe,
        "selection_uses_outer_test": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print("\nSelection counts:\n", selections.selected_candidate.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
