#!/usr/bin/env python3
"""Strict nested mechanism-informed expert and AutoML policies.

The outer test labels are evaluation-only. The published primary system is
``expert_rule_scale``. Other policies, including ``full_automl``, are controls.
The historical CLI default is retained; always supply --policy explicitly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_aso_atlas2_project_representation import (  # noqa: E402
    canonical_aso_sequence,
    load_task,
)
from scripts.benchmark_registry_nine_endpoints import (  # noqa: E402
    ATLAS,
    EARLIER,
    earlier_frame,
)
from scripts.benchmark_xgb_robust_nested import (  # noqa: E402
    _outer_partitions,
    _seed_values,
    allowed_transforms,
    inverse_target,
    transform_target,
)
from src.features.automl_descriptor_routes import (  # noqa: E402
    ROUTES,
    SCHEMA_VERSION,
    RouteSpec,
    build_route_matrices,
    route_by_name,
    route_feature_count,
)
from src.data.grouping import build_sequence_contract  # noqa: E402


CAPACITIES: tuple[dict[str, Any], ...] = (
    {
        "name": "ultra_compact",
        "complexity_rank": 0,
        "params": {
            "n_estimators": 150, "max_depth": 2, "learning_rate": 0.03,
            "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_lambda": 5.0, "reg_alpha": 0.2,
        },
    },
    {
        "name": "compact_regularized",
        "complexity_rank": 1,
        "params": {
            "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
            "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_lambda": 3.0, "reg_alpha": 0.1,
        },
    },
    {
        "name": "conservative_shallow",
        "complexity_rank": 2,
        "params": {
            "n_estimators": 600, "max_depth": 3, "learning_rate": 0.02,
            "min_child_weight": 8, "subsample": 0.75, "colsample_bytree": 0.7,
            "reg_lambda": 8.0, "reg_alpha": 0.5,
        },
    },
    {
        "name": "current_balanced",
        "complexity_rank": 3,
        "params": {
            "n_estimators": 600, "max_depth": 4, "learning_rate": 0.02,
            "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_lambda": 3.0, "reg_alpha": 0.1,
        },
    },
    {
        "name": "deep_regularized_control",
        "complexity_rank": 4,
        "params": {
            "n_estimators": 600, "max_depth": 5, "learning_rate": 0.02,
            "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_lambda": 5.0, "reg_alpha": 0.2,
        },
    },
)

INCUMBENT_ROUTE = "legacy_fixed_50_50"
ADMISSIBLE_CAPACITIES = (
    "compact_regularized",
    "conservative_shallow",
    "current_balanced",
    "deep_regularized_control",
)
FIXED_RULE_CAPACITY_POLICIES = {
    "expert_rule_fixed_ultra": "ultra_compact",
    "expert_rule_fixed_compact": "compact_regularized",
    "expert_rule_fixed_conservative": "conservative_shallow",
    "expert_rule_fixed_deep": "deep_regularized_control",
}
POLICIES = (
    "fixed_expert",
    "expert_nested_scale",
    "expert_rule_scale",
    *FIXED_RULE_CAPACITY_POLICIES,
    "expert_rule_capacity_best",
    "expert_rule_capacity_guarded",
    "route_only",
    "route_scale",
    "full_automl",
    "full_rule_scale",
)
SELECTION_SCHEMA_VERSION = "mechanism_informed_nested_policy_v2"
DEPLOYMENT_SCHEMA_VERSION = "mechanism_informed_deployment_policy_v1"


def _connected_group_labels(primary: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    """Join rows sharing either a source group or an exact molecular sequence."""

    parent = np.arange(len(primary), dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for values in (primary, sequence):
        first: dict[str, int] = {}
        for index, value in enumerate(values.astype(str)):
            if value in first:
                union(first[value], index)
            else:
                first[value] = index
    roots = [find(index) for index in range(len(primary))]
    lookup = {root: number for number, root in enumerate(sorted(set(roots)))}
    return np.asarray([f"joint_{lookup[root]:05d}" for root in roots], dtype=object)


def strict_exact_sequence_partitions(dataset: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Build five folds with zero exact molecular-sequence leakage.

    OligoGym endpoints use the base sequence instantiated by HELM. ASO-Atlas
    uses connected components over earliest-patent group OR exact HELM-derived
    base sequence, so neither source nor exact sequence crosses a fold.
    """

    contract = build_sequence_contract(frame, dataset_key=dataset)
    sequence = contract["molecular_sequence"].astype(str).to_numpy()
    if dataset in ATLAS:
        groups = _connected_group_labels(frame.cv_group.astype(str).to_numpy(), sequence)
        split_mode = "source_or_exact_molecular_sequence_connected_groupkfold"
    else:
        groups = sequence
        split_mode = "exact_molecular_sequence_groupkfold"
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    dummy = np.zeros((len(frame), 1), dtype=np.float32)
    result = []
    for partition, (train, test) in enumerate(splitter.split(dummy, frame.y, groups), start=1):
        result.append({
            "partition": partition,
            "train": np.asarray(train, dtype=int),
            "test": np.asarray(test, dtype=int),
            "groups": groups,
            "split_mode": split_mode,
        })
    return result


def outer_partitions(dataset: str, frame: pd.DataFrame, grouping: str) -> list[dict[str, Any]]:
    if grouping == "exact_sequence":
        return strict_exact_sequence_partitions(dataset, frame)
    return _outer_partitions(dataset, frame)


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
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _model(params: dict[str, Any], seed: int, device: str) -> XGBRegressor:
    return XGBRegressor(
        **params, objective="reg:squarederror", random_state=seed,
        tree_method="hist", device=device, n_jobs=1, verbosity=0,
    )


def _safe_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.ptp(y) <= np.finfo(float).eps:
        return -1.0
    value = float(r2_score(y, prediction))
    return value if np.isfinite(value) else -1.0


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if np.ptp(y) <= np.finfo(float).eps or np.ptp(prediction) <= np.finfo(float).eps:
        spearman = -1.0
    else:
        correlation = spearmanr(y, prediction).statistic
        spearman = float(correlation) if np.isfinite(correlation) else -1.0
    y_std = float(np.std(y, ddof=0))
    rmse = float(np.sqrt(mean_squared_error(y, prediction)))
    return {
        "spearman": spearman,
        "r2": _safe_r2(y, prediction),
        "rmse": rmse,
        "y_std": y_std,
        "rmse_over_y_std": rmse / max(y_std, np.finfo(float).eps),
    }


def balanced_score(spearman: float, r2: float) -> float:
    return 0.5 * float(spearman) + 0.5 * float(np.clip(r2, -1.0, 1.0))


def rule_target_scale(y_outer_train: np.ndarray) -> tuple[str, dict[str, Any]]:
    """Predeclared outer-training-only scale rule used by sensitivity policy."""

    values = np.asarray(y_outer_train, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("target scale rule requires finite outer-training labels")
    minimum = float(values.min())
    if minimum < 0:
        return "identity", {
            "reason": "negative_outer_train_value", "minimum": minimum,
            "fisher_pearson_skewness": None, "threshold": 2.0,
        }
    value = float(skew(values, bias=False)) if len(values) >= 3 and np.ptp(values) > 0 else 0.0
    transform = "log1p" if value > 2.0 else "identity"
    return transform, {
        "reason": "nonnegative_skewness_above_2" if transform == "log1p" else "nonnegative_skewness_at_most_2",
        "minimum": minimum, "fisher_pearson_skewness": value, "threshold": 2.0,
    }


def policy_grid(policy: str, y_outer_train: np.ndarray) -> tuple[set[str], set[str], set[str], dict[str, Any]]:
    """Return legal route/capacity/scale values without consulting outer-test labels."""

    if policy not in POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    routes = {route.name for route in ROUTES}
    capacities = {item["name"] for item in CAPACITIES}
    transforms = set(allowed_transforms(np.asarray(y_outer_train, dtype=float)))
    rule: dict[str, Any] = {"kind": "inner_selected"}
    if policy == "fixed_expert":
        routes, capacities, transforms = {INCUMBENT_ROUTE}, {"current_balanced"}, {"identity"}
        rule = {"kind": "fixed_expert", "reason": "predeclared A/B, current_balanced, identity"}
    elif policy == "expert_nested_scale":
        routes, capacities = {INCUMBENT_ROUTE}, {"current_balanced"}
        rule = {
            "kind": "inner_selected_scale_only",
            "reason": "predeclared A/B and current_balanced; grouped inner CV selects only the legal target scale",
        }
    elif policy == "expert_rule_scale" or policy in FIXED_RULE_CAPACITY_POLICIES:
        capacity = (
            "current_balanced" if policy == "expert_rule_scale"
            else FIXED_RULE_CAPACITY_POLICIES[policy]
        )
        routes, capacities = {INCUMBENT_ROUTE}, {capacity}
        transform, detail = rule_target_scale(y_outer_train)
        transforms = {transform}
        rule = {
            "kind": "outer_train_rule", **detail, "selected_transform": transform,
            "reason_detail": f"predeclared expert A/B and {capacity}; target scale determined only from outer-train labels",
        }
    elif policy in {"expert_rule_capacity_best", "expert_rule_capacity_guarded"}:
        routes, capacities = {INCUMBENT_ROUTE}, set(ADMISSIBLE_CAPACITIES)
        transform, detail = rule_target_scale(y_outer_train)
        transforms = {transform}
        rule = {
            "kind": "outer_train_rule", **detail, "selected_transform": transform,
            "reason_detail": (
                "expert A/B is fixed; grouped inner CV selects capacity from the admissible grid; "
                "ultra_compact is retained only as a fixed underfitting control"
            ),
        }
    elif policy == "route_only":
        capacities, transforms = {"current_balanced"}, {"identity"}
    elif policy == "route_scale":
        capacities = {"current_balanced"}
    elif policy == "full_rule_scale":
        transform, detail = rule_target_scale(y_outer_train)
        transforms = {transform}
        rule = {"kind": "outer_train_rule", **detail, "selected_transform": transform}
    return routes, capacities, transforms, rule


def filter_policy_candidates(inner: pd.DataFrame, policy: str, y_outer_train: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    routes, capacities, transforms, rule = policy_grid(policy, y_outer_train)
    result = inner[
        inner.route.isin(routes)
        & inner.candidate.isin(capacities)
        & inner["transform"].isin(transforms)
    ].copy()
    if result.empty:
        raise RuntimeError(f"no legal inner candidates for {policy}")
    result["policy"] = policy
    result = result[["policy", *[column for column in result.columns if column != "policy"]]]
    return result, rule


def select_policy_candidate(
    aggregate: pd.DataFrame,
    policy: str,
    inner: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Pareto plus joint one-SE selection, with no outer adoption gate."""

    if policy in {"fixed_expert", "expert_rule_scale", *FIXED_RULE_CAPACITY_POLICIES}:
        selected = aggregate.iloc[0]
        return selected, {
            "selected_reason": (
                "predeclared expert A/B and fixed capacity with outer-train-only scale rule; no inner candidate choice"
                if policy != "fixed_expert"
                else "predeclared expert-engineered reference; no data-driven choice"
            ),
            "pareto_candidates": 1, "one_se_candidates": 1,
        }
    if policy == "expert_rule_capacity_best":
        selected = aggregate.sort_values(
            ["mean_score", "mean_spearman", "mean_r2", "complexity_rank"],
            ascending=[False, False, False, True],
        ).iloc[0]
        return selected, {
            "selected_reason": "highest mean balanced score across grouped inner folds in the admissible capacity grid",
            "admissible_capacities": list(ADMISSIBLE_CAPACITIES),
            "excluded_diagnostic_capacity": "ultra_compact",
            "best_mean_score": float(selected.mean_score),
            "pareto_candidates": int(len(pareto_frontier(aggregate))),
            "one_se_candidates": None,
        }
    if policy == "expert_rule_capacity_guarded":
        if inner is None:
            raise ValueError("expert_rule_capacity_guarded requires paired inner-fold rows")
        incumbent_rows = aggregate[aggregate.candidate.eq("current_balanced")]
        if len(incumbent_rows) != 1:
            raise RuntimeError("guarded capacity policy requires exactly one current_balanced aggregate row")
        incumbent = incumbent_rows.iloc[0]
        current = inner[inner.candidate.eq("current_balanced")][
            ["inner_fold", "selection_score", "spearman", "r2"]
        ].rename(columns={
            "selection_score": "incumbent_score", "spearman": "incumbent_spearman", "r2": "incumbent_r2",
        })
        accepted: list[tuple[float, pd.Series]] = []
        comparisons: list[dict[str, Any]] = []
        for _index, challenger in aggregate[aggregate.candidate.ne("current_balanced")].iterrows():
            paired = inner[inner.candidate.eq(challenger.candidate)][
                ["inner_fold", "selection_score", "spearman", "r2"]
            ].merge(current, on="inner_fold", validate="one_to_one")
            score_delta = paired.selection_score - paired.incumbent_score
            rho_delta = paired.spearman - paired.incumbent_spearman
            r2_delta = paired.r2 - paired.incumbent_r2
            n = len(paired)
            se_delta = float(score_delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
            mean_score_delta = float(score_delta.mean())
            mean_rho_delta = float(rho_delta.mean())
            mean_r2_delta = float(r2_delta.mean())
            passes = bool(
                mean_score_delta > se_delta
                and mean_rho_delta >= 0.0
                and mean_r2_delta >= 0.0
            )
            comparisons.append({
                "candidate": str(challenger.candidate), "paired_folds": n,
                "mean_balanced_score_delta": mean_score_delta, "paired_delta_se": se_delta,
                "mean_spearman_delta": mean_rho_delta, "mean_raw_r2_delta": mean_r2_delta,
                "accepted": passes,
            })
            if passes:
                accepted.append((mean_score_delta, challenger))
        if accepted:
            selected = sorted(
                accepted,
                key=lambda item: (item[0], float(item[1].mean_spearman), float(item[1].mean_r2), -int(item[1].complexity_rank)),
                reverse=True,
            )[0][1]
            reason = "challenger exceeded current_balanced by more than one paired-fold SE and did not reduce mean Spearman or raw R2"
        else:
            selected = incumbent
            reason = "retained current_balanced because no challenger passed the paired stability and two-metric guard"
        return selected, {
            "selected_reason": reason,
            "incumbent_capacity": "current_balanced",
            "admissible_capacities": list(ADMISSIBLE_CAPACITIES),
            "excluded_diagnostic_capacity": "ultra_compact",
            "challenger_comparisons": comparisons,
            "accepted_challengers": int(sum(row["accepted"] for row in comparisons)),
            "pareto_candidates": int(len(pareto_frontier(aggregate))),
            "one_se_candidates": None,
        }
    frontier = pareto_frontier(aggregate)
    best = frontier.sort_values(
        ["mean_score", "feature_count", "model_count", "complexity_rank", "transform_rank", "route_rank"],
        ascending=[False, True, True, True, True, True],
    ).iloc[0]
    threshold = float(best.mean_score - best.se_score)
    eligible = aggregate[aggregate.mean_score.ge(threshold)].copy()
    selected = eligible.sort_values(
        ["complexity_rank", "transform_rank", "model_count", "feature_count", "route_rank", "mean_score"],
        ascending=[True, True, True, True, True, False],
    ).iloc[0]
    return selected, {
        "selected_reason": "joint one-SE simplification relative to the best balanced-score Pareto candidate",
        "best_candidate": str(best.candidate), "best_route": str(best.route),
        "best_transform": str(best["transform"]), "best_mean_score": float(best.mean_score),
        "best_se_score": float(best.se_score), "one_se_threshold": threshold,
        "pareto_candidates": int(len(frontier)), "one_se_candidates": int(len(eligible)),
    }


def pareto_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    """Return candidates not dominated on mean Spearman and mean raw R2."""

    keep: list[bool] = []
    for index, row in frame.iterrows():
        other = frame.drop(index=index)
        dominated = (
            other.mean_spearman.ge(row.mean_spearman)
            & other.mean_r2.ge(row.mean_r2)
            & (
                other.mean_spearman.gt(row.mean_spearman)
                | other.mean_r2.gt(row.mean_r2)
            )
        ).any()
        keep.append(not bool(dominated))
    return frame.loc[keep].copy()


def _joint_one_se(
    rows: pd.DataFrame,
    best: pd.Series,
    *,
    incumbent: pd.Series | None = None,
) -> pd.Series:
    threshold = float(best.mean_score - best.se_score)
    eligible = rows[rows.mean_score.ge(threshold)].copy()
    if incumbent is not None:
        eligible = eligible[
            eligible.mean_spearman.ge(float(incumbent.mean_spearman))
            & eligible.mean_r2.ge(float(incumbent.mean_r2))
            & (
                eligible.mean_spearman.gt(float(incumbent.mean_spearman))
                | eligible.mean_r2.gt(float(incumbent.mean_r2))
            )
        ]
    if eligible.empty:
        raise RuntimeError("no one-SE capacity satisfies the incumbent performance gate")
    return eligible.sort_values(
        ["complexity_rank", "transform_rank", "model_count", "feature_count", "route_rank", "mean_score"],
        ascending=[True, True, True, True, True, False],
    ).iloc[0]


def select_incumbent_guarded(aggregate: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Legacy reader compatibility; current policy runner never calls this adoption gate."""

    incumbent_rows = aggregate[aggregate.route.eq(INCUMBENT_ROUTE)]
    if incumbent_rows.empty:
        raise ValueError("incumbent route missing")
    incumbent_best = pareto_frontier(incumbent_rows).sort_values(
        ["mean_score", "complexity_rank", "transform_rank"],
        ascending=[False, True, True],
    ).iloc[0]
    incumbent = _joint_one_se(incumbent_rows, incumbent_best)
    challengers = aggregate[
        aggregate.route.ne(INCUMBENT_ROUTE)
        & aggregate.mean_spearman.ge(float(incumbent.mean_spearman))
        & aggregate.mean_r2.ge(float(incumbent.mean_r2))
        & (
            aggregate.mean_spearman.gt(float(incumbent.mean_spearman))
            | aggregate.mean_r2.gt(float(incumbent.mean_r2))
        )
    ]
    audit = {
        "incumbent_route": INCUMBENT_ROUTE,
        "incumbent_candidate": str(incumbent.candidate),
        "incumbent_transform": str(incumbent["transform"]),
        "incumbent_mean_spearman": float(incumbent.mean_spearman),
        "incumbent_mean_r2": float(incumbent.mean_r2),
        "eligible_challengers": int(len(challengers)),
    }
    if challengers.empty:
        audit.update({"selected_role": "incumbent", "gate_reason": "no challenger Pareto-dominated the incumbent on both inner means"})
        return incumbent, audit
    frontier = pareto_frontier(challengers)
    best = frontier.sort_values(
        ["mean_score", "feature_count", "model_count", "complexity_rank", "transform_rank", "route_rank"],
        ascending=[False, True, True, True, True, True],
    ).iloc[0]
    selected = _joint_one_se(challengers, best, incumbent=incumbent)
    audit.update({
        "selected_role": "challenger",
        "gate_reason": "selected challenger Pareto-dominated the incumbent on inner mean Spearman and raw R2",
        "challenger_best_route": str(best.route),
        "challenger_best_candidate": str(best.candidate),
        "final_one_se_route": str(selected.route),
        "final_one_se_candidate": str(selected.candidate),
        "final_one_se_transform": str(selected["transform"]),
    })
    return selected, audit


def _fit_component(
    matrix: np.ndarray,
    fit_y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    params: dict[str, Any],
    seed: int,
    device: str,
) -> tuple[np.ndarray, XGBRegressor]:
    estimator = _model(params, seed, device)
    estimator.fit(matrix[train], fit_y[train])
    return np.asarray(estimator.predict(matrix[test]), dtype=float), estimator


def _fit_route(
    route: RouteSpec,
    matrices: dict[str, np.ndarray],
    fit_y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    params: dict[str, Any],
    seed: int,
    device: str,
) -> tuple[np.ndarray, list[tuple[str, XGBRegressor]]]:
    predictions: list[np.ndarray] = []
    models: list[tuple[str, XGBRegressor]] = []
    for offset, component in enumerate(route.components):
        prediction, estimator = _fit_component(
            matrices[component], fit_y, train, test, params,
            seed + 499 * offset, device,
        )
        predictions.append(prediction)
        models.append((component, estimator))
    fused = sum(weight * prediction for weight, prediction in zip(route.weights, predictions, strict=True))
    return np.asarray(fused, dtype=float), models


def _inner_scores(
    dataset: str,
    partition_number: int,
    outer_train: np.ndarray,
    groups: np.ndarray,
    y_raw: np.ndarray,
    matrices: dict[str, np.ndarray],
    schemas: dict[str, list[str]],
    inner_folds: int,
    device: str,
    policy: str = "full_automl",
) -> pd.DataFrame:
    train_groups = groups[outer_train]
    splitter = GroupKFold(n_splits=min(inner_folds, len(np.unique(train_groups))))
    rows: list[dict[str, Any]] = []
    legal_routes, legal_capacities, legal_transforms, _rule = policy_grid(policy, y_raw[outer_train])
    routes = [route for route in ROUTES if route.name in legal_routes]
    capacities = [item for item in CAPACITIES if item["name"] in legal_capacities]
    for transform in sorted(legal_transforms, key=lambda value: 0 if value == "identity" else 1):
        fit_y = np.full_like(y_raw, np.nan, dtype=float)
        fit_y[outer_train] = transform_target(y_raw[outer_train], transform)
        for capacity in capacities:
            params = dict(capacity["params"])
            for inner_fold, (train_local, valid_local) in enumerate(
                splitter.split(outer_train, y_raw[outer_train], train_groups), start=1
            ):
                inner_train = outer_train[train_local]
                inner_valid = outer_train[valid_local]
                if set(groups[inner_train]).intersection(groups[inner_valid]):
                    raise RuntimeError("inner group leakage")
                seed = _seed_values(dataset, partition_number * 100 + inner_fold, 1)[0]
                cache: dict[str, np.ndarray] = {}
                for component in {component for route in routes for component in route.components}:
                    cache[component], _ = _fit_component(
                        matrices[component], fit_y, inner_train, inner_valid,
                        params, seed + 1009 * sorted(matrices).index(component), device,
                    )
                for route in routes:
                    model_prediction = sum(
                        weight * cache[component]
                        for weight, component in zip(route.weights, route.components, strict=True)
                    )
                    raw_prediction = inverse_target(model_prediction, transform)
                    value = metrics(y_raw[inner_valid], raw_prediction)
                    rows.append({
                        "policy": policy, "dataset": dataset, "outer_partition": partition_number,
                        "inner_fold": inner_fold, "route": route.name,
                        "route_rank": route.route_rank,
                        "candidate": capacity["name"],
                        "complexity_rank": capacity["complexity_rank"],
                        "transform": transform,
                        "transform_rank": 0 if transform == "identity" else 1,
                        "feature_count": route_feature_count(route, schemas),
                        "model_count": route.model_count,
                        "train_rows": len(inner_train), "valid_rows": len(inner_valid),
                        "train_groups": len(np.unique(groups[inner_train])),
                        "valid_groups": len(np.unique(groups[inner_valid])),
                        "group_overlap": 0,
                        **value,
                        "model_scale_r2": _safe_r2(fit_y[inner_valid], model_prediction),
                        "selection_score": balanced_score(value["spearman"], value["r2"]),
                    })
    return pd.DataFrame(rows)


def aggregate_inner(inner: pd.DataFrame) -> pd.DataFrame:
    aggregate = inner.groupby(
        ["route", "route_rank", "candidate", "complexity_rank", "transform", "transform_rank", "feature_count", "model_count"],
        as_index=False,
    ).agg(
        mean_score=("selection_score", "mean"), sd_score=("selection_score", "std"),
        folds=("inner_fold", "nunique"), mean_spearman=("spearman", "mean"),
        mean_r2=("r2", "mean"), mean_model_scale_r2=("model_scale_r2", "mean"),
    )
    aggregate["se_score"] = aggregate.sd_score.fillna(0.0) / np.sqrt(aggregate.folds)
    return aggregate


def _capacity(name: str) -> dict[str, Any]:
    return next(item for item in CAPACITIES if item["name"] == name)


def _membership_payload(
    dataset: str, frame: pd.DataFrame, train: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    helms = frame["HELM Annotation"].astype(str)
    sequences = frame.x.map(canonical_aso_sequence).astype(str)
    return {
        "dataset": dataset,
        "train_row_indices": [int(value) for value in train],
        "train_group_ids": sorted(set(groups[train].astype(str))),
        "train_helm_sha256": sorted({hashlib.sha256(value.encode()).hexdigest() for value in helms.iloc[train]}),
        "train_sequence_sha256": sorted({hashlib.sha256(value.encode()).hexdigest() for value in sequences.iloc[train]}),
    }


def _compatible_outer_artifacts(
    roots: tuple[Path, ...], *, dataset: str, number: int,
    train: np.ndarray, test: np.ndarray, y_raw: np.ndarray,
    route: RouteSpec, capacity: dict[str, Any], transform: str,
    seeds: list[int] | tuple[int, ...],
) -> tuple[pd.DataFrame, list[dict[str, Any]], str] | None:
    """Reuse an exactly compatible fitted outer ensemble after full provenance checks."""

    for root in roots:
        selection_path = root / "fold_checkpoints" / f"{dataset}__partition{number}__selection.json"
        prediction_path = root / "fold_checkpoints" / f"{dataset}__partition{number}__predictions.csv"
        if not selection_path.exists() or not prediction_path.exists():
            continue
        previous = json.loads(selection_path.read_text())
        if (
            previous.get("selected_route") != route.name
            or previous.get("selected_candidate") != capacity["name"]
            or previous.get("selected_transform") != transform
            or previous.get("selected_components") != list(route.components)
            or not np.allclose(previous.get("selected_weights", []), route.weights, atol=0, rtol=0)
            or previous.get("selected_xgboost_params") != capacity["params"]
            or [int(value) for value in previous.get("seed_values", [])] != [int(value) for value in seeds]
        ):
            continue
        membership_path = Path(previous.get("membership_path", ""))
        if not membership_path.exists():
            continue
        membership = json.loads(membership_path.read_text())
        if [int(value) for value in membership.get("train_row_indices", [])] != [int(value) for value in train]:
            continue
        models = [dict(record) for record in previous.get("models", [])]
        if not models or any(
            not Path(record["path"]).exists() or _sha256(Path(record["path"])) != record.get("sha256")
            for record in models
        ):
            continue
        output = pd.read_csv(prediction_path)
        if (
            len(output) != len(test)
            or not np.array_equal(output.row_index.to_numpy(dtype=int), test)
            or not np.allclose(output.y.to_numpy(dtype=float), y_raw[test], atol=1e-7, rtol=0)
        ):
            continue
        required = {
            "pred_policy", "pred_model_scale_policy",
            *(f"pred_member_{index}" for index in range(1, len(seeds) + 1)),
        }
        if not required.issubset(output.columns):
            continue
        return output.copy(), models, str(root.resolve())
    return None


def run_partition(
    dataset: str,
    frame: pd.DataFrame,
    partition: dict[str, Any],
    matrices: dict[str, np.ndarray],
    schemas: dict[str, list[str]],
    *, inner_folds: int, seed_count: int, device: str, out_dir: Path,
    policy: str = "full_automl", reuse_inner_from: Path | None = None,
    reuse_compatible_outer_from: tuple[Path, ...] = (),
) -> None:
    number = int(partition["partition"])
    checkpoint = out_dir / "fold_checkpoints"
    model_dir = out_dir / "outer_models" / dataset / f"partition_{number}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    selection_path = checkpoint / f"{dataset}__partition{number}__selection.json"
    prediction_path = checkpoint / f"{dataset}__partition{number}__predictions.csv"
    inner_path = checkpoint / f"{dataset}__partition{number}__inner_scores.csv.gz"
    if selection_path.exists() and prediction_path.exists() and inner_path.exists():
        previous = json.loads(selection_path.read_text())
        if (
            previous.get("schema_version") == SELECTION_SCHEMA_VERSION
            and previous.get("policy") == policy
            and previous.get("models")
            and len({int(item["member"]) for item in previous["models"]}) == seed_count
            and all(Path(item["path"]).exists() for item in previous["models"])
        ):
            return
    train = np.asarray(partition["train"], dtype=int)
    test = np.asarray(partition["test"], dtype=int)
    groups = np.asarray(partition["groups"]).astype(str)
    if set(groups[train]).intersection(groups[test]):
        raise RuntimeError(f"outer group leakage: {dataset}/{number}")
    y_raw = frame.y.to_numpy(dtype=float)
    reusable_path = (
        reuse_inner_from / "fold_checkpoints" / f"{dataset}__partition{number}__inner_scores.csv.gz"
        if reuse_inner_from is not None else None
    )
    if reusable_path is not None and reusable_path.exists():
        reusable = pd.read_csv(reusable_path)
        inner, scale_rule = filter_policy_candidates(reusable, policy, y_raw[train])
        inner_cache_source = str(reusable_path.resolve())
    else:
        inner = _inner_scores(
            dataset, number, train, groups, y_raw, matrices, schemas,
            inner_folds, device, policy,
        )
        _routes, _capacities, _transforms, scale_rule = policy_grid(policy, y_raw[train])
        inner_cache_source = None
    aggregate = aggregate_inner(inner)
    selected, selection_audit = select_policy_candidate(aggregate, policy, inner)
    selected_route = route_by_name(str(selected.route))
    selected_capacity = _capacity(str(selected.candidate))
    transform = str(selected["transform"])
    seeds = _seed_values(dataset, number, seed_count)
    membership = _membership_payload(dataset, frame, train, groups)
    reused = _compatible_outer_artifacts(
        reuse_compatible_outer_from, dataset=dataset, number=number,
        train=train, test=test, y_raw=y_raw, route=selected_route,
        capacity=selected_capacity, transform=transform, seeds=seeds,
    )
    reused_from = None
    if reused is not None:
        output, model_records, reused_from = reused
        output["policy"] = policy
        output["selected_route"] = selected_route.name
        output["selected_candidate"] = selected_capacity["name"]
        output["selected_transform"] = transform
        output["selection_reason"] = selection_audit["selected_reason"]
    else:
        fit_y = transform_target(y_raw, transform)
        member_predictions: list[np.ndarray] = []
        member_model_scale_predictions: list[np.ndarray] = []
        model_records = []
        for member, seed in enumerate(seeds, start=1):
            prediction, models = _fit_route(
                selected_route, matrices, fit_y, train, test,
                dict(selected_capacity["params"]), seed, device,
            )
            member_model_scale_predictions.append(prediction)
            member_predictions.append(inverse_target(prediction, transform))
            for component, estimator in models:
                path = model_dir / f"member_{member}__{component}.joblib"
                joblib.dump(estimator, path)
                model_records.append({
                    "member": member, "seed": seed, "component": component,
                    "weight": selected_route.weights[selected_route.components.index(component)],
                    "path": str(path.resolve()), "sha256": _sha256(path),
                    "feature_count": len(schemas[component]),
                })
        values = np.column_stack(member_predictions)
        model_scale_values = np.column_stack(member_model_scale_predictions)
        output = pd.DataFrame({
            "policy": policy, "dataset": dataset, "outer_partition": number, "row_index": test,
            "y": y_raw[test], "selected_route": selected_route.name,
            "selected_candidate": selected_capacity["name"], "selected_transform": transform,
            "selection_reason": selection_audit["selected_reason"],
            "pred_policy": values.mean(axis=1),
            "pred_automl_v2": values.mean(axis=1), "member_sd": values.std(axis=1, ddof=0),
            "member_min": values.min(axis=1), "member_max": values.max(axis=1),
            "pred_model_scale_policy": model_scale_values.mean(axis=1),
            "pred_model_scale_automl_v2": model_scale_values.mean(axis=1),
            **{f"pred_member_{i + 1}": values[:, i] for i in range(values.shape[1])},
        })
    inner.to_csv(inner_path, index=False, compression="gzip")
    output.to_csv(prediction_path, index=False)
    membership_path = model_dir / "membership.json"
    membership_path.write_text(json.dumps(membership, indent=2) + "\n", encoding="utf-8")
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "policy": policy, "dataset": dataset, "outer_partition": number,
        "outer_train_rows": len(train), "outer_test_rows": len(test),
        "outer_train_groups": len(np.unique(groups[train])),
        "outer_test_groups": len(np.unique(groups[test])), "outer_group_overlap": 0,
        "selected_route": selected_route.name,
        "selected_components": list(selected_route.components),
        "selected_weights": list(selected_route.weights),
        "selected_candidate": selected_capacity["name"],
        "selected_xgboost_params": selected_capacity["params"],
        "selected_transform": transform,
        "inverse_transform": "identity" if transform == "identity" else "expm1",
        "model_output_scale": "y" if transform == "identity" else "log1p(y)",
        "treeshap_scale": "y" if transform == "identity" else "log1p(y)",
        "selection_uses_outer_test": False, "seed_values": list(seeds),
        "selection_audit": selection_audit, "scale_rule": scale_rule,
        "inner_cache_source": inner_cache_source,
        "reused_compatible_outer_artifacts_from": reused_from,
        "models": model_records,
        "membership_path": str(membership_path.resolve()),
    }
    selection_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fit_deployment(
    dataset: str,
    frame: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    schemas: dict[str, list[str]],
    *, inner_folds: int, seed_count: int, device: str, out_dir: Path,
    policy: str = "full_automl", outer_grouping: str = "source_protocol",
) -> dict[str, Any]:
    """Repeat the policy on all rows for a deployment recipe, never performance evidence."""

    model_dir = out_dir / "deployment_models" / dataset
    manifest_path = model_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        persisted = previous.get("models", [])
        if (
            previous.get("schema_version") == DEPLOYMENT_SCHEMA_VERSION
            and previous.get("policy") == policy
            and previous.get("seed_count") == seed_count
            and previous.get("models")
            and all(Path(row["path"]).exists() for row in persisted)
        ):
            return previous
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = np.arange(len(frame), dtype=int)
    groups = np.asarray(outer_partitions(dataset, frame, outer_grouping)[0]["groups"]).astype(str)
    if dataset in ATLAS:
        groups = frame.cv_group.astype(str).to_numpy()
    y_raw = frame.y.to_numpy(dtype=float)
    legal_routes, legal_capacities, legal_transforms, scale_rule = policy_grid(policy, y_raw)
    fixed_deployment_policies = {"fixed_expert", "expert_rule_scale", *FIXED_RULE_CAPACITY_POLICIES}
    if policy in fixed_deployment_policies:
        # These policies have exactly one legal recipe. Running grouped inner CV
        # here cannot change deployment and only wastes a full candidate refit.
        route = route_by_name(next(iter(legal_routes)))
        capacity = _capacity(next(iter(legal_capacities)))
        transform = next(iter(legal_transforms))
        selection_audit = {
            "selected_reason": "single predeclared deployment recipe; no inner candidate choice",
            "pareto_candidates": 1,
            "one_se_candidates": 1,
            "inner_cv_skipped_for_fixed_deployment_recipe": True,
        }
        inner = pd.DataFrame([{
            "policy": policy, "dataset": dataset, "outer_partition": 999,
            "route": route.name, "candidate": capacity["name"],
            "transform": transform, "selection_role": "fixed_deployment_recipe",
        }])
    else:
        inner = _inner_scores(dataset, 999, rows, groups, y_raw, matrices, schemas, inner_folds, device, policy)
        aggregate = aggregate_inner(inner)
        selected, selection_audit = select_policy_candidate(aggregate, policy, inner)
        route = route_by_name(str(selected.route))
        capacity = _capacity(str(selected.candidate))
        transform = str(selected["transform"])
    fit_y = transform_target(y_raw, transform)
    models: list[dict[str, Any]] = []
    for member, seed in enumerate(_seed_values(dataset, 999, seed_count), start=1):
        _prediction, fitted = _fit_route(
            route, matrices, fit_y, rows, rows[:1], dict(capacity["params"]), seed, device
        )
        for component, estimator in fitted:
            path = model_dir / f"member_{member}__{component}.joblib"
            joblib.dump(estimator, path)
            models.append({
                "member": member, "seed": seed, "component": component,
                "weight": route.weights[route.components.index(component)],
                "path": str(path.resolve()), "sha256": _sha256(path),
                "feature_count": len(schemas[component]),
            })
    inner_path = model_dir / "inner_scores.csv.gz"
    inner.to_csv(inner_path, index=False, compression="gzip")
    membership = _membership_payload(dataset, frame, rows, groups)
    membership_path = model_dir / "membership.json"
    membership_path.write_text(json.dumps(membership, indent=2) + "\n", encoding="utf-8")
    payload = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION, "policy": policy, "dataset": dataset,
        "training_rows": len(rows), "training_groups": len(np.unique(groups)),
        "selection_uses_external_or_outer_test": False,
        "selected_route": route.name, "selected_components": list(route.components),
        "selected_weights": list(route.weights), "selected_candidate": capacity["name"],
        "selected_xgboost_params": capacity["params"], "selected_transform": transform,
        "model_output_scale": "y" if transform == "identity" else "log1p(y)",
        "treeshap_scale": "y" if transform == "identity" else "log1p(y)",
        "inverse_transform": "identity" if transform == "identity" else "expm1",
        "seed_count": seed_count, "models": models, "membership_path": str(membership_path.resolve()),
        "training_device": device,
        "selection_audit": selection_audit, "scale_rule": scale_rule,
        "deployment_evidence_boundary": "development-selected full-data recipe; not unbiased performance evidence",
        "inner_scores_path": str(inner_path.resolve()),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def summarize(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.concat(
        [pd.read_csv(path) for path in sorted((out_dir / "fold_checkpoints").glob("*__predictions.csv"))],
        ignore_index=True,
    )
    rows: list[dict[str, Any]] = []
    for (dataset, partition), group in predictions.groupby(["dataset", "outer_partition"]):
        policies = group.policy.unique()
        if len(policies) != 1:
            raise RuntimeError(f"mixed policies within outer fold: {dataset}/{partition}")
        transforms = group.selected_transform.unique()
        if len(transforms) != 1:
            raise RuntimeError(f"mixed transforms within outer fold: {dataset}/{partition}")
        value = metrics(group.y.to_numpy(), group.pred_policy.to_numpy())
        model_y = transform_target(group.y.to_numpy(dtype=float), str(transforms[0]))
        model_scale_r2 = _safe_r2(model_y, group.pred_model_scale_policy.to_numpy(dtype=float))
        rows.append({
            "policy": str(policies[0]), "dataset": dataset, "outer_partition": partition,
            "evaluation_route": "strict_nested_policy", "transform": str(transforms[0]),
            "model_scale_r2": model_scale_r2, **value,
        })
    partition_metrics = pd.DataFrame(rows)
    summary = partition_metrics.groupby(["policy", "dataset", "evaluation_route"], as_index=False).agg(
        spearman=("spearman", "mean"), raw_r2=("r2", "mean"),
        model_scale_r2=("model_scale_r2", "mean"),
        rmse=("rmse", "mean"), y_std=("y_std", "mean"),
        rmse_over_y_std=("rmse_over_y_std", "mean"), partitions=("outer_partition", "nunique"),
    )
    transform_counts = (
        partition_metrics.groupby(["policy", "dataset", "evaluation_route"])["transform"]
        .nunique().rename("model_scale_r2_transform_count").reset_index()
    )
    summary = summary.merge(transform_counts, on=["policy", "dataset", "evaluation_route"], how="left")
    # R² values computed on y and log1p(y) do not share one estimand. Keep the
    # partition table intact and suppress an uninterpretable mixed-scale mean.
    summary.loc[summary.model_scale_r2_transform_count.gt(1), "model_scale_r2"] = np.nan
    predictions.to_csv(out_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    partition_metrics.to_csv(out_dir / "partition_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return predictions, summary


def run_probe(datasets: tuple[str, ...], out_dir: Path, device: str, seed_count: int = 5) -> None:
    """Bounded five-seed fit at maximum observed length, width, and capacity."""

    candidates: list[tuple[int, int, str, pd.DataFrame, dict[str, np.ndarray]]] = []
    for dataset in datasets:
        frame = earlier_frame(dataset) if dataset in EARLIER else load_task(dataset).reset_index(drop=True)
        matrices, _schemas, _manifest = build_route_matrices(frame)
        maximum_length = int(frame.x.map(canonical_aso_sequence).str.len().max())
        widest = max(matrices, key=lambda name: matrices[name].shape[1])
        candidates.append((maximum_length, matrices[widest].shape[1], dataset, frame, matrices))
    maximum_length, width, dataset, frame, matrices = max(candidates)
    component = max(matrices, key=lambda name: matrices[name].shape[1])
    matrix = matrices[component]
    groups = np.asarray(_outer_partitions(dataset, frame)[0]["groups"]).astype(str)
    selected: list[int] = []
    for group in np.unique(groups):
        selected.extend(np.flatnonzero(groups == group)[:4].tolist())
        if len(selected) >= 64:
            break
    train = np.asarray(selected[:64], dtype=int)
    if len(train) < 8:
        raise RuntimeError("probe could not obtain enough bounded training rows")
    y = frame.y.to_numpy(dtype=float)
    fit_y = transform_target(y, "log1p") if np.min(y) >= 0 else y
    predictions = []
    for seed in _seed_values(dataset, 997, seed_count):
        estimator = _model(dict(CAPACITIES[-1]["params"]), seed, device)
        estimator.fit(matrix[train], fit_y[train])
        predictions.append(estimator.predict(matrix[train[: min(16, len(train))]]))
    prediction = np.column_stack(predictions)
    payload = {
        "status": "completed", "dataset": dataset,
        "maximum_observed_sequence_length": maximum_length,
        "component": component, "feature_width": width,
        "training_rows": len(train), "prediction_rows": min(16, len(train)),
        "capacity": CAPACITIES[-1]["name"], "seed_count": seed_count,
        "target_scale": "log1p" if np.min(y) >= 0 else "identity",
        "precision": "float32 features / XGBoost default numeric precision",
        "device": device, "finite_predictions": bool(np.isfinite(prediction).all()),
    }
    (out_dir / "probe.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/mechanism_informed_automl_nested_v1/full_automl")
    parser.add_argument("--policy", choices=POLICIES, default="full_automl")
    parser.add_argument(
        "--reuse-inner-from", type=Path,
        help="Reuse a superset of saved inner-score rows, then apply this policy's legal filter.",
    )
    parser.add_argument(
        "--reuse-compatible-outer-from", type=Path, nargs="*", default=(),
        help="Reuse fitted outer members only after exact split/seed/route/capacity/scale/hash checks.",
    )
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--parallel-folds", type=int, default=1)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--skip-deployment", action="store_true")
    parser.add_argument(
        "--deployment-only", action="store_true",
        help="Build full-data deployment refits without revisiting completed outer partitions.",
    )
    parser.add_argument("--datasets", nargs="*", choices=EARLIER + ATLAS)
    parser.add_argument(
        "--outer-grouping", choices=("source_protocol", "exact_sequence"),
        default="source_protocol",
        help="Published/source grouping or strict exact molecular-sequence grouping; Atlas strict mode also preserves patent groups.",
    )
    args = parser.parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = tuple(args.datasets or (EARLIER + ATLAS))
    if args.probe:
        run_probe(datasets, args.out_dir, args.device, args.seed_count)
        return
    schema_rows: list[pd.DataFrame] = []
    for dataset in datasets:
        if args.deployment_only:
            existing_path = args.out_dir / "deployment_models" / dataset / "manifest.json"
            if existing_path.exists():
                existing = json.loads(existing_path.read_text())
                records = existing.get("models", [])
                if (
                    existing.get("schema_version") == DEPLOYMENT_SCHEMA_VERSION
                    and existing.get("policy") == args.policy
                    and existing.get("seed_count") == args.seed_count
                    and records
                    and all(Path(record["path"]).exists() for record in records)
                ):
                    continue
        frame = earlier_frame(dataset) if dataset in EARLIER else load_task(dataset).reset_index(drop=True)
        matrices, schemas, feature_manifest = build_route_matrices(frame)
        feature_manifest.insert(0, "dataset", dataset)
        schema_rows.append(feature_manifest)
        schema_dir = args.out_dir / "feature_schemas"
        schema_dir.mkdir(exist_ok=True)
        for component, names in schemas.items():
            (schema_dir / f"{dataset}__{component}.json").write_text(
                json.dumps(names, indent=2) + "\n", encoding="utf-8"
            )
        if not args.deployment_only:
            partitions = outer_partitions(dataset, frame, args.outer_grouping)
            def execute(partition: dict[str, Any]) -> None:
                run_partition(
                    dataset, frame, partition, matrices, schemas,
                    inner_folds=args.inner_folds, seed_count=args.seed_count,
                    device=args.device, out_dir=args.out_dir, policy=args.policy,
                    reuse_inner_from=args.reuse_inner_from,
                    reuse_compatible_outer_from=tuple(args.reuse_compatible_outer_from),
                )

            if args.parallel_folds == 1:
                for partition in partitions:
                    execute(partition)
            else:
                with ThreadPoolExecutor(max_workers=args.parallel_folds) as pool:
                    list(pool.map(execute, partitions))
        if not args.skip_deployment:
            fit_deployment(
                dataset, frame, matrices, schemas, inner_folds=args.inner_folds,
                seed_count=args.seed_count, device=args.device, out_dir=args.out_dir,
                policy=args.policy, outer_grouping=args.outer_grouping,
            )
    if schema_rows and not args.deployment_only:
        pd.concat(schema_rows, ignore_index=True).to_csv(args.out_dir / "descriptor_family_manifest.csv", index=False)
    _predictions, summary = summarize(args.out_dir)
    manifest = {
        "schema_version": "mechanism_informed_automl_strict_nested_v1",
        "policy": args.policy,
        "descriptor_schema_version": SCHEMA_VERSION,
        "datasets": list(datasets),
        "outer_grouping": args.outer_grouping,
        "outer_protocol": {
            "oligogym": "base 1/2/3-mer KMeans groups, fixed seeds 0..4, 80:20 GroupShuffleSplit",
            "aso_atlas2": "earliest-patent five-fold GroupKFold",
        } if args.outer_grouping == "source_protocol" else {
            "oligogym": "five-fold GroupKFold by exact HELM-derived molecular base sequence",
            "aso_atlas2": "five-fold GroupKFold by connected components sharing earliest patent OR exact HELM-derived molecular base sequence",
        },
        "inner_protocol": f"GroupKFold({args.inner_folds}) on outer-training groups only",
        "selection_policy": (
            "predeclared expert A/B + current_balanced + outer-train-only scale rule; no candidate selection"
            if args.policy == "expert_rule_scale"
            else "predeclared expert A/B + fixed capacity + outer-train-only scale rule; fixed capacity control"
            if args.policy in FIXED_RULE_CAPACITY_POLICIES
            else "expert A/B + outer-train-only scale rule; highest inner mean balanced score selects among admissible capacities"
            if args.policy == "expert_rule_capacity_best"
            else "expert A/B + outer-train-only scale rule; current_balanced changes only after a paired inner-fold stability and two-metric guard"
            if args.policy == "expert_rule_capacity_guarded"
            else "fixed expert A/B + current_balanced; grouped inner CV selects only target scale"
            if args.policy == "expert_nested_scale"
            else "policy-specific legal grid followed by Pareto and joint one-SE selection inside outer-train only; no outer adoption gate"
        ),
        "balanced_score": "0.5*Spearman + 0.5*clip(raw-scale R2,-1,1)",
        "routes": [route.__dict__ for route in ROUTES],
        "capacities": CAPACITIES,
        "candidate_transforms": ["identity", "log1p"],
        "rule_scale": "negative outer-train -> identity; otherwise Fisher-Pearson skewness >2 -> log1p; else identity",
        "reuse_inner_from": str(args.reuse_inner_from.resolve()) if args.reuse_inner_from else None,
        "reuse_compatible_outer_from": [str(path.resolve()) for path in args.reuse_compatible_outer_from],
        "deployment_fitted": not args.skip_deployment,
        "deployment_only": args.deployment_only,
        "seed_ensemble": args.seed_count,
        "parallel_folds": args.parallel_folds,
        "selection_uses_outer_test": False,
        "source_repository": str(ROOT), "source_commit": _git_commit(),
        "training_entry_point": str(Path(__file__).resolve()),
        "training_entry_point_sha256": _sha256(Path(__file__)),
        "external_task_checkpoint_usage": "none",
        "assay_context_policy": (
            "fixed endpoint conditions are applicability constraints; reliably row-joined Atlas "
            "dose/schedule fields are an always-on exposure-context block, not selectable molecular descriptors"
        ),
        "leakage_safeguards": {
            "outer_group_overlap_required": 0, "inner_group_overlap_required": 0,
            "feature_route_capacity_transform_selection_scope": (
                "outer-train grouped inner CV only; expert_rule_* and full_rule_scale transforms are fixed by outer-train rule"
            ),
            "outer_test_labels_used_for_selection": False,
        },
        "probe": bool(args.probe), "elapsed_seconds": time.time() - started,
        "output_hashes": {"summary.csv": _sha256(args.out_dir / "summary.csv")},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
