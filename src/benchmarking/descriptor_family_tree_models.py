"""Leakage-safe tree-model comparison for descriptor families.

The public functions in this module operate on already-built feature matrices.
All model-specification and family selection happens inside each outer-training
partition; outer-test labels are used only for final metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.benchmarking.regression_evaluation import (
    add_balanced_selection_score,
    regression_metrics as _shared_regression_metrics,
)


@dataclass(frozen=True)
class TreeSpec:
    """One pre-registered tree learner configuration."""

    name: str
    engine: str
    params: dict[str, object]


DEFAULT_TREE_SPECS: tuple[TreeSpec, ...] = (
    TreeSpec(
        "xgb_current",
        "xgb",
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.08},
    ),
    TreeSpec(
        "xgb_regularized",
        "xgb",
        {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ),
    TreeSpec(
        "rf_shallow",
        "rf",
        {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
    ),
    TreeSpec(
        "rf_regularized",
        "rf",
        {
            "n_estimators": 300,
            "max_depth": 30,
            "min_samples_leaf": 2,
            "max_features": 0.3,
        },
    ),
)


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute the mandatory ranking and absolute-prediction metric set."""

    return _shared_regression_metrics(y, prediction)


def fit_predict_tree(
    spec: TreeSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    device: str = "cpu",
) -> np.ndarray:
    """Fit one tree specification with one worker and return predictions."""

    model = build_tree_model(spec, seed=seed, device=device)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def build_tree_model(spec: TreeSpec, *, seed: int, device: str = "cpu"):
    """Construct one pre-registered tree learner."""

    if spec.engine == "xgb":
        model = XGBRegressor(
            **spec.params,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            device=device,
            verbosity=0,
        )
    elif spec.engine == "rf":
        model = RandomForestRegressor(
            **spec.params,
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"unsupported tree engine: {spec.engine}")
    return model


def _safe_group_splits(
    indices: np.ndarray,
    groups: np.ndarray,
    folds: int,
    *,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    local_groups = np.asarray(groups)[indices]
    n_splits = min(int(folds), int(pd.Series(local_groups).nunique()))
    if n_splits < 2:
        return []
    splitter = GroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    dummy = np.zeros((len(indices), 1), dtype=np.float32)
    return [
        (indices[train_pos], indices[test_pos])
        for train_pos, test_pos in splitter.split(dummy, groups=local_groups)
    ]


def _fit_candidate_fold(
    spec: TreeSpec,
    family_name: str,
    matrix: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    device: str,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    prediction = fit_predict_tree(
        spec,
        matrix[train_idx],
        y[train_idx],
        matrix[test_idx],
        seed=seed,
        device=device,
    )
    return spec.name, family_name, test_idx, prediction


def matched_candidate_seed(
    base_seed: int,
    mode_index: int,
    fold_index: int,
    spec_index: int,
) -> int:
    """Return a seed shared by all feature families for one model/fold."""

    return base_seed + mode_index * 10_000 + fold_index * 100 + spec_index * 10


def inner_candidate_scores(
    families: dict[str, np.ndarray],
    y: np.ndarray,
    groups_by_mode: dict[str, np.ndarray],
    train_pool: np.ndarray,
    inner_modes: Iterable[str],
    specs: tuple[TreeSpec, ...],
    *,
    folds: int,
    seed: int,
    parallel_jobs: int,
    shuffle_group_folds: bool = False,
    xgb_device: str = "cpu",
    parallel_prefer: str = "processes",
) -> pd.DataFrame:
    """Score every model/family candidate using only an outer-training pool."""

    mode_frames: list[pd.DataFrame] = []
    for mode_index, mode in enumerate(inner_modes):
        splits = _safe_group_splits(
            train_pool,
            groups_by_mode[mode],
            folds,
            shuffle=shuffle_group_folds,
            random_state=(seed + mode_index * 1_000)
            if shuffle_group_folds
            else None,
        )
        if not splits:
            continue
        predictions = {
            (spec.name, family): np.full(len(y), np.nan, dtype=float)
            for spec in specs
            for family in families
        }
        for fold_index, (train_idx, valid_idx) in enumerate(splits, start=1):
            tasks = (
                delayed(_fit_candidate_fold)(
                    spec,
                    family,
                    matrix,
                    y,
                    train_idx,
                    valid_idx,
                        matched_candidate_seed(
                        seed,
                        mode_index,
                        fold_index,
                            spec_index,
                        ),
                        xgb_device if spec.engine == "xgb" else "cpu",
                )
                for spec_index, spec in enumerate(specs)
                for family, matrix in families.items()
            )
            results = Parallel(n_jobs=parallel_jobs, prefer=parallel_prefer)(tasks)
            for spec_name, family, row_idx, prediction in results:
                predictions[(spec_name, family)][row_idx] = prediction

        rows: list[dict[str, object]] = []
        for spec in specs:
            for family in families:
                candidate_prediction = predictions[(spec.name, family)][train_pool]
                if np.isnan(candidate_prediction).any():
                    raise RuntimeError(f"incomplete inner OOF predictions for {mode}/{spec.name}/{family}")
                rows.append(
                    {
                        "inner_mode": mode,
                        "engine": spec.engine,
                        "spec": spec.name,
                        "family": family,
                        **regression_metrics(y[train_pool], candidate_prediction),
                    }
                )
        mode_frames.append(pd.DataFrame(rows))
    if not mode_frames:
        raise RuntimeError("no valid inner grouped folds")

    scores = pd.concat(mode_frames, ignore_index=True)
    balanced = (
        scores.groupby(["engine", "spec", "family"], as_index=False)
        .agg(
            spearman=("spearman", "mean"),
            pearson=("pearson", "mean"),
            r2=("r2", "mean"),
            rmse=("rmse", "mean"),
            y_std=("y_std", "mean"),
            rmse_over_y_std=("rmse_over_y_std", "mean"),
            n_inner_modes=("inner_mode", "nunique"),
        )
    )
    balanced = add_balanced_selection_score(balanced)
    balanced.insert(0, "inner_mode", "balanced_mean")
    return pd.concat([scores, balanced], ignore_index=True)


def best_spec_per_engine_family(inner_scores: pd.DataFrame) -> pd.DataFrame:
    """Select a model configuration for each engine/family from inner CV."""

    balanced = inner_scores[inner_scores["inner_mode"].eq("balanced_mean")].copy()
    balanced = add_balanced_selection_score(balanced)
    balanced = balanced.sort_values(
        ["engine", "family", "selection_score", "r2", "spearman", "rmse", "spec"],
        ascending=[True, True, False, False, False, True, True],
    )
    return balanced.drop_duplicates(["engine", "family"], keep="first").reset_index(drop=True)


def select_top_k_families(
    selected_specs: pd.DataFrame,
    *,
    engine: str | None,
    top_k: int,
    include_baseline: bool = False,
    family_group_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Rank families by inner CV and return at most ``top_k`` candidates.

    When ``engine`` is ``None``, the best inner-CV engine/configuration is kept
    for each family before ranking. The outer-test partition is never consulted.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    candidates = add_balanced_selection_score(selected_specs)
    if engine is not None:
        candidates = candidates[candidates["engine"].eq(engine)].copy()
    else:
        candidates = candidates.sort_values(
            ["family", "selection_score", "r2", "spearman", "rmse", "engine"],
            ascending=[True, False, False, False, True, True],
        ).drop_duplicates("family", keep="first")
    if candidates.empty:
        raise ValueError(f"no candidates for engine={engine!r}")

    descriptor_rows = candidates[~candidates["family"].eq("core_baseline")].copy()
    if not include_baseline and not descriptor_rows.empty:
        candidates = descriptor_rows
    candidates = candidates.sort_values(
        ["selection_score", "r2", "spearman", "rmse", "family", "engine"],
        ascending=[False, False, False, True, True, True],
    )
    if family_group_map:
        candidates = candidates.assign(
            _selection_group=candidates["family"].map(family_group_map).fillna(
                candidates["family"]
            )
        ).drop_duplicates("_selection_group", keep="first")
        candidates = candidates.drop(columns="_selection_group")
    return candidates.head(top_k).reset_index(drop=True)


def select_near_best_families(
    selected_specs: pd.DataFrame,
    *,
    engine: str | None,
    near_best_fraction: float,
    baseline_margin: float,
    max_families: int,
) -> pd.DataFrame:
    """Select a small set of families using inner scores only."""

    candidates = add_balanced_selection_score(selected_specs)
    if engine is not None:
        candidates = candidates[candidates["engine"].eq(engine)].copy()
    else:
        candidates = candidates.sort_values(
            ["family", "selection_score", "r2", "spearman", "rmse", "engine"],
            ascending=[True, False, False, False, True, True],
        ).drop_duplicates("family", keep="first")
    if candidates.empty:
        raise ValueError(f"no candidates for engine={engine!r}")

    baseline_rows = candidates[candidates["family"].eq("core_baseline")]
    baseline_score = (
        float(baseline_rows["selection_score"].max())
        if not baseline_rows.empty
        else -np.inf
    )
    descriptor_rows = candidates[~candidates["family"].eq("core_baseline")].copy()
    best_score = (
        float(descriptor_rows["selection_score"].max())
        if not descriptor_rows.empty
        else baseline_score
    )
    threshold = best_score - abs(best_score) * near_best_fraction
    selected = descriptor_rows[
        (descriptor_rows["selection_score"] >= threshold)
        | (descriptor_rows["selection_score"] >= baseline_score + baseline_margin)
    ].copy()
    if selected.empty:
        selected = baseline_rows.copy()
    selected = selected.sort_values(
        ["selection_score", "r2", "spearman", "rmse", "family", "engine"],
        ascending=[False, False, False, True, True, True],
    )
    return selected.head(max_families).reset_index(drop=True)


def assert_zero_group_overlap(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray,
) -> tuple[int, int, int]:
    """Return train/test group counts and raise if groups cross a fold."""

    train_groups = set(np.asarray(groups)[train_idx].tolist())
    test_groups = set(np.asarray(groups)[test_idx].tolist())
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"group leakage detected: {len(overlap)} overlapping groups")
    return len(train_groups), len(test_groups), len(overlap)
