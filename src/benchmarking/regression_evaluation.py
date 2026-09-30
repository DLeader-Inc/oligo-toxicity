"""Shared regression metrics and multi-objective model-selection policy.

Spearman measures ranking only.  Every benchmark in this project must also
retain an absolute-fit metric (R-squared) and a scale-normalized error
(RMSE/y_std).  The normalized error is mathematically redundant with pooled
R-squared when both use the same rows, so it is reported for interpretation but
is not counted twice in the balanced selection score.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


REQUIRED_REGRESSION_METRICS: tuple[str, ...] = (
    "spearman",
    "r2",
    "rmse",
    "y_std",
    "rmse_over_y_std",
)


def regression_metrics(
    y: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
    *,
    normalization_std: float | None = None,
) -> dict[str, float]:
    """Return ranking, calibration, and normalized-error metrics together.

    ``normalization_std`` can be supplied when fold metrics must use one fixed
    endpoint scale.  Pooled OOF metrics normally leave it unset so ``y_std`` is
    computed from exactly the evaluated rows.
    """

    observed = np.asarray(y, dtype=float).reshape(-1)
    predicted = np.asarray(prediction, dtype=float).reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError(
            f"y and prediction shapes differ: {observed.shape} != {predicted.shape}"
        )
    if observed.size < 2:
        raise ValueError("at least two observations are required")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("y and prediction must contain only finite values")

    pearson = stats.pearsonr(observed, predicted).statistic
    spearman = stats.spearmanr(observed, predicted).statistic
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    y_std = (
        float(np.std(observed, ddof=0))
        if normalization_std is None
        else float(normalization_std)
    )
    if not np.isfinite(y_std) or y_std < 0:
        raise ValueError(f"normalization_std must be finite and non-negative, got {y_std}")
    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "r2": float(r2_score(observed, predicted)),
        "rmse": rmse,
        "y_std": y_std,
        "rmse_over_y_std": rmse / y_std if y_std > 0 else float("nan"),
    }


def balanced_selection_score(spearman: float, r2: float) -> float:
    """Combine rank and absolute-fit evidence without double-counting RMSE.

    R-squared is clipped to ``[-1, 1]`` so one catastrophically miscalibrated
    candidate remains strongly penalized without dominating all rank signal.
    The unmodified R-squared and RMSE/y_std are still retained in every table.
    """

    if not np.isfinite(spearman) or not np.isfinite(r2):
        return float("-inf")
    return 0.5 * float(spearman) + 0.5 * float(np.clip(r2, -1.0, 1.0))


def add_balanced_selection_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the shared multi-metric ``selection_score`` column."""

    missing = {"spearman", "r2"} - set(frame.columns)
    if missing:
        raise ValueError(f"cannot score candidates; missing columns: {sorted(missing)}")
    scored = frame.copy()
    scored["selection_score"] = [
        balanced_selection_score(spearman, r2)
        for spearman, r2 in zip(scored["spearman"], scored["r2"], strict=True)
    ]
    return scored


def assert_complete_metric_columns(
    frame: pd.DataFrame,
    *,
    context: str,
    prefixes: Iterable[str] = ("",),
) -> None:
    """Reject result tables that omit any required regression metric."""

    missing = [
        f"{prefix}{metric}"
        for prefix in prefixes
        for metric in REQUIRED_REGRESSION_METRICS
        if f"{prefix}{metric}" not in frame.columns
    ]
    if missing:
        raise ValueError(f"{context} is missing required metric columns: {missing}")
