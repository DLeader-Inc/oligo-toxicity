from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benchmarking.regression_evaluation import (
    assert_complete_metric_columns,
    balanced_selection_score,
    regression_metrics,
)


def test_shared_metrics_include_ranking_fit_and_scale() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    metrics = regression_metrics(y, y + 0.25)

    assert set(("spearman", "r2", "rmse", "y_std", "rmse_over_y_std")) <= set(metrics)
    assert np.isclose(metrics["rmse_over_y_std"], metrics["rmse"] / metrics["y_std"])


def test_shared_metrics_can_use_fixed_endpoint_scale() -> None:
    metrics = regression_metrics(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 1.0]),
        normalization_std=4.0,
    )

    assert metrics["y_std"] == 4.0
    assert np.isclose(metrics["rmse_over_y_std"], metrics["rmse"] / 4.0)


def test_metric_schema_guard_rejects_spearman_only_table() -> None:
    with pytest.raises(ValueError, match="missing required metric columns"):
        assert_complete_metric_columns(
            pd.DataFrame({"spearman": [0.5]}), context="historical_result"
        )


def test_balanced_score_uses_rank_and_r2() -> None:
    assert balanced_selection_score(0.65, 0.55) > balanced_selection_score(0.90, -0.80)
