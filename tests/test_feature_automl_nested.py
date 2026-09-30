from __future__ import annotations

import pandas as pd

from scripts.benchmark_feature_automl_nested import (
    INCUMBENT_ROUTE,
    _connected_group_labels,
    metrics,
    pareto_frontier,
    select_incumbent_guarded,
)
from scripts.benchmark_xgb_robust_nested import allowed_transforms, inverse_target, transform_target

import numpy as np


def _row(route: str, score: float, spearman: float, r2: float, rank: int = 1) -> dict[str, object]:
    return {
        "route": route, "route_rank": rank, "candidate": "compact",
        "complexity_rank": 1, "transform": "identity", "transform_rank": 0,
        "feature_count": 10, "model_count": 1, "mean_score": score,
        "se_score": 0.0, "mean_spearman": spearman, "mean_r2": r2,
    }


def test_pareto_frontier_removes_dominated_candidate() -> None:
    frame = pd.DataFrame([
        _row("a", 0.5, 0.6, 0.4),
        _row("b", 0.4, 0.5, 0.3),
        _row("c", 0.5, 0.7, 0.2),
    ])
    assert set(pareto_frontier(frame).route) == {"a", "c"}


def test_incumbent_is_kept_when_challenger_trades_metrics() -> None:
    frame = pd.DataFrame([
        _row(INCUMBENT_ROUTE, 0.50, 0.60, 0.40),
        _row("motif_only", 0.51, 0.70, 0.32),
    ])
    selected, audit = select_incumbent_guarded(frame)
    assert selected.route == INCUMBENT_ROUTE
    assert audit["selected_role"] == "incumbent"


def test_challenger_must_improve_both_metrics() -> None:
    frame = pd.DataFrame([
        _row(INCUMBENT_ROUTE, 0.50, 0.60, 0.40),
        _row("motif_chemistry", 0.55, 0.65, 0.45),
    ])
    selected, audit = select_incumbent_guarded(frame)
    assert selected.route == "motif_chemistry"
    assert audit["selected_role"] == "challenger"


def test_one_se_cannot_downgrade_challenger_below_incumbent_gate() -> None:
    incumbent = _row(INCUMBENT_ROUTE, 0.50, 0.60, 0.40)
    incumbent["candidate"] = "compact"
    incumbent["complexity_rank"] = 1
    best = _row("motif_chemistry", 0.56, 0.66, 0.46)
    best.update({"candidate": "larger", "complexity_rank": 2, "se_score": 0.08})
    simpler = _row("motif_chemistry", 0.51, 0.70, 0.30)
    simpler.update({"candidate": "smaller", "complexity_rank": 0})
    selected, audit = select_incumbent_guarded(pd.DataFrame([incumbent, best, simpler]))
    assert audit["selected_role"] == "challenger"
    assert selected.candidate == "larger"
    assert selected.mean_spearman >= incumbent["mean_spearman"]
    assert selected.mean_r2 >= incumbent["mean_r2"]


def test_target_transform_edge_cases_and_roundtrip() -> None:
    assert allowed_transforms(np.array([-0.1, 0.0, 2.0])) == ("identity",)
    assert allowed_transforms(np.array([0.0, 0.0, 1000.0])) == ("identity", "log1p")
    values = np.array([0.0, 1e-9, 1.0, 1e6])
    np.testing.assert_allclose(inverse_target(transform_target(values, "log1p"), "log1p"), values)


def test_constant_validation_fold_has_finite_penalized_metrics() -> None:
    result = metrics(np.ones(4), np.ones(4))
    assert result["spearman"] == -1.0
    assert result["r2"] == -1.0
    assert np.isfinite(list(result.values())).all()


def test_joint_one_se_prefers_identity_when_qualified_and_within_threshold() -> None:
    incumbent = _row(INCUMBENT_ROUTE, 0.50, 0.60, 0.40)
    log_best = _row("motif_chemistry", 0.60, 0.70, 0.50)
    log_best.update({"transform": "log1p", "transform_rank": 1, "se_score": 0.03})
    identity = _row("motif_chemistry", 0.58, 0.68, 0.48)
    identity.update({"transform": "identity", "transform_rank": 0})
    selected, audit = select_incumbent_guarded(pd.DataFrame([incumbent, log_best, identity]))
    assert audit["selected_role"] == "challenger"
    assert selected["transform"] == "identity"


def test_joint_groups_close_source_or_sequence_transitively() -> None:
    source = np.asarray(["patent_a", "patent_a", "patent_b", "patent_c", "patent_d"])
    sequence = np.asarray(["AAA", "CCC", "CCC", "GGG", "TTT"])
    groups = _connected_group_labels(source, sequence)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    assert groups[4] != groups[0]
