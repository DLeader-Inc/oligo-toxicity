from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_feature_automl_nested import (
    _capacity,
    _compatible_outer_artifacts,
    _sha256,
    filter_policy_candidates,
    inverse_target,
    policy_grid,
    rule_target_scale,
    select_policy_candidate,
    transform_target,
)
from src.features.automl_descriptor_routes import route_by_name


def _candidate_rows() -> pd.DataFrame:
    rows = []
    for route_rank, route in enumerate(("registry_only", "motif_only"), start=1):
        for complexity_rank, candidate in enumerate(("ultra_compact", "current_balanced")):
            for transform_rank, transform in enumerate(("identity", "log1p")):
                rows.append({
                    "route": route, "route_rank": route_rank, "candidate": candidate,
                    "complexity_rank": complexity_rank, "transform": transform,
                    "transform_rank": transform_rank, "feature_count": 10 + route_rank,
                    "model_count": 1, "mean_score": 0.5, "sd_score": 0.0,
                    "folds": 3, "mean_spearman": 0.5, "mean_r2": 0.5,
                    "mean_model_scale_r2": 0.5, "se_score": 0.0,
                })
    return pd.DataFrame(rows)


def test_rule_scale_negative_and_constant_are_identity() -> None:
    assert rule_target_scale(np.array([-0.1, 0.0, 10.0]))[0] == "identity"
    assert rule_target_scale(np.zeros(8))[0] == "identity"


def test_rule_scale_uses_strict_skewness_boundary() -> None:
    values = np.array([0.0] * 98 + [1.0, 1.0])
    transform, audit = rule_target_scale(values)
    assert audit["fisher_pearson_skewness"] > 2.0
    assert transform == "log1p"


def test_rule_scale_skewness_equal_to_two_stays_identity() -> None:
    values = np.arange(8, dtype=float)
    with patch("scripts.benchmark_feature_automl_nested.skew", return_value=2.0):
        assert rule_target_scale(values)[0] == "identity"
    with patch("scripts.benchmark_feature_automl_nested.skew", return_value=2.000001):
        assert rule_target_scale(values)[0] == "log1p"


def test_log1p_roundtrip_including_zero_and_large_value() -> None:
    raw = np.array([0.0, 1.0, 1e8])
    np.testing.assert_allclose(inverse_target(transform_target(raw, "log1p"), "log1p"), raw)


def test_negative_values_disable_log1p_for_selectable_policy() -> None:
    _routes, _capacities, transforms, _audit = policy_grid("full_automl", np.array([-1.0, 0.0, 2.0]))
    assert transforms == {"identity"}


def test_policy_filter_and_one_se_tie_prefer_simpler_candidate() -> None:
    rows, _audit = filter_policy_candidates(_candidate_rows(), "route_scale", np.array([0.0, 1.0, 2.0]))
    assert set(rows.candidate) == {"current_balanced"}
    selected, _selection = select_policy_candidate(_candidate_rows(), "full_automl")
    assert selected.candidate == "ultra_compact"
    assert selected["transform"] == "identity"
    assert selected.route == "registry_only"


def test_expert_scale_policies_freeze_route_and_capacity() -> None:
    skewed = np.array([0.0] * 98 + [1.0, 1.0])
    routes, capacities, transforms, audit = policy_grid("expert_rule_scale", skewed)
    assert routes == {"legacy_fixed_50_50"}
    assert capacities == {"current_balanced"}
    assert transforms == {"log1p"}
    assert audit["kind"] == "outer_train_rule"
    routes, capacities, transforms, audit = policy_grid("expert_nested_scale", skewed)
    assert routes == {"legacy_fixed_50_50"}
    assert capacities == {"current_balanced"}
    assert transforms == {"identity", "log1p"}
    assert audit["kind"] == "inner_selected_scale_only"


def _capacity_inner_rows() -> pd.DataFrame:
    rows = []
    deltas = {
        "compact_regularized": (0.03, 0.02, 0.04),
        "conservative_shallow": (0.01, 0.02, 0.00),
        "current_balanced": (0.00, 0.00, 0.00),
        "deep_regularized_control": (0.04, -0.01, 0.09),
    }
    for rank, (candidate, (score_delta, rho_delta, r2_delta)) in enumerate(deltas.items(), start=1):
        for fold, jitter in enumerate((-0.002, 0.0, 0.002), start=1):
            rows.append({
                "route": "legacy_fixed_50_50", "route_rank": 9,
                "candidate": candidate, "complexity_rank": rank,
                "transform": "identity", "transform_rank": 0,
                "feature_count": 20, "model_count": 2, "inner_fold": fold,
                "selection_score": 0.4 + score_delta + jitter,
                "spearman": 0.5 + rho_delta + jitter,
                "r2": 0.3 + r2_delta + jitter,
                "model_scale_r2": 0.3 + r2_delta + jitter,
            })
    return pd.DataFrame(rows)


def test_capacity_grid_excludes_ultra_from_selectors_but_keeps_fixed_control() -> None:
    y = np.arange(10, dtype=float)
    routes, capacities, _transforms, _audit = policy_grid("expert_rule_capacity_best", y)
    assert routes == {"legacy_fixed_50_50"}
    assert "ultra_compact" not in capacities
    assert capacities == {
        "compact_regularized", "conservative_shallow", "current_balanced", "deep_regularized_control"
    }
    _routes, capacities, _transforms, _audit = policy_grid("expert_rule_fixed_ultra", y)
    assert capacities == {"ultra_compact"}


def test_capacity_best_and_guarded_selection_are_distinct() -> None:
    inner = _capacity_inner_rows()
    from scripts.benchmark_feature_automl_nested import aggregate_inner
    aggregate = aggregate_inner(inner)
    best, _audit = select_policy_candidate(aggregate, "expert_rule_capacity_best", inner)
    assert best.candidate == "deep_regularized_control"
    guarded, audit = select_policy_candidate(aggregate, "expert_rule_capacity_guarded", inner)
    assert guarded.candidate == "compact_regularized"
    assert audit["accepted_challengers"] == 2


def test_compatible_outer_reuse_requires_model_hash_and_exact_membership(tmp_path: Path) -> None:
    root = tmp_path / "source"
    checkpoint = root / "fold_checkpoints"
    model_dir = root / "outer_models" / "cytotoxicity" / "partition_1"
    checkpoint.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    model_records = []
    for component in ("legacy_branch_a", "legacy_branch_b"):
        path = model_dir / f"member_1__{component}.joblib"
        path.write_bytes(component.encode())
        model_records.append({"path": str(path), "sha256": _sha256(path)})
    membership = model_dir / "membership.json"
    membership.write_text(json.dumps({"train_row_indices": [0, 1]}))
    route = route_by_name("legacy_fixed_50_50")
    capacity = _capacity("current_balanced")
    selection = {
        "selected_route": route.name, "selected_candidate": capacity["name"],
        "selected_transform": "identity", "selected_components": list(route.components),
        "selected_weights": list(route.weights), "selected_xgboost_params": capacity["params"],
        "seed_values": [17], "membership_path": str(membership), "models": model_records,
    }
    (checkpoint / "cytotoxicity__partition1__selection.json").write_text(json.dumps(selection))
    pd.DataFrame({
        "row_index": [2], "y": [3.0], "pred_policy": [2.5],
        "pred_model_scale_policy": [2.5], "pred_member_1": [2.5],
    }).to_csv(checkpoint / "cytotoxicity__partition1__predictions.csv", index=False)
    reused = _compatible_outer_artifacts(
        (root,), dataset="cytotoxicity", number=1,
        train=np.array([0, 1]), test=np.array([2]), y_raw=np.array([1.0, 2.0, 3.0]),
        route=route, capacity=capacity, transform="identity", seeds=[17],
    )
    assert reused is not None
    Path(model_records[0]["path"]).write_bytes(b"tampered")
    assert _compatible_outer_artifacts(
        (root,), dataset="cytotoxicity", number=1,
        train=np.array([0, 1]), test=np.array([2]), y_raw=np.array([1.0, 2.0, 3.0]),
        route=route, capacity=capacity, transform="identity", seeds=[17],
    ) is None
