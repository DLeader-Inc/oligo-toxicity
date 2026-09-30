from __future__ import annotations

import numpy as np
import pandas as pd

from src.explainability.automl_concepts import (
    aggregate_contributions,
    concept_for_feature,
    high_correlation_clusters,
)


def test_feature_concepts_cover_motif_and_chemistry() -> None:
    assert concept_for_feature("ucore__kmer_TCC") == "literature_linked_or_g_rich_motif"
    assert concept_for_feature("scint_L_fraction") == "lna_amount_and_position"
    assert concept_for_feature("scint_linkage_count_PS") == "backbone_amount_and_position"


def test_concept_aggregation_preserves_signed_member_sum() -> None:
    frame = pd.DataFrame({
        "member_id": ["m1", "m1", "m2", "m2"],
        "model_scale": ["y"] * 4,
        "feature": ["ucore__kmer_TCC", "scint_L_fraction"] * 2,
        "contribution": [0.2, -0.1, 0.3, -0.2],
    })
    result = aggregate_contributions(frame)
    assert result.members.eq(2).all()
    assert result.feature_count.eq(1).all()


def test_high_correlation_cluster_uses_training_values() -> None:
    matrix = np.array([[0, 0, 1], [1, 2, 0], [2, 4, 1], [3, 6, 0]], dtype=float)
    clusters = high_correlation_clusters(matrix, ["ucore__kmer_A", "ucore__kmer_C", "scint_L_fraction"])
    assert clusters[0]["size"] == 2

