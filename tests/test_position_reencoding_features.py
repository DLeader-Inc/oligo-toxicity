from __future__ import annotations

import numpy as np
import pandas as pd
from types import SimpleNamespace

from scripts.train_current_descriptor_families_grouped import _build_families
from scripts.augment_descriptor_family_cache_position_reencoding import augment_families
from scripts.augment_descriptor_family_cache_token_padding import (
    augment_token_padding_families,
)
from src.features.position_reencoding_features import (
    dataframes_for_helms,
    interaction_features_for_helm,
    relative_features_for_helm,
    token_ohe_features_for_helm,
)


HELM_5MER = "RNA1{[moe](A)[sp].[moe](C)[sp].d(G)[sp].d(T)[sp].[moe](A)}$$$$"
HELM_7MER = "RNA1{[moe](T)[sp].[moe](T)[sp].[moe](A)[sp].d(C)[sp].d(G)[sp].d(T)[sp].[moe](A)}$$$$"


def test_token_ohe_preserves_base_sugar_linkage_and_padding() -> None:
    features = token_ohe_features_for_helm(HELM_5MER, max_pos=6)

    assert features["ptok_5p_01_exists"] == 1.0
    assert features["ptok_5p_01_base_A"] == 1.0
    assert features["ptok_5p_01_mod_M"] == 1.0
    assert features["ptok_5p_01_link_PS"] == 1.0
    assert features["ptok_5p_03_base_G"] == 1.0
    assert features["ptok_5p_03_mod_D"] == 1.0
    assert features["ptok_5p_05_link_TERMINAL"] == 1.0
    assert features["ptok_5p_06_exists"] == 0.0
    assert sum(value for key, value in features.items() if key.startswith("ptok_5p_06_base_")) == 0.0


def test_token_padding_alignments_place_the_same_sequence_on_different_canvas_slots() -> None:
    right = token_ohe_features_for_helm(HELM_5MER, max_pos=8, alignment="right")
    left = token_ohe_features_for_helm(HELM_5MER, max_pos=8, alignment="left")
    center = token_ohe_features_for_helm(HELM_5MER, max_pos=8, alignment="center")

    assert right["ptok_5p_01_base_A"] == 1.0
    assert right["ptok_5p_05_link_TERMINAL"] == 1.0
    assert left["ptok_leftpad_04_base_A"] == 1.0
    assert left["ptok_leftpad_08_link_TERMINAL"] == 1.0
    assert center["ptok_centerpad_02_base_A"] == 1.0
    assert center["ptok_centerpad_06_link_TERMINAL"] == 1.0
    assert sum(value for key, value in right.items() if key.endswith("_exists")) == 5.0
    assert sum(value for key, value in left.items() if key.endswith("_exists")) == 5.0
    assert sum(value for key, value in center.items() if key.endswith("_exists")) == 5.0


def test_token_padding_truncation_retains_the_alignment_anchor() -> None:
    right = token_ohe_features_for_helm(HELM_7MER, max_pos=5, alignment="right")
    left = token_ohe_features_for_helm(HELM_7MER, max_pos=5, alignment="left")
    center = token_ohe_features_for_helm(HELM_7MER, max_pos=5, alignment="center")

    assert right["ptok_5p_01_base_T"] == 1.0
    assert right["ptok_5p_05_base_G"] == 1.0
    assert left["ptok_leftpad_01_base_A"] == 1.0
    assert left["ptok_leftpad_05_base_A"] == 1.0
    assert center["ptok_centerpad_01_base_T"] == 1.0
    assert center["ptok_centerpad_05_base_T"] == 1.0


def test_default_token_alignment_remains_historical_right_padding() -> None:
    assert token_ohe_features_for_helm(HELM_5MER, max_pos=8) == (
        token_ohe_features_for_helm(HELM_5MER, max_pos=8, alignment="right")
    )


def test_relative_encoding_aligns_3p_center_and_gap_landmarks() -> None:
    features = relative_features_for_helm(HELM_5MER)

    assert features["prel_3p_01_base_A"] == 1.0
    assert features["prel_3p_01_link_TERMINAL"] == 1.0
    assert features["prel_center_p00_base_G"] == 1.0
    assert features["prel_center_p00_in_gap"] == 1.0
    assert features["prel_gapstart_p00_base_G"] == 1.0
    assert features["prel_gapstart_p00_mod_D"] == 1.0


def test_relative_3p_slot_is_length_invariant_but_absolute_slot_is_not() -> None:
    short_token = token_ohe_features_for_helm(HELM_5MER, max_pos=8)
    long_token = token_ohe_features_for_helm(HELM_7MER, max_pos=8)
    short_relative = relative_features_for_helm(HELM_5MER)
    long_relative = relative_features_for_helm(HELM_7MER)

    assert short_relative["prel_3p_01_base_A"] == long_relative["prel_3p_01_base_A"] == 1.0
    assert short_token["ptok_5p_05_base_A"] == 1.0
    assert long_token["ptok_5p_05_base_A"] == 0.0


def test_interactions_are_named_sequence_chemistry_region_counts() -> None:
    features = interaction_features_for_helm(HELM_5MER)

    assert features["pint_base_A__mod_M__region_wing5"] == 1.0
    assert features["pint_base_G__mod_D__region_gap_first3"] == 1.0
    assert features["pint_motif_CG__region_gap_first3"] == 1.0
    assert features["pint_motif_CG__mod_M"] == 1.0
    assert features["pint_motif_CG__mod_D"] == 1.0


def test_reencoded_dataframes_share_rows_and_have_disjoint_finite_columns() -> None:
    blocks = dataframes_for_helms([HELM_5MER, HELM_7MER], max_pos=8)

    assert set(blocks) == {
        "position_token_ohe",
        "position_relative",
        "position_interactions",
    }
    assert {len(frame) for frame in blocks.values()} == {2}
    all_columns = [set(frame.columns) for frame in blocks.values()]
    assert not (all_columns[0] & all_columns[1])
    assert not (all_columns[0] & all_columns[2])
    assert not (all_columns[1] & all_columns[2])
    assert all(np.isfinite(frame.to_numpy()).all() for frame in blocks.values())


def test_descriptor_family_builder_exposes_separate_and_composite_blocks() -> None:
    parsed = pd.DataFrame(
        {
            "x": [HELM_5MER, HELM_7MER],
            "bases": ["ACGTA", "TTACGTA"],
        }
    )
    baseline = pd.DataFrame({"length": [5.0, 7.0]})
    requested = [
        "position_token_ohe",
        "position_token_ohe_leftpad",
        "position_token_ohe_centerpad",
        "position_relative",
        "position_interactions",
        "position_token_relative",
        "position_reencoded_all",
    ]
    families, coverage = _build_families(
        parsed,
        baseline,
        SimpleNamespace(families=requested, max_pos=8, no_pair_probs=False),
        {},
        dataset_key="toy",
    )

    assert set(families) == {"core_baseline", *requested}
    added = coverage.set_index("family")["n_features_added"]
    assert added["position_token_ohe_leftpad"] == added["position_token_ohe"]
    assert added["position_token_ohe_centerpad"] == added["position_token_ohe"]
    assert added["position_token_relative"] == (
        added["position_token_ohe"] + added["position_relative"]
    )
    assert added["position_reencoded_all"] == (
        added["position_token_ohe"]
        + added["position_relative"]
        + added["position_interactions"]
    )


def test_cache_augmentation_preserves_existing_families_and_baseline() -> None:
    helms = pd.Series([HELM_5MER, HELM_7MER])
    baseline = pd.DataFrame({"baseline__length": [5.0, 7.0]})
    existing = {
        "core_baseline": baseline,
        "seqchem": pd.concat(
            [baseline, pd.DataFrame({"seqchem__signal": [1.0, 2.0]})],
            axis=1,
        ),
    }
    augmented = augment_families(existing, helms, max_pos=8)

    assert augmented["seqchem"].equals(existing["seqchem"])
    assert augmented["position_token_ohe"]["baseline__length"].tolist() == [5.0, 7.0]
    assert set(augmented) == {
        "core_baseline",
        "seqchem",
        "position_token_ohe",
        "position_relative",
        "position_interactions",
        "position_token_relative",
        "position_reencoded_all",
    }


def test_padding_cache_augmentation_preserves_source_and_adds_two_alternatives() -> None:
    helms = pd.Series([HELM_5MER, HELM_7MER])
    baseline = pd.DataFrame({"baseline__length": [5.0, 7.0]})
    existing = {
        "core_baseline": baseline,
        "position_token_ohe": pd.concat(
            [baseline, pd.DataFrame({"position_token_ohe__signal": [1.0, 2.0]})],
            axis=1,
        ),
    }

    augmented = augment_token_padding_families(existing, helms, max_pos=8)

    assert augmented["position_token_ohe"].equals(existing["position_token_ohe"])
    assert set(augmented) == {
        *existing,
        "position_token_ohe_leftpad",
        "position_token_ohe_centerpad",
    }
    assert augmented["position_token_ohe_leftpad"].shape == augmented[
        "position_token_ohe_centerpad"
    ].shape
