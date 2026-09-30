from src.explainability.automl_concepts import MECHANISM_AXES, mechanism_axis_for_feature


def test_registry_columns_reveal_information_overlap():
    assert mechanism_axis_for_feature("raw_pos_01_sugar_lna") == "sugar_modification"
    assert mechanism_axis_for_feature("raw_pos_02_linkage_PS") == "backbone_linkage"
    assert mechanism_axis_for_feature("raw_pos_03_base_m5C") == "modified_base"


def test_engineered_mechanism_axes_are_specific():
    assert mechanism_axis_for_feature("scint_motif_TCC_mod_LNA") == "sequence_chemistry_interaction"
    assert mechanism_axis_for_feature("gap_length") == "wing_gap_architecture"
    assert mechanism_axis_for_feature("kmer_TCC") == "sequence_identity_motif"
    assert mechanism_axis_for_feature("assay__dose") == "assay_exposure_context"


def test_axis_contract_is_closed():
    names = ["raw_pos_01_base_A", "moe_fraction", "link_PS_count", "m5C_count", "wing_length"]
    assert {mechanism_axis_for_feature(name) for name in names}.issubset(set(MECHANISM_AXES))
