from src.features.sequence_chemistry_interaction_features import (
    bhamra_rule_summary,
    dataframe_for_helms,
    features_for_helm,
)


def test_sequence_chemistry_interactions_capture_gap_motif_and_linkage_context():
    helm = (
        "RNA1{[lna](A)[sp].[lna](C)[sp].[lna](T)[sp]."
        "d(T)[sp].d(G)[mp].d(T)[sp].d(G)[sp].d(G)[sp]."
        "[lna](A)[sp].[lna](C)[sp].[lna](T)}$$$$"
    )

    feats = features_for_helm(helm)

    assert feats["scint_gap_length"] == 5
    assert feats["scint_motif_TGT_count"] == 1
    assert feats["scint_motif_TGT_overlap_gap"] == 1
    assert feats["scint_gap_pos2_MP"] == 1
    assert feats["scint_gap_pos2_or_3_neutral"] == 1
    assert feats["scint_TGT_gap_neutral_context"] > 0


def test_sequence_chemistry_interaction_dataframe_is_numeric_and_stable():
    helms = [
        "RNA1{[moe](C)[sp].[moe](G)p.d(T)[sp].d(G)[sp].d(T)[sp].[moe](A)}$$$$",
        "RNA1{[lna](T)[sp].d(G)[sp].d(T)[sp].[lna](A)}$$$$",
    ]

    df = dataframe_for_helms(helms, max_pos=6)

    assert len(df) == 2
    assert "scint_TGT_gap_PS_context" in df.columns
    assert df.select_dtypes(exclude=["number"]).empty
    assert df.fillna(0).equals(df)


def test_bhamra_rule_summary_marks_rules_as_interaction_not_causal_labels():
    summary = bhamra_rule_summary()

    assert "10.1002/cbic.202500584" in summary["source"]
    assert "interaction" in summary["core_claim"].lower()
    assert "do not train paper labels without SI" in summary["data_limit"]
