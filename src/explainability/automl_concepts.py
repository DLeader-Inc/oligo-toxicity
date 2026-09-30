"""Concept aggregation for the feature-selected AutoML system."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.automl_descriptor_routes import descriptor_family


MECHANISM_AXES = (
    "sequence_identity_motif",
    "sugar_modification",
    "backbone_linkage",
    "modified_base",
    "wing_gap_architecture",
    "sequence_chemistry_interaction",
    "assay_exposure_context",
)


def mechanism_axis_for_feature(name: str) -> str:
    """Assign one molecular-information axis independent of legacy family labels.

    The legacy Registry/motif/chemical-modification families are implementation
    partitions.  They are column-disjoint but not information-disjoint: raw
    Registry columns encode sugar and linkage state, while engineered motif
    columns can encode regional sequence context.  This second mapping is used
    for scientific interpretation and keeps those two questions separate.
    """

    value = str(name)
    upper = value.upper()
    if value.startswith("assay__") or any(token in upper for token in ("DOSE", "ROUTE", "SPECIES", "TIMEPOINT")):
        return "assay_exposure_context"
    # Explicit cross-terms take precedence over their component tokens.
    if upper.startswith(("SCINT_", "PINT_BASE_", "PINT_MOTIF_")) or (
        any(token in upper for token in ("KMER", "MOTIF", "BASE_"))
        and any(token in upper for token in ("SUGAR", "MOD_", "LINK", "WING", "GAP", "REGION"))
    ):
        return "sequence_chemistry_interaction"
    if any(token in upper for token in ("5MC", "M5C", "5MEC", "MODIFIED_BASE")):
        return "modified_base"
    if any(token in upper for token in ("LINK", "MSPA", "MOP", "PHOSPHATE", "BACKBONE")) or re.search(
        r"(?:^|_)(?:PS|PO|MP)(?:_|$)", upper
    ):
        return "backbone_linkage"
    if any(token in upper for token in ("LNA", "MOE", "2OME", "2'-O", "CET", "SUGAR")) or re.search(
        r"(?:MOD|FRACTION|COUNT)_[LMODR](?:_|$)", upper
    ):
        return "sugar_modification"
    if any(token in upper for token in ("GAP", "WING", "REGION", "LANDMARK", "RELATIVE_POS")):
        return "wing_gap_architecture"
    return "sequence_identity_motif"


def concept_for_feature(name: str) -> str:
    value = str(name)
    upper = value.upper()
    if value.startswith("assay__"):
        return "fixed_assay_context"
    if any(token in upper for token in ("TCC", "TGC", "CPG", "_CG", "G4", "GGG")):
        return "literature_linked_or_g_rich_motif"
    if "KMER" in upper or "MOTIF" in upper or any(f"BASE_{base}" in upper for base in "ACGT"):
        return "sequence_motif_and_position"
    if "LNA" in upper or re.search(r"(?:SUGAR|MOD|FRACTION|COUNT)_L(?:_|$)", upper) or upper.startswith("SCINT_L_"):
        return "lna_amount_and_position"
    if "MOE" in upper or re.search(r"(?:SUGAR|MOD|FRACTION|COUNT)_M(?:_|$)", upper) or upper.startswith("SCINT_M_"):
        return "moe_amount_and_position"
    if "CET" in upper or "5MC" in upper or "M5C" in upper:
        return "cet_or_modified_base_context"
    if "LINK" in upper or "_PS" in upper or "_PO" in upper or "MSPA" in upper:
        return "backbone_amount_and_position"
    if "GAP" in upper or "WING" in upper or "REGION" in upper or "POS_" in upper:
        return "gapmer_architecture_and_position"
    family = descriptor_family(value)
    if family == "registry":
        return "registry_token_identity"
    if family == "motif":
        return "sequence_composition"
    return "other_chemical_modification_context"


def aggregate_contributions(frame: pd.DataFrame) -> pd.DataFrame:
    """Sum signed SHAP within member/scale/concept and rank by mean absolute mass."""

    required = {"member_id", "model_scale", "feature", "contribution"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing contribution columns: {sorted(missing)}")
    work = frame.copy()
    work["concept"] = work.feature.map(concept_for_feature)
    grouped = work.groupby(
        ["member_id", "model_scale", "concept"], as_index=False
    ).agg(
        contribution=("contribution", "sum"),
        absolute_feature_mass=("contribution", lambda values: float(np.abs(values).sum())),
        feature_count=("feature", "nunique"),
    )
    summary = grouped.groupby(["model_scale", "concept"], as_index=False).agg(
        mean_signed_contribution=("contribution", "mean"),
        mean_absolute_contribution=("contribution", lambda values: float(np.abs(values).mean())),
        member_min=("contribution", "min"), member_max=("contribution", "max"),
        members=("member_id", "nunique"), feature_count=("feature_count", "max"),
    )
    summary["rank"] = summary.groupby("model_scale").mean_absolute_contribution.rank(
        method="first", ascending=False
    ).astype(int)
    return summary.sort_values(["model_scale", "rank"])


def high_correlation_clusters(
    matrix: np.ndarray,
    names: Iterable[str],
    *, threshold: float = 0.90,
    candidate_names: set[str] | None = None,
) -> list[dict[str, object]]:
    """Greedy training-only absolute-Spearman clusters for a bounded feature set."""

    all_names = list(map(str, names))
    indices = [
        index for index, name in enumerate(all_names)
        if candidate_names is None or name in candidate_names
    ]
    if not indices:
        return []
    values = pd.DataFrame(np.asarray(matrix)[:, indices], columns=[all_names[i] for i in indices])
    values = values.loc[:, values.nunique(dropna=False).gt(1)]
    if values.empty:
        return []
    correlation = values.corr(method="spearman").abs().fillna(0.0)
    remaining = list(correlation.columns)
    clusters: list[dict[str, object]] = []
    while remaining:
        seed = remaining[0]
        members = [name for name in remaining if correlation.loc[seed, name] >= threshold]
        clusters.append({
            "cluster_id": f"cluster_{len(clusters) + 1:03d}",
            "representative": seed,
            "concept": concept_for_feature(seed),
            "members": members,
            "size": len(members),
            "threshold": threshold,
        })
        member_set = set(members)
        remaining = [name for name in remaining if name not in member_set]
    return clusters
