"""Sequence × chemistry × position descriptors for ASO toxicity.

The feature block encodes the interaction premise emphasized by Bhamra et al.
2025: sequence context, chemical class, backbone/linkage class, and exact
position context should be modeled together rather than as independent effects.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd

from src.features.chemistry_registry import (
    MODIFIED_SUGAR_CODES,
    SUGAR_CHEMISTRIES,
    base_modification_name,
)
from src.features.chemistry_features import gap_linkage_profile
from src.features.featurizer import parse_helm, parse_helm_components


MOTIFS = ("CG", "GG", "GGG", "TGT", "TGC", "TCC", "TT", "TG")
KNOWN_SUGAR_CODES = tuple(item.code for item in SUGAR_CHEMISTRIES)
SYMBOLIC_SUGAR_TOKENS = {
    "ENA": ("[ena]", "ena"),
    "PRNA": ("[prna]", "prna", "propyl lna", "propyllna"),
    "AL_LNA": ("[al-lna]", "[alpha-l-lna]", "α-l-lna", "alpha-l-lna"),
    "AALNA": ("[aalna]", "[aa-lna]", "aalna", "acetylamino"),
    "F": ("[f]", "2'-f", "2-f", "fluoro"),
}


def _count_overlapping(seq: str, motif: str) -> int:
    return sum(1 for i in range(max(0, len(seq) - len(motif) + 1)) if seq[i : i + len(motif)] == motif)


def _motif_positions(seq: str, motif: str) -> list[int]:
    return [i for i in range(max(0, len(seq) - len(motif) + 1)) if seq[i : i + len(motif)] == motif]


def _span_overlaps_region(start: int, motif_len: int, region: set[int]) -> bool:
    return any(pos in region for pos in range(start, start + motif_len))


def _region_sets(n: int, gap_start: int | None, gap_end: int | None) -> dict[str, set[int]]:
    if gap_start is None or gap_end is None or gap_end <= gap_start:
        # Fallback: use approximate wing/gap thirds if no DNA gap was parsed.
        wing = min(5, max(n // 4, 1))
        gap_start, gap_end = wing, max(wing, n - wing)
    gap = set(range(max(0, gap_start), min(n, gap_end)))
    wing5 = set(range(0, max(0, min(n, gap_start))))
    wing3 = set(range(max(0, min(n, gap_end)), n))
    gap_first3 = set(range(max(0, gap_start), min(n, gap_start + 3)))
    gap_mid = gap - gap_first3 - set(range(max(0, gap_end - 3), min(n, gap_end)))
    return {
        "wing5": wing5,
        "wing3": wing3,
        "gap": gap,
        "gap_first3": gap_first3,
        "gap_mid": gap_mid,
    }


def _normalized_mod_pattern(mod_pattern: str, n: int) -> str:
    out = (mod_pattern or "")[:n]
    return out + ("?" * max(0, n - len(out)))


def _symbolic_token_flags(helm: str) -> dict[str, float]:
    lower = str(helm).lower()
    out = {}
    for name, tokens in SYMBOLIC_SUGAR_TOKENS.items():
        out[f"scint_has_symbolic_{name}"] = float(any(tok.lower() in lower for tok in tokens))
    return out


def features_for_helm(helm: str, max_pos: int = 30) -> dict[str, float]:
    """Compute named sequence-chemistry interaction descriptors for one HELM."""
    mod_pattern, bases = parse_helm(str(helm))
    seq = bases.upper().replace("U", "T")
    n = len(seq)
    mods = _normalized_mod_pattern(mod_pattern, n)
    residues = parse_helm_components(str(helm))
    base_modifications = tuple(base_modification_name(item.raw_base) for item in residues)
    gap = gap_linkage_profile(str(helm))
    gap_start = gap.get("gap_start")
    gap_end = gap.get("gap_end")
    regions = _region_sets(n, gap_start, gap_end)
    linkages = gap.get("linkages", [])
    linkage_counts = Counter(linkages)
    gap_links = gap.get("gap_linkages", [])
    by_gap_rel = {int(x["gap_relative_position"]): str(x["linkage"]) for x in gap_links}

    feats: dict[str, float] = {
        "scint_seq_len": float(n),
        "scint_gap_length": float(gap.get("gap_length", 0) or 0),
        "scint_wing5_len": float(len(regions["wing5"])),
        "scint_wing3_len": float(len(regions["wing3"])),
        "scint_gap_fraction": float(len(regions["gap"]) / max(n, 1)),
        "scint_mod_transition_count": float(sum(1 for a, b in zip(mods, mods[1:]) if a != b)),
        "scint_unique_sugar_count": float(len(set(mods) - {"?"})),
        "scint_gap_ps_fraction": float(
            sum(1 for x in gap_links if x["linkage"] == "PS") / max(len(gap_links), 1)
        ),
        "scint_gap_neutral_linkage_fraction": float(
            sum(1 for x in gap_links if x["linkage"] in {"MP", "MOP"}) / max(len(gap_links), 1)
        ),
        "scint_gap_mspa_fraction": float(
            sum(1 for x in gap_links if x["linkage"] == "MSPA") / max(len(gap_links), 1)
        ),
    }
    for code in KNOWN_SUGAR_CODES:
        positions = {idx for idx, mod in enumerate(mods) if mod == code}
        feats[f"scint_{code}_fraction"] = float(len(positions) / max(n, 1))
        for region_name, region in regions.items():
            feats[f"scint_{region_name}_{code}_count"] = float(len(positions & region))
            feats[f"scint_{region_name}_{code}_fraction"] = float(len(positions & region) / max(len(region), 1))

    for base_modification in ("m5C",):
        positions = {
            index for index, value in enumerate(base_modifications)
            if value == base_modification
        }
        feats[f"scint_basechem_{base_modification}_fraction"] = float(
            len(positions) / max(n, 1)
        )
        for region_name, region in regions.items():
            feats[f"scint_{region_name}_basechem_{base_modification}_count"] = float(
                len(positions & region)
            )
            feats[f"scint_{region_name}_basechem_{base_modification}_fraction"] = float(
                len(positions & region) / max(len(region), 1)
            )

    for linkage in ("PS", "PO", "MP", "MOP", "MSPA"):
        feats[f"scint_linkage_count_{linkage}"] = float(linkage_counts.get(linkage, 0))
        feats[f"scint_gap_pos2_{linkage}"] = float(by_gap_rel.get(2) == linkage)
        feats[f"scint_gap_pos3_{linkage}"] = float(by_gap_rel.get(3) == linkage)

    for motif in MOTIFS:
        positions = _motif_positions(seq, motif)
        tag = motif.replace(" ", "_")
        feats[f"scint_motif_{tag}_count"] = float(len(positions))
        for region_name, region in regions.items():
            feats[f"scint_motif_{tag}_overlap_{region_name}"] = float(
                sum(1 for start in positions if _span_overlaps_region(start, len(motif), region))
            )
        for code in tuple(item.code for item in SUGAR_CHEMISTRIES if item.modified):
            code_positions = {idx for idx, mod in enumerate(mods) if mod == code}
            feats[f"scint_motif_{tag}_overlap_{code}"] = float(
                sum(1 for start in positions if _span_overlaps_region(start, len(motif), code_positions))
            )
        modified_base_positions = {
            idx for idx, value in enumerate(base_modifications) if value == "m5C"
        }
        feats[f"scint_motif_{tag}_overlap_basechem_m5C"] = float(
            sum(
                1
                for start in positions
                if _span_overlaps_region(start, len(motif), modified_base_positions)
            )
        )
        feats[f"scint_motif_{tag}_with_gap_pos2_PS"] = feats[f"scint_motif_{tag}_overlap_gap"] * feats[
            "scint_gap_pos2_PS"
        ]
        feats[f"scint_motif_{tag}_with_gap_neutral_linkage"] = feats[
            f"scint_motif_{tag}_overlap_gap"
        ] * feats["scint_gap_neutral_linkage_fraction"]

    # Bhamra 2025 highlights TGT as a toxic trimer whose electrostatic context
    # changes under 2'-OMe or neutral MP chemistry.
    feats["scint_TGT_gap_PS_context"] = feats["scint_motif_TGT_overlap_gap"] * feats["scint_gap_ps_fraction"]
    feats["scint_TGT_gap_neutral_context"] = feats["scint_motif_TGT_overlap_gap"] * feats[
        "scint_gap_neutral_linkage_fraction"
    ]
    feats["scint_gap_pos2_or_3_neutral"] = float(
        by_gap_rel.get(2) in {"MP", "MOP"} or by_gap_rel.get(3) in {"MP", "MOP"}
    )
    feats["scint_gap_pos2_or_3_PS"] = float(by_gap_rel.get(2) == "PS" or by_gap_rel.get(3) == "PS")
    for i in range(max_pos):
        mod = mods[i] if i < n else "?"
        feats[f"scint_pos_{i+1:02d}_modified_in_gap"] = float(
            i in regions["gap"] and mod in MODIFIED_SUGAR_CODES
        )
        feats[f"scint_pos_{i+1:02d}_BNA_wing"] = float(i in (regions["wing5"] | regions["wing3"]) and mod == "L")
        feats[f"scint_pos_{i+1:02d}_basechem_m5C"] = float(
            i < len(base_modifications) and base_modifications[i] == "m5C"
        )

    feats.update(_symbolic_token_flags(str(helm)))
    return feats


def dataframe_for_helms(helms: Iterable[str], max_pos: int = 30) -> pd.DataFrame:
    records = []
    for helm in helms:
        try:
            records.append(features_for_helm(str(helm), max_pos=max_pos))
        except Exception:
            records.append({})
    return pd.DataFrame(records).fillna(0.0).astype(float)


def bhamra_rule_summary() -> dict[str, str]:
    """Human-readable summary of paper-derived interpretation constraints."""
    return {
        "source": "Bhamra et al. ChemBioChem 2025, DOI 10.1002/cbic.202500584",
        "core_claim": "ASO toxicity depends on sequence-chemistry-position interactions, not sequence or chemistry alone.",
        "model_use": "Use named interaction descriptors and grouped validation; avoid universal hard rules.",
        "data_limit": "Main PDF references SI tables for sequence-level data; do not train paper labels without SI extraction.",
    }
