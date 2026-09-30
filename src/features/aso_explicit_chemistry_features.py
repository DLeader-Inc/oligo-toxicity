"""Versioned explicit cEt and 5meC channels for ASO HELM records.

This additive block preserves the frozen universal v2 representation while
making two common ASO Atlas chemistries named and locally attributable.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.features.featurizer import parse_helm_components


SCHEMA_VERSION = "aso_explicit_cet_5mec_v1"
MAX_POSITIONS = 30
REGIONS = ("wing5", "gap_first3", "gap_mid", "gap_last3", "wing3", "other")


def _is_cet(raw_sugar: str) -> bool:
    return str(raw_sugar).strip().lower() == "cet"


def _is_5mec(raw_base: str) -> bool:
    token = str(raw_base).strip().upper().replace("-", "").replace("_", "")
    return token in {"5MEC", "M5C", "5MC"}


def canonical_aso_sequence(helm: str) -> str:
    """Return chemistry-independent bases while treating 5meC/m5C as C."""

    bases = []
    for residue in parse_helm_components(str(helm)):
        token = str(residue.raw_base).strip().upper().replace("-", "").replace("_", "")
        if token in {"5MEC", "M5C", "5MC"}:
            bases.append("C")
        elif token in {"A", "C", "G", "T", "U"}:
            bases.append("T" if token == "U" else token)
        else:
            bases.append(residue.base)
    return "".join(bases)


def _regions(modifications: list[str]) -> list[str]:
    n = len(modifications)
    dna = [index for index, value in enumerate(modifications) if value == "D"]
    if not dna:
        return ["other"] * n
    gap_start, gap_end = min(dna), max(dna) + 1
    labels = []
    for index in range(n):
        if index < gap_start:
            labels.append("wing5")
        elif index >= gap_end:
            labels.append("wing3")
        elif index < gap_start + 3:
            labels.append("gap_first3")
        elif index >= max(gap_start + 3, gap_end - 3):
            labels.append("gap_last3")
        else:
            labels.append("gap_mid")
    return labels


def feature_names(max_positions: int = MAX_POSITIONS) -> tuple[str, ...]:
    names = []
    for position in range(1, max_positions + 1):
        names.extend(
            (
                f"aso3_pos_{position:02d}_sugar_cEt",
                f"aso3_pos_{position:02d}_base_5meC",
            )
        )
    names.extend(("aso3_n_cEt", "aso3_frac_cEt", "aso3_n_5meC", "aso3_frac_5meC"))
    for region in REGIONS:
        for chemistry in ("cEt", "5meC"):
            names.extend(
                (
                    f"aso3_{region}_{chemistry}_count",
                    f"aso3_{region}_{chemistry}_fraction",
                )
            )
    names.extend(
        (
            "aso3_5prime_cEt",
            "aso3_3prime_cEt",
            "aso3_5prime_5meC",
            "aso3_3prime_5meC",
            "aso3_n_cEt_5meC_same_residue",
            "aso3_frac_cEt_5meC_same_residue",
            "aso3_cEt_transition_count",
            "aso3_5meC_transition_count",
        )
    )
    return tuple(names)


def features_for_helm(helm: str, max_positions: int = MAX_POSITIONS) -> dict[str, float]:
    residues = list(parse_helm_components(str(helm)))
    n = len(residues)
    cet = [_is_cet(residue.raw_sugar) for residue in residues]
    mec = [_is_5mec(residue.raw_base) for residue in residues]
    regions = _regions([residue.modification for residue in residues])
    values = {name: 0.0 for name in feature_names(max_positions)}

    for index in range(min(n, max_positions)):
        values[f"aso3_pos_{index + 1:02d}_sugar_cEt"] = float(cet[index])
        values[f"aso3_pos_{index + 1:02d}_base_5meC"] = float(mec[index])

    values["aso3_n_cEt"] = float(sum(cet))
    values["aso3_frac_cEt"] = float(sum(cet) / max(n, 1))
    values["aso3_n_5meC"] = float(sum(mec))
    values["aso3_frac_5meC"] = float(sum(mec) / max(n, 1))
    for region in REGIONS:
        selected = [index for index, label in enumerate(regions) if label == region]
        for chemistry, flags in (("cEt", cet), ("5meC", mec)):
            count = sum(flags[index] for index in selected)
            values[f"aso3_{region}_{chemistry}_count"] = float(count)
            values[f"aso3_{region}_{chemistry}_fraction"] = float(
                count / max(len(selected), 1)
            )

    if n:
        values["aso3_5prime_cEt"] = float(cet[0])
        values["aso3_3prime_cEt"] = float(cet[-1])
        values["aso3_5prime_5meC"] = float(mec[0])
        values["aso3_3prime_5meC"] = float(mec[-1])
    cooccurrence = sum(left and right for left, right in zip(cet, mec))
    values["aso3_n_cEt_5meC_same_residue"] = float(cooccurrence)
    values["aso3_frac_cEt_5meC_same_residue"] = float(cooccurrence / max(n, 1))
    values["aso3_cEt_transition_count"] = float(
        sum(left != right for left, right in zip(cet, cet[1:]))
    )
    values["aso3_5meC_transition_count"] = float(
        sum(left != right for left, right in zip(mec, mec[1:]))
    )
    return values


def dataframe_for_helms(
    helms: Iterable[str], max_positions: int = MAX_POSITIONS
) -> pd.DataFrame:
    names = feature_names(max_positions)
    rows = []
    for helm in helms:
        try:
            rows.append(features_for_helm(str(helm), max_positions=max_positions))
        except Exception:
            rows.append({name: 0.0 for name in names})
    return pd.DataFrame(rows, columns=names).fillna(0.0).astype(float)
