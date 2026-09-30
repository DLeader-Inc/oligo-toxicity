"""Compact, separable positional representations for modified oligonucleotides.

The original ``position`` family deliberately exposes several redundant views in
one large table.  This module separates three hypotheses so nested validation can
decide whether an endpoint needs direct token identity, relative coordinates, or
explicit local interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src.features.chemistry_registry import SUGAR_DESCRIPTOR_CODES
from src.features.featurizer import parse_helm_components
from src.features.sequence_chemistry_interaction_features import MOTIFS, _motif_positions


BASES = ("A", "C", "G", "T")
MODIFICATIONS = SUGAR_DESCRIPTOR_CODES
LINKAGES = ("PS", "PO", "MP", "MOP", "MSPA", "OTHER")
TOKEN_LINKAGES = (*LINKAGES, "TERMINAL")
REGIONS = ("wing5", "gap_first3", "gap_mid", "gap_last3", "wing3", "other")
TokenAlignment = Literal["right", "left", "center"]


@dataclass(frozen=True)
class PositionContext:
    sequence: str
    modifications: tuple[str, ...]
    outgoing_linkages: tuple[str, ...]
    regions: tuple[str, ...]
    gap_start: int
    gap_end: int

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def sequence_center(self) -> float:
        return (self.length - 1) / 2.0 if self.length else 0.0

    @property
    def gap_center(self) -> float:
        return (self.gap_start + self.gap_end - 1) / 2.0 if self.length else 0.0


def _modification(value: str) -> str:
    return value if value in MODIFICATIONS[:-1] else "OTHER"


def _linkage(value: str) -> str:
    normalized = str(value).upper()
    return normalized if normalized in LINKAGES[:-1] else "OTHER"


def _region(index: int, gap_start: int, gap_end: int) -> str:
    if index < gap_start:
        return "wing5"
    if index >= gap_end:
        return "wing3"
    if index < gap_start + 3:
        return "gap_first3"
    if index >= gap_end - 3:
        return "gap_last3"
    return "gap_mid"


def _context(helm: str) -> PositionContext:
    residues = parse_helm_components(str(helm))
    sequence = "".join(residue.base for residue in residues)
    n = len(sequence)
    modifications = tuple(
        _modification(residue.modification) for residue in residues
    )
    outgoing = [
        residue.linkage
        if residue.linkage in TOKEN_LINKAGES
        else "OTHER"
        for residue in residues
    ]

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, modification in enumerate((*modifications, "__STOP__")):
        if modification == "D":
            if start is None:
                start = index
            continue
        if start is not None:
            runs.append((start, index))
            start = None
    gap_start, gap_end = max(
        runs,
        key=lambda bounds: bounds[1] - bounds[0],
        default=(0, 0),
    )
    if gap_end <= gap_start:
        wing = min(5, max(n // 4, 1)) if n else 0
        gap_start, gap_end = wing, max(wing, n - wing)
    gap_start = max(0, min(int(gap_start), n))
    gap_end = max(gap_start, min(int(gap_end), n))
    regions = tuple(_region(index, gap_start, gap_end) for index in range(n))
    return PositionContext(
        sequence=sequence,
        modifications=modifications,
        outgoing_linkages=tuple(outgoing),
        regions=regions,
        gap_start=gap_start,
        gap_end=gap_end,
    )


def _slot(
    features: dict[str, float],
    prefix: str,
    context: PositionContext,
    index: int,
    *,
    include_region: bool = False,
) -> None:
    exists = 0 <= index < context.length
    features[f"{prefix}_exists"] = float(exists)
    base = context.sequence[index] if exists else ""
    modification = context.modifications[index] if exists else ""
    linkage = context.outgoing_linkages[index] if exists else ""
    for value in BASES:
        features[f"{prefix}_base_{value}"] = float(base == value)
    for value in MODIFICATIONS:
        features[f"{prefix}_mod_{value}"] = float(modification == value)
    for value in TOKEN_LINKAGES:
        features[f"{prefix}_link_{value}"] = float(linkage == value)
    if include_region:
        region = context.regions[index] if exists else ""
        for value in REGIONS:
            features[f"{prefix}_region_{value}"] = float(region == value)


def _token_canvas_layout(
    length: int,
    max_pos: int,
    alignment: TokenAlignment,
) -> tuple[int, int, int]:
    """Return canvas start, sequence start, and represented token count."""

    if max_pos < 1:
        raise ValueError("max_pos must be positive")
    represented = min(length, max_pos)
    if alignment == "right":
        return 0, 0, represented
    if alignment == "left":
        return max(max_pos - length, 0), max(length - max_pos, 0), represented
    if alignment == "center":
        return max((max_pos - length) // 2, 0), max((length - max_pos) // 2, 0), represented
    raise ValueError(f"unsupported token alignment: {alignment!r}")


def token_ohe_features_for_helm(
    helm: str,
    max_pos: int = 30,
    *,
    alignment: TokenAlignment = "right",
) -> dict[str, float]:
    """Return base/sugar/linkage one-hot tokens on a fixed positional canvas.

    ``right`` retains the historical 5'-aligned representation and pads after
    the sequence. ``left`` aligns the 3' terminus and pads before the sequence.
    ``center`` divides padding around the sequence, placing the extra zero on
    the 3' side when an odd number of padding slots is required.
    """

    context = _context(helm)
    features: dict[str, float] = {}
    canvas_start, sequence_start, represented = _token_canvas_layout(
        context.length, max_pos, alignment
    )
    prefix = {
        "right": "ptok_5p",
        "left": "ptok_leftpad",
        "center": "ptok_centerpad",
    }[alignment]
    for canvas_index in range(max_pos):
        relative_index = canvas_index - canvas_start
        sequence_index = (
            sequence_start + relative_index
            if 0 <= relative_index < represented
            else -1
        )
        _slot(
            features,
            f"{prefix}_{canvas_index + 1:02d}",
            context,
            sequence_index,
        )
    return features


def token_ohe_dataframe_for_helms(
    helms: Iterable[str],
    *,
    max_pos: int = 30,
    alignment: TokenAlignment = "right",
) -> pd.DataFrame:
    """Build one aligned token block while preserving failed rows as zeros."""

    rows: list[dict[str, float]] = []
    for helm in helms:
        try:
            rows.append(
                token_ohe_features_for_helm(
                    str(helm), max_pos=max_pos, alignment=alignment
                )
            )
        except Exception:
            rows.append({})
    return pd.DataFrame(rows).fillna(0.0).astype(float)


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _category_coordinate_summary(
    features: dict[str, float],
    prefix: str,
    positions: list[int],
    context: PositionContext,
) -> None:
    n = context.length
    denom = max(n - 1, 1)
    center_denom = max(context.sequence_center, 1.0)
    gap_denom = max(context.gap_end - context.gap_start, 1)
    features[f"{prefix}_count"] = float(len(positions))
    features[f"{prefix}_fraction"] = float(len(positions) / max(n, 1))
    features[f"{prefix}_5p_norm_mean"] = _mean_or_zero([index / denom for index in positions])
    features[f"{prefix}_3p_norm_mean"] = _mean_or_zero([(n - 1 - index) / denom for index in positions])
    features[f"{prefix}_center_norm_mean"] = _mean_or_zero(
        [(index - context.sequence_center) / center_denom for index in positions]
    )
    features[f"{prefix}_center_abs_mean"] = _mean_or_zero(
        [abs(index - context.sequence_center) / center_denom for index in positions]
    )
    features[f"{prefix}_gap_center_abs_mean"] = _mean_or_zero(
        [abs(index - context.gap_center) / gap_denom for index in positions]
    )


def relative_features_for_helm(
    helm: str,
    *,
    terminal_window: int = 5,
    center_radius: int = 4,
    gap_offset_min: int = -3,
    gap_offset_max: int = 12,
) -> dict[str, float]:
    """Return 3', center, and DNA-gap-relative positional features."""

    context = _context(helm)
    n = context.length
    denom = max(n - 1, 1)
    features: dict[str, float] = {
        "prel_seq_len": float(n),
        "prel_gap_start_norm": float(context.gap_start / max(n, 1)),
        "prel_gap_end_norm": float(context.gap_end / max(n, 1)),
        "prel_gap_center_norm": float(context.gap_center / denom),
        "prel_gap_length_fraction": float((context.gap_end - context.gap_start) / max(n, 1)),
    }

    for distance in range(terminal_window):
        _slot(features, f"prel_3p_{distance + 1:02d}", context, n - 1 - distance)

    center_anchor = n // 2
    for offset in range(-center_radius, center_radius + 1):
        tag = f"m{abs(offset):02d}" if offset < 0 else f"p{offset:02d}"
        index = center_anchor + offset
        _slot(features, f"prel_center_{tag}", context, index)
        features[f"prel_center_{tag}_in_gap"] = float(
            0 <= index < n and context.gap_start <= index < context.gap_end
        )

    for offset in range(gap_offset_min, gap_offset_max + 1):
        tag = f"m{abs(offset):02d}" if offset < 0 else f"p{offset:02d}"
        _slot(features, f"prel_gapstart_{tag}", context, context.gap_start + offset)

    for base in BASES:
        positions = [index for index, observed in enumerate(context.sequence) if observed == base]
        _category_coordinate_summary(features, f"prel_base_{base}", positions, context)
    for modification in MODIFICATIONS:
        positions = [
            index for index, observed in enumerate(context.modifications) if observed == modification
        ]
        _category_coordinate_summary(features, f"prel_mod_{modification}", positions, context)
    for linkage in LINKAGES:
        positions = [
            index
            for index, observed in enumerate(context.outgoing_linkages[:-1])
            if observed == linkage
        ]
        _category_coordinate_summary(features, f"prel_link_{linkage}", positions, context)

    for motif in MOTIFS:
        starts = _motif_positions(context.sequence, motif)
        centers = [start + (len(motif) - 1) / 2.0 for start in starts]
        tag = motif.replace(" ", "_")
        features[f"prel_motif_{tag}_count"] = float(len(starts))
        features[f"prel_motif_{tag}_center_norm_mean"] = _mean_or_zero(
            [(center - context.sequence_center) / max(context.sequence_center, 1.0) for center in centers]
        )
        features[f"prel_motif_{tag}_gap_center_abs_mean"] = _mean_or_zero(
            [
                abs(center - context.gap_center) / max(context.gap_end - context.gap_start, 1)
                for center in centers
            ]
        )
        features[f"prel_motif_{tag}_in_gap_count"] = float(
            sum(context.gap_start <= start < context.gap_end for start in starts)
        )
        features[f"prel_motif_{tag}_near_5p_count"] = float(sum(start < 5 for start in starts))
        features[f"prel_motif_{tag}_near_3p_count"] = float(
            sum(n - (start + len(motif)) < 5 for start in starts)
        )
    return features


def interaction_features_for_helm(helm: str) -> dict[str, float]:
    """Return sparse base-chemistry-region and motif-location interactions."""

    context = _context(helm)
    features: dict[str, float] = {}
    for base in BASES:
        for modification in MODIFICATIONS:
            for region in REGIONS:
                features[f"pint_base_{base}__mod_{modification}__region_{region}"] = 0.0
        for linkage in LINKAGES:
            for region in REGIONS:
                features[f"pint_base_{base}__link_{linkage}__region_{region}"] = 0.0
    for motif in MOTIFS:
        for region in REGIONS:
            features[f"pint_motif_{motif}__region_{region}"] = 0.0
        for modification in MODIFICATIONS:
            features[f"pint_motif_{motif}__mod_{modification}"] = 0.0
    for left in MODIFICATIONS:
        for right in MODIFICATIONS:
            features[f"pint_mod_transition_{left}_{right}"] = 0.0
    for left in LINKAGES:
        for right in LINKAGES:
            features[f"pint_link_transition_{left}_{right}"] = 0.0

    for index, base in enumerate(context.sequence):
        if base not in BASES:
            continue
        modification = context.modifications[index]
        region = context.regions[index]
        features[f"pint_base_{base}__mod_{modification}__region_{region}"] += 1.0
        linkage = context.outgoing_linkages[index]
        if linkage in LINKAGES:
            features[f"pint_base_{base}__link_{linkage}__region_{region}"] += 1.0

    for left, right in zip(context.modifications, context.modifications[1:]):
        features[f"pint_mod_transition_{left}_{right}"] += 1.0
    nonterminal_linkages = [value for value in context.outgoing_linkages if value in LINKAGES]
    for left, right in zip(nonterminal_linkages, nonterminal_linkages[1:]):
        features[f"pint_link_transition_{left}_{right}"] += 1.0

    for motif in MOTIFS:
        starts = _motif_positions(context.sequence, motif)
        for start in starts:
            center_index = min(start + len(motif) // 2, max(context.length - 1, 0))
            if context.length:
                features[f"pint_motif_{motif}__region_{context.regions[center_index]}"] += 1.0
            observed_modifications = set(context.modifications[start : start + len(motif)])
            for modification in observed_modifications:
                features[f"pint_motif_{motif}__mod_{modification}"] += 1.0
    return features


def dataframes_for_helms(
    helms: Iterable[str],
    *,
    max_pos: int = 30,
) -> dict[str, pd.DataFrame]:
    """Build all re-encoded blocks while preserving a shared row contract."""

    records: dict[str, list[dict[str, float]]] = {
        "position_token_ohe": [],
        "position_relative": [],
        "position_interactions": [],
    }
    for helm in helms:
        try:
            row_blocks = {
                "position_token_ohe": token_ohe_features_for_helm(
                    str(helm), max_pos=max_pos, alignment="right"
                ),
                "position_relative": relative_features_for_helm(str(helm)),
                "position_interactions": interaction_features_for_helm(str(helm)),
            }
        except Exception:
            row_blocks = {name: {} for name in records}
        for name, values in records.items():
            values.append(row_blocks[name])
    return {
        name: pd.DataFrame(values).fillna(0.0).astype(float)
        for name, values in records.items()
    }
