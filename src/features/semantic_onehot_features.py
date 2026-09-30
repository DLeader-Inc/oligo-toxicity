"""Compact semantic one-hot representations for modified oligonucleotides.

The representation starts from factorized nucleotide identity and adds
descriptor-family evidence as fixed, named discrete channels. It deliberately
avoids a learned composite-token vocabulary so unseen base/sugar/linkage
combinations remain representable.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src.features.chemistry_features import gap_linkage_profile
from src.features.position_reencoding_features import (
    LINKAGES as POSITION_LINKAGES,
    MODIFICATIONS,
    REGIONS,
    PositionContext,
    _context,
)
from src.features.secondary_structure_features import FoldResult, fold_sequence
from src.features.sequence_chemistry_interaction_features import (
    MOTIFS,
    SYMBOLIC_SUGAR_TOKENS,
    _motif_positions,
)


BASES = ("A", "C", "G", "T", "N")
LINKAGES = (*POSITION_LINKAGES, "TERMINAL")
STRUCTURE_STATES = ("paired", "stem", "loop", "unpaired", "long_range_pair")
PPB_LOCAL_STATES = (
    "ps",
    "sensitive_gap_ps",
    "sensitive_gap_neutral",
    "early_gap_mspa",
    "ps_g_context",
)
PAIR_MODIFICATION_COMBOS = tuple(combinations_with_replacement(MODIFICATIONS, 2))
MAX_BASE_CANVAS = 30
ANCHOR_OFFSETS = (
    ("5p_01", "5p", 0),
    ("5p_02", "5p", 1),
    ("5p_03", "5p", 2),
    ("3p_01", "3p", 0),
    ("3p_02", "3p", 1),
    ("3p_03", "3p", 2),
    ("center_m01", "center", -1),
    ("center_p00", "center", 0),
    ("center_p01", "center", 1),
    ("gap_01", "gap", 0),
    ("gap_02", "gap", 1),
    ("gap_03", "gap", 2),
)
SemanticMode = Literal[
    "semantic_chem_ohe",
    "semantic_mechanism_ohe",
    "semantic_interaction_ohe",
]
SemanticScope = Literal[
    "position_local",
    "regional_summary",
    "global_summary",
]
SEMANTIC_MODES: tuple[SemanticMode, ...] = (
    "semantic_chem_ohe",
    "semantic_mechanism_ohe",
    "semantic_interaction_ohe",
)
SEMANTIC_SCOPES: tuple[SemanticScope, ...] = (
    "position_local",
    "regional_summary",
    "global_summary",
)


def _anchor_feature_names() -> list[str]:
    names: list[str] = []
    for tag, _, _ in ANCHOR_OFFSETS:
        names.append(f"anchor_{tag}_exists")
        names.extend(f"anchor_{tag}_base_{value}" for value in BASES)
        names.extend(f"anchor_{tag}_sugar_{value}" for value in MODIFICATIONS)
        names.extend(f"anchor_{tag}_link_{value}" for value in LINKAGES)
    return names


def _base_canvas_feature_names() -> list[str]:
    return [
        f"canvas_{position:02d}_base_{base}"
        for position in range(1, MAX_BASE_CANVAS + 1)
        for base in BASES
    ]


def _pool_names(categories: tuple[str, ...], family: str) -> list[str]:
    return [
        f"pool_{region}_{family}_{category}"
        for region in REGIONS
        for category in categories
    ]


def _chemistry_feature_names() -> list[str]:
    names = _base_canvas_feature_names()
    names.extend(_anchor_feature_names())
    names.extend(_pool_names(BASES, "base"))
    names.extend(_pool_names(MODIFICATIONS, "sugar"))
    names.extend(_pool_names(LINKAGES, "link"))
    names.extend(
        [
            *(f"global_length_{label}" for label in ("le14", "15_16", "17_18", "19_20", "21_24", "ge25")),
            *(f"global_gap_length_{label}" for label in ("zero", "1_5", "6_8", "9_10", "ge11")),
            *(f"global_ps_fraction_{label}" for label in ("zero", "le25", "le50", "lt100", "all")),
            *(f"global_gc_fraction_{label}" for label in ("le25", "le50", "le75", "gt75")),
            *(f"global_mod_transitions_{label}" for label in ("zero", "one", "two", "ge3")),
            *(f"global_symbolic_{name}" for name in SYMBOLIC_SUGAR_TOKENS),
            "global_has_explicit_dna_gap",
            "global_canvas_truncated",
        ]
    )
    return names


def _mechanism_feature_names() -> list[str]:
    names = _chemistry_feature_names()
    names.extend(_pool_names(MOTIFS, "motif"))
    names.extend(_pool_names(STRUCTURE_STATES, "structure"))
    names.extend(_pool_names(PPB_LOCAL_STATES, "ppb"))
    names.extend(
        f"global_pair_fraction_{label}"
        for label in ("zero", "le25", "le50", "gt50")
    )
    names.extend(f"global_motif_present_{motif}" for motif in MOTIFS)
    names.append("global_structure_ok")
    names.extend(
        f"pair_sugar_{left}_{right}_fraction"
        for left, right in PAIR_MODIFICATION_COMBOS
    )
    return names


@lru_cache(maxsize=None)
def semantic_feature_names(mode: SemanticMode) -> tuple[str, ...]:
    """Return the stable, ordered schema for one semantic representation."""

    if mode == "semantic_chem_ohe":
        names = _chemistry_feature_names()
    elif mode == "semantic_mechanism_ohe":
        names = _mechanism_feature_names()
    elif mode == "semantic_interaction_ohe":
        names = _mechanism_feature_names()
        names.extend(
            f"interaction_{region}_base_{base}_sugar_{sugar}"
            for region in REGIONS
            for base in BASES
            for sugar in MODIFICATIONS
        )
    else:
        raise ValueError(f"unsupported semantic one-hot mode: {mode!r}")
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate semantic feature names for {mode}")
    return tuple(names)


def semantic_dimensions() -> dict[str, int]:
    """Return fixed dimensions for all semantic representation levels."""

    return {mode: len(semantic_feature_names(mode)) for mode in SEMANTIC_MODES}


def semantic_feature_scope(feature_name: str) -> SemanticScope:
    """Classify a named channel by where it belongs in a hybrid encoder.

    Position-local channels are derived from one explicit residue or anchor.
    Regional and global summaries are molecule-level vectors and should be
    concatenated after flattening or pooling rather than copied to every node.
    """

    name = str(feature_name)
    if name.startswith(("canvas_", "anchor_")):
        return "position_local"
    if name.startswith(("pool_", "interaction_")):
        return "regional_summary"
    if name.startswith(("global_", "pair_sugar_")):
        return "global_summary"
    raise ValueError(f"unknown semantic feature scope: {feature_name!r}")


def _set_one_hot(features: dict[str, float], prefix: str, labels: tuple[str, ...], selected: str) -> None:
    for label in labels:
        features[f"{prefix}_{label}"] = float(label == selected)


def _length_bin(length: int) -> str:
    if length <= 14:
        return "le14"
    if length <= 16:
        return "15_16"
    if length <= 18:
        return "17_18"
    if length <= 20:
        return "19_20"
    if length <= 24:
        return "21_24"
    return "ge25"


def _gap_length_bin(length: int) -> str:
    if length <= 0:
        return "zero"
    if length <= 5:
        return "1_5"
    if length <= 8:
        return "6_8"
    if length <= 10:
        return "9_10"
    return "ge11"


def _ps_fraction_bin(value: float) -> str:
    if value <= 0:
        return "zero"
    if value <= 0.25:
        return "le25"
    if value <= 0.50:
        return "le50"
    if value < 1.0:
        return "lt100"
    return "all"


def _quartile_bin(value: float) -> str:
    if value <= 0.25:
        return "le25"
    if value <= 0.50:
        return "le50"
    if value <= 0.75:
        return "le75"
    return "gt75"


def _transition_bin(value: int) -> str:
    if value <= 0:
        return "zero"
    if value == 1:
        return "one"
    if value == 2:
        return "two"
    return "ge3"


def _pair_fraction_bin(value: float) -> str:
    if value <= 0:
        return "zero"
    if value <= 0.25:
        return "le25"
    if value <= 0.50:
        return "le50"
    return "gt50"


def _anchor_index(
    context: PositionContext,
    kind: str,
    offset: int,
    *,
    has_explicit_gap: bool,
) -> int:
    if kind == "5p":
        return offset
    if kind == "3p":
        return context.length - 1 - offset
    if kind == "center":
        return context.length // 2 + offset
    if kind == "gap":
        return context.gap_start + offset if has_explicit_gap else -1
    raise ValueError(kind)


def _anchor_features(
    context: PositionContext,
    *,
    has_explicit_gap: bool,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for tag, kind, offset in ANCHOR_OFFSETS:
        index = _anchor_index(
            context,
            kind,
            offset,
            has_explicit_gap=has_explicit_gap,
        )
        exists = 0 <= index < context.length
        features[f"anchor_{tag}_exists"] = float(exists)
        base = context.sequence[index] if exists else ""
        sugar = context.modifications[index] if exists else ""
        linkage = context.outgoing_linkages[index] if exists else ""
        _set_one_hot(features, f"anchor_{tag}_base", BASES, base)
        _set_one_hot(features, f"anchor_{tag}_sugar", MODIFICATIONS, sugar)
        _set_one_hot(features, f"anchor_{tag}_link", LINKAGES, linkage)
    return features


def _base_canvas_features(context: PositionContext) -> dict[str, float]:
    features: dict[str, float] = {}
    for position in range(MAX_BASE_CANVAS):
        observed = context.sequence[position] if position < context.length else ""
        _set_one_hot(
            features,
            f"canvas_{position + 1:02d}_base",
            BASES,
            observed,
        )
    return features


def _stem_positions(pair_map: dict[int, int]) -> set[int]:
    positions: set[int] = set()
    for left, right in ((i, j) for i, j in pair_map.items() if i < j):
        if pair_map.get(left + 1) == right - 1 or pair_map.get(left - 1) == right + 1:
            positions.update((left, right))
    return positions


@lru_cache(maxsize=None)
def _cached_fold(sequence: str) -> FoldResult:
    return fold_sequence(sequence, with_pair_probs=False)


def _structure_membership(
    context: PositionContext,
) -> tuple[dict[str, set[int]], FoldResult | None]:
    try:
        fold = _cached_fold(context.sequence)
    except Exception:
        return {state: set() for state in STRUCTURE_STATES}, None
    pair_map = fold.pair_map
    paired = set(pair_map)
    stem = _stem_positions(pair_map)
    enclosed: set[int] = set()
    for left, right in ((i, j) for i, j in pair_map.items() if i < j):
        enclosed.update(range(left + 1, right))
    loop = {index for index in enclosed if index not in paired}
    unpaired = set(range(context.length)) - paired
    threshold = max(8, context.length // 2)
    long_range = {
        index
        for index, partner in pair_map.items()
        if abs(partner - index) >= threshold
    }
    return {
        "paired": paired,
        "stem": stem,
        "loop": loop,
        "unpaired": unpaired,
        "long_range_pair": long_range,
    }, fold


def _motif_membership(context: PositionContext) -> dict[str, set[int]]:
    membership: dict[str, set[int]] = {}
    for motif in MOTIFS:
        positions: set[int] = set()
        for start in _motif_positions(context.sequence, motif):
            positions.update(range(start, min(start + len(motif), context.length)))
        membership[motif] = positions
    return membership


def _ppb_membership(
    context: PositionContext,
    profile: dict[str, object],
    motif_membership: dict[str, set[int]],
) -> dict[str, set[int]]:
    linkages = context.outgoing_linkages
    ps_positions = {
        index for index, linkage in enumerate(linkages) if linkage == "PS"
    }
    gap_links = profile.get("gap_linkages", [])
    sensitive_ps: set[int] = set()
    sensitive_neutral: set[int] = set()
    early_mspa: set[int] = set()
    for item in gap_links if isinstance(gap_links, list) else []:
        index = int(item["absolute_position"])
        relative = int(item["gap_relative_position"])
        linkage = str(item["linkage"])
        if relative in {2, 3} and linkage == "PS":
            sensitive_ps.add(index)
        if relative in {2, 3} and linkage in {"MP", "MOP"}:
            sensitive_neutral.add(index)
        if relative in {1, 2, 3, 4, 5} and linkage == "MSPA":
            early_mspa.add(index)
    g_context = motif_membership["GG"] | motif_membership["GGG"]
    ps_g_context = {
        index
        for index in ps_positions
        if index in g_context or index + 1 in g_context
    }
    return {
        "ps": ps_positions,
        "sensitive_gap_ps": sensitive_ps,
        "sensitive_gap_neutral": sensitive_neutral,
        "early_gap_mspa": early_mspa,
        "ps_g_context": ps_g_context,
    }


def _region_indices(context: PositionContext) -> dict[str, list[int]]:
    return {
        region: [
            index
            for index, observed in enumerate(context.regions)
            if observed == region
        ]
        for region in REGIONS
    }


def _pool_categorical(
    features: dict[str, float],
    context: PositionContext,
    family: str,
    categories: tuple[str, ...],
    observed: tuple[str, ...],
    region_indices: dict[str, list[int]],
) -> None:
    for region in REGIONS:
        indices = region_indices[region]
        denominator = max(len(indices), 1)
        for category in categories:
            count = sum(
                1
                for index in indices
                if index < len(observed) and observed[index] == category
            )
            features[f"pool_{region}_{family}_{category}"] = float(
                count / denominator
            )


def _pool_membership(
    features: dict[str, float],
    family: str,
    categories: tuple[str, ...],
    membership: dict[str, set[int]],
    region_indices: dict[str, list[int]],
) -> None:
    for region in REGIONS:
        indices = region_indices[region]
        denominator = max(len(indices), 1)
        region_set = set(indices)
        for category in categories:
            features[f"pool_{region}_{family}_{category}"] = float(
                len(region_set & membership[category]) / denominator
            )


def _global_chemistry_features(
    helm: str,
    context: PositionContext,
    profile: dict[str, object],
    *,
    has_explicit_gap: bool,
) -> dict[str, float]:
    features: dict[str, float] = {}
    _set_one_hot(
        features,
        "global_length",
        ("le14", "15_16", "17_18", "19_20", "21_24", "ge25"),
        _length_bin(context.length),
    )
    gap_length = int(profile.get("gap_length", 0) or 0)
    _set_one_hot(
        features,
        "global_gap_length",
        ("zero", "1_5", "6_8", "9_10", "ge11"),
        _gap_length_bin(gap_length),
    )
    nonterminal = context.outgoing_linkages[:-1] if context.length else ()
    ps_fraction = (
        sum(linkage == "PS" for linkage in nonterminal) / max(len(nonterminal), 1)
    )
    _set_one_hot(
        features,
        "global_ps_fraction",
        ("zero", "le25", "le50", "lt100", "all"),
        _ps_fraction_bin(float(ps_fraction)),
    )
    gc_fraction = (
        (context.sequence.count("G") + context.sequence.count("C"))
        / max(context.length, 1)
    )
    _set_one_hot(
        features,
        "global_gc_fraction",
        ("le25", "le50", "le75", "gt75"),
        _quartile_bin(float(gc_fraction)),
    )
    transitions = sum(
        left != right
        for left, right in zip(
            context.modifications,
            context.modifications[1:],
        )
    )
    _set_one_hot(
        features,
        "global_mod_transitions",
        ("zero", "one", "two", "ge3"),
        _transition_bin(int(transitions)),
    )
    lower = str(helm).lower()
    for name, tokens in SYMBOLIC_SUGAR_TOKENS.items():
        features[f"global_symbolic_{name}"] = float(
            any(token.lower() in lower for token in tokens)
        )
    features["global_has_explicit_dna_gap"] = float(has_explicit_gap)
    features["global_canvas_truncated"] = float(context.length > MAX_BASE_CANVAS)
    return features


def _pair_sugar_features(
    context: PositionContext,
    fold: FoldResult | None,
) -> dict[str, float]:
    counts = {combo: 0 for combo in PAIR_MODIFICATION_COMBOS}
    pair_count = 0
    if fold is not None:
        order = {value: index for index, value in enumerate(MODIFICATIONS)}
        for left, right in (
            (left, right)
            for left, right in fold.pair_map.items()
            if left < right
        ):
            if left >= context.length or right >= context.length:
                continue
            observed = sorted(
                (context.modifications[left], context.modifications[right]),
                key=order.get,
            )
            combo = (observed[0], observed[1])
            counts[combo] += 1
            pair_count += 1
    return {
        f"pair_sugar_{left}_{right}_fraction": float(
            counts[(left, right)] / max(pair_count, 1)
        )
        for left, right in PAIR_MODIFICATION_COMBOS
    }


def _interaction_features(
    context: PositionContext,
    region_indices: dict[str, list[int]],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for region in REGIONS:
        indices = region_indices[region]
        denominator = max(len(indices), 1)
        for base in BASES:
            for sugar in MODIFICATIONS:
                count = sum(
                    context.sequence[index] == base
                    and context.modifications[index] == sugar
                    for index in indices
                )
                features[
                    f"interaction_{region}_base_{base}_sugar_{sugar}"
                ] = float(count / denominator)
    return features


@lru_cache(maxsize=None)
def _all_features_for_helm(helm: str) -> dict[str, float]:
    context = _context(str(helm))
    profile = gap_linkage_profile(str(helm))
    has_explicit_gap = (
        profile.get("gap_start") is not None
        and profile.get("gap_end") is not None
        and int(profile.get("gap_end") or 0) > int(profile.get("gap_start") or 0)
    )
    region_indices = _region_indices(context)
    motif_membership = _motif_membership(context)
    structure_membership, fold = _structure_membership(context)
    ppb_membership = _ppb_membership(context, profile, motif_membership)

    features = _base_canvas_features(context)
    features.update(_anchor_features(
        context,
        has_explicit_gap=has_explicit_gap,
    ))
    _pool_categorical(
        features,
        context,
        "base",
        BASES,
        tuple(context.sequence),
        region_indices,
    )
    _pool_categorical(
        features,
        context,
        "sugar",
        MODIFICATIONS,
        context.modifications,
        region_indices,
    )
    _pool_categorical(
        features,
        context,
        "link",
        LINKAGES,
        context.outgoing_linkages,
        region_indices,
    )
    features.update(
        _global_chemistry_features(
            str(helm),
            context,
            profile,
            has_explicit_gap=has_explicit_gap,
        )
    )
    _pool_membership(
        features,
        "motif",
        MOTIFS,
        motif_membership,
        region_indices,
    )
    _pool_membership(
        features,
        "structure",
        STRUCTURE_STATES,
        structure_membership,
        region_indices,
    )
    _pool_membership(
        features,
        "ppb",
        PPB_LOCAL_STATES,
        ppb_membership,
        region_indices,
    )
    pair_fraction = (
        len(fold.pair_map) / max(context.length, 1) if fold is not None else 0.0
    )
    _set_one_hot(
        features,
        "global_pair_fraction",
        ("zero", "le25", "le50", "gt50"),
        _pair_fraction_bin(float(pair_fraction)),
    )
    for motif in MOTIFS:
        features[f"global_motif_present_{motif}"] = float(
            bool(motif_membership[motif])
        )
    features["global_structure_ok"] = float(fold is not None)
    features.update(_pair_sugar_features(context, fold))
    features.update(_interaction_features(context, region_indices))
    return features


def features_for_helm(helm: str, mode: SemanticMode) -> dict[str, float]:
    """Return one fixed-schema semantic feature block for a HELM molecule."""

    all_features = _all_features_for_helm(str(helm))
    return {
        name: float(all_features.get(name, 0.0))
        for name in semantic_feature_names(mode)
    }


def dataframes_for_helms(
    helms: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Build every semantic representation with a shared row contract."""

    helm_values = [str(helm) for helm in helms]
    unique_rows: dict[str, dict[str, float]] = {}
    for helm in dict.fromkeys(helm_values):
        try:
            unique_rows[helm] = _all_features_for_helm(helm)
        except Exception:
            unique_rows[helm] = {}
    result: dict[str, pd.DataFrame] = {}
    for mode in SEMANTIC_MODES:
        names = list(semantic_feature_names(mode))
        frame = pd.DataFrame(
            [
                {name: unique_rows[helm].get(name, 0.0) for name in names}
                for helm in helm_values
            ],
            columns=names,
            dtype=np.float32,
        )
        result[mode] = frame
    return result


def active_feature_table(
    helm: str,
    *,
    mode: SemanticMode = "semantic_mechanism_ohe",
) -> pd.DataFrame:
    """Return non-zero named channels for one auditable explanation."""

    values = features_for_helm(str(helm), mode)
    rows = [
        {"feature": name, "value": value}
        for name, value in values.items()
        if not np.isclose(value, 0.0)
    ]
    return pd.DataFrame(rows).sort_values(
        ["value", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)
