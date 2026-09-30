"""Dataset-independent representation for modified oligonucleotides.

The schema is frozen in code. Known chemistry is factorized into named,
interpretable channels. Novel raw HELM tokens use deterministic hash fallback
channels, so inference never needs to fit a vocabulary on a dataset.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.chemistry_registry import (
    BASE_ONEHOT_VOCAB,
    SUGAR_ONEHOT_VOCAB,
    base_onehot_category,
    registry_manifest,
    sugar_onehot_category,
)

from src.features.featurizer import (
    HelmResidue,
    featurize_dataset,
    parse_helm_components,
    reconcile_helm_and_source_sequence,
    universal_core_feature_names,
)
from src.features.semantic_onehot_features import (
    features_for_helm as semantic_features_for_helm,
    semantic_feature_names,
)


SCHEMA_VERSION = "universal_oligo_representation_v3_registry"
MAX_POSITIONS = 30
UNKNOWN_HASH_BINS = 8
AUXILIARY_HASH_BINS = 16
UNKNOWN_COMPONENTS = ("base", "sugar", "linkage")
RAW_PHOSPHATE_VOCAB = ("sp", "p", "EMPTY", "OTHER")
RAW_SUGAR_VOCAB = SUGAR_ONEHOT_VOCAB
RAW_BASE_VOCAB = BASE_ONEHOT_VOCAB


def _stable_bin(value: str, bins: int) -> int:
    digest = hashlib.blake2b(
        str(value).encode("utf-8"),
        digest_size=8,
        person=b"oligo-v1",
    ).digest()
    return int.from_bytes(digest, "little") % bins


def _schema_hash(names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _raw_base_category(raw_base: str) -> str:
    return base_onehot_category(raw_base)


def _raw_sugar_category(raw_sugar: str) -> str:
    return sugar_onehot_category(raw_sugar)


def _raw_phosphate_category(residue: HelmResidue) -> str:
    if residue.linkage == "TERMINAL":
        return "EMPTY"
    token = str(residue.raw_linkage).strip().lower()
    if not token and residue.linkage == "PO":
        return "p"
    return token if token in RAW_PHOSPHATE_VOCAB[:-2] else "OTHER"


def _unknown_raw_token(residue: HelmResidue, component: str) -> str | None:
    if component == "base":
        return (
            residue.raw_base
            if _raw_base_category(residue.raw_base) == "OTHER"
            else None
        )
    if component == "sugar":
        return (
            residue.raw_sugar
            if _raw_sugar_category(residue.raw_sugar) == "OTHER"
            else None
        )
    if component == "linkage":
        return (
            residue.raw_linkage
            if _raw_phosphate_category(residue) == "OTHER"
            else None
        )
    raise ValueError(f"unsupported unknown component: {component}")


def _token_feature_names() -> list[str]:
    names: list[str] = []
    for position in range(1, MAX_POSITIONS + 1):
        prefix = f"raw_pos_{position:02d}"
        names.extend(
            f"{prefix}_phosphate_{value}" for value in RAW_PHOSPHATE_VOCAB
        )
        names.extend(
            f"{prefix}_sugar_{value}" for value in RAW_SUGAR_VOCAB
        )
        names.extend(f"{prefix}_base_{value}" for value in RAW_BASE_VOCAB)
    return names


def _global_feature_names() -> list[str]:
    names = [
        "global_overflow_count",
        "global_overflow_fraction",
    ]
    for component in UNKNOWN_COMPONENTS:
        names.append(f"global_unknown_{component}_count")
        names.extend(
            f"global_unknown_{component}_hash_fraction_{bucket:02d}"
            for bucket in range(UNKNOWN_HASH_BINS)
        )
    names.extend(
        [
            "global_auxiliary_component_count",
            "global_has_auxiliary_component",
            *(
                f"global_auxiliary_hash_{bucket:02d}"
                for bucket in range(AUXILIARY_HASH_BINS)
            ),
        ]
    )
    return names


def universal_ohe_feature_names() -> tuple[str, ...]:
    """Return the immutable flattened universal One-Hot schema."""

    names = [
        *_token_feature_names(),
        *_global_feature_names(),
    ]
    if len(names) != len(set(names)):
        raise RuntimeError("universal One-Hot schema contains duplicate names")
    return tuple(f"uohe__{name}" for name in names)


def pre_cet_ohe_feature_names() -> tuple[str, ...]:
    """Return the controlled pre-cEt registry schema used in ablations.

    This is the current immutable One-Hot schema with only the thirty
    position-specific cEt sugar channels removed. It isolates the value of
    registering cEt without changing any base, linkage, unknown, or overflow
    semantics.
    """

    names = tuple(
        name
        for name in universal_ohe_feature_names()
        if not name.endswith("_sugar_cet")
    )
    removed = len(universal_ohe_feature_names()) - len(names)
    if removed != MAX_POSITIONS:
        raise RuntimeError(
            f"expected {MAX_POSITIONS} cEt position channels, removed {removed}"
        )
    return names


def universal_core_names() -> tuple[str, ...]:
    return tuple(
        f"ucore__{name}"
        for name in universal_core_feature_names(max_pos=MAX_POSITIONS)
    )


def universal_context_names() -> tuple[str, ...]:
    """Chemistry context excluding the duplicated fixed base canvas."""

    return tuple(
        f"ucontext__{name}"
        for name in semantic_feature_names("semantic_chem_ohe")
        if not name.startswith("canvas_")
    )


def universal_enhanced_feature_names() -> tuple[str, ...]:
    return (
        *universal_ohe_feature_names(),
        *universal_core_names(),
        *universal_context_names(),
    )


def _auxiliary_components(helm: str) -> tuple[str, ...]:
    polymer_section, *sections = str(helm).split("$")
    polymers = [
        part.strip()
        for part in polymer_section.split("|")
        if part.strip() and not part.strip().upper().startswith("RNA")
    ]
    connections = sections[0].strip() if sections else ""
    if connections:
        polymers.append(f"CONNECTIONS:{connections}")
    return tuple(polymers)


def _source_complete_residues(
    helm: str,
    source_sequence: object,
) -> tuple[HelmResidue, ...]:
    """Add source-recorded terminal bases while keeping absent chemistry unknown."""

    residues = list(parse_helm_components(helm))
    reconciled = reconcile_helm_and_source_sequence(helm, source_sequence)
    if not reconciled.appended_source_bases:
        return tuple(residues)
    residues[-1] = replace(
        residues[-1],
        raw_linkage="UNKNOWN",
        linkage="OTHER",
    )
    for index, base in enumerate(reconciled.appended_source_bases):
        terminal = index == len(reconciled.appended_source_bases) - 1
        residues.append(
            HelmResidue(
                raw_sugar="UNKNOWN",
                raw_base=base,
                raw_linkage="" if terminal else "UNKNOWN",
                modification="X",
                base=base,
                linkage="TERMINAL" if terminal else "OTHER",
            )
        )
    return tuple(residues)


def _row_values(
    helm: str,
    source_sequence: object | None = None,
) -> tuple[dict[str, float], dict[str, object]]:
    residues = (
        parse_helm_components(helm)
        if source_sequence is None
        else _source_complete_residues(helm, source_sequence)
    )
    length = len(residues)
    features: dict[str, float] = {}

    for position in range(MAX_POSITIONS):
        residue = residues[position] if position < length else None
        phosphate = (
            _raw_phosphate_category(residue)
            if residue is not None
            else ""
        )
        sugar = (
            _raw_sugar_category(residue.raw_sugar)
            if residue is not None
            else ""
        )
        base = (
            _raw_base_category(residue.raw_base)
            if residue is not None
            else ""
        )
        prefix = f"raw_pos_{position + 1:02d}"
        for category in RAW_PHOSPHATE_VOCAB:
            features[
                f"{prefix}_phosphate_{category}"
            ] = float(phosphate == category)
        for category in RAW_SUGAR_VOCAB:
            features[f"{prefix}_sugar_{category}"] = float(sugar == category)
        for category in RAW_BASE_VOCAB:
            features[f"{prefix}_base_{category}"] = float(base == category)

    denominator = max(length, 1)
    features["global_overflow_count"] = float(max(length - MAX_POSITIONS, 0))
    features["global_overflow_fraction"] = float(
        max(length - MAX_POSITIONS, 0) / denominator
    )

    unknown_counts: dict[str, int] = {}
    for component in UNKNOWN_COMPONENTS:
        tokens = [
            token
            for residue in residues
            if (token := _unknown_raw_token(residue, component)) is not None
        ]
        unknown_counts[component] = len(tokens)
        features[f"global_unknown_{component}_count"] = float(len(tokens))
        hashed = Counter(
            _stable_bin(token, UNKNOWN_HASH_BINS) for token in tokens
        )
        for bucket in range(UNKNOWN_HASH_BINS):
            features[
                f"global_unknown_{component}_hash_fraction_{bucket:02d}"
            ] = float(hashed[bucket] / denominator)

    auxiliaries = _auxiliary_components(helm)
    features["global_auxiliary_component_count"] = float(len(auxiliaries))
    features["global_has_auxiliary_component"] = float(bool(auxiliaries))
    auxiliary_bins = Counter(
        _stable_bin(value, AUXILIARY_HASH_BINS) for value in auxiliaries
    )
    for bucket in range(AUXILIARY_HASH_BINS):
        features[f"global_auxiliary_hash_{bucket:02d}"] = float(
            auxiliary_bins[bucket]
        )

    names = [name.removeprefix("uohe__") for name in universal_ohe_feature_names()]
    complete = {name: float(features.get(name, 0.0)) for name in names}
    audit = {
        "length": length,
        "overflow_count": max(length - MAX_POSITIONS, 0),
        "unknown_base_count": unknown_counts["base"],
        "unknown_sugar_count": unknown_counts["sugar"],
        "unknown_linkage_count": unknown_counts["linkage"],
        "auxiliary_component_count": len(auxiliaries),
    }
    return complete, audit


def universal_ohe_for_helm(
    helm: str,
    source_sequence: object | None = None,
) -> pd.Series:
    values, _ = _row_values(str(helm), source_sequence)
    return pd.Series(
        [values[name.removeprefix("uohe__")] for name in universal_ohe_feature_names()],
        index=universal_ohe_feature_names(),
        dtype=float,
    )


def universal_ohe_dataframe(
    helms: Iterable[str],
    source_sequences: Iterable[object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    audits: list[dict[str, object]] = []
    helm_values = list(helms)
    source_values = (
        [None] * len(helm_values)
        if source_sequences is None
        else list(source_sequences)
    )
    if len(source_values) != len(helm_values):
        raise ValueError("source_sequences must have the same length as helms")
    for row_index, (helm, source_sequence) in enumerate(
        zip(helm_values, source_values)
    ):
        values, audit = _row_values(str(helm), source_sequence)
        rows.append(
            {
                f"uohe__{name}": value
                for name, value in values.items()
            }
        )
        audits.append({"row_index": row_index, **audit})
    frame = pd.DataFrame(rows).reindex(
        columns=universal_ohe_feature_names(),
        fill_value=0.0,
    )
    return frame.astype(np.float32), pd.DataFrame(audits)


def universal_representation_frames(
    frame: pd.DataFrame,
    *,
    helm_col: str = "x",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Build raw and enhanced fixed representations for one source frame."""

    core, parsed = featurize_dataset(
        frame,
        helm_col=helm_col,
        max_pos=MAX_POSITIONS,
        fixed_schema=True,
    )
    core.columns = list(universal_core_names())
    raw, audit = universal_ohe_dataframe(parsed[helm_col].astype(str))
    context_rows = [
        semantic_features_for_helm(helm, "semantic_chem_ohe")
        for helm in parsed[helm_col].astype(str)
    ]
    context_source_names = [
        name
        for name in semantic_feature_names("semantic_chem_ohe")
        if not name.startswith("canvas_")
    ]
    context = pd.DataFrame(context_rows).reindex(
        columns=context_source_names,
        fill_value=0.0,
    )
    context.columns = list(universal_context_names())
    enhanced = pd.concat(
        [
            raw.reset_index(drop=True),
            core.reset_index(drop=True).astype(np.float32),
            context.reset_index(drop=True).astype(np.float32),
        ],
        axis=1,
    )
    if tuple(raw.columns) != universal_ohe_feature_names():
        raise RuntimeError("raw universal One-Hot schema mismatch")
    if tuple(enhanced.columns) != universal_enhanced_feature_names():
        raise RuntimeError("enhanced universal schema mismatch")
    return {
        "universal_ohe_raw": raw,
        "universal_ohe_enhanced": enhanced,
    }, parsed, audit


def schema_manifest() -> dict[str, object]:
    raw = universal_ohe_feature_names()
    enhanced = universal_enhanced_feature_names()
    return {
        "schema_version": SCHEMA_VERSION,
        "max_positions": MAX_POSITIONS,
        "unknown_hash_bins": UNKNOWN_HASH_BINS,
        "auxiliary_hash_bins": AUXILIARY_HASH_BINS,
        "raw_dimension": len(raw),
        "enhanced_dimension": len(enhanced),
        "core_dimension": len(universal_core_names()),
        "context_dimension": len(universal_context_names()),
        "raw_schema_hash": _schema_hash(raw),
        "enhanced_schema_hash": _schema_hash(enhanced),
        "unknown_policy": (
            "named OTHER channel plus deterministic position/global hash; "
            "no dataset-fitted vocabulary"
        ),
        "length_policy": (
            "first 30 residues retain direct position channels; all residues "
            "contribute to global fractions/hashes and overflow is explicit"
        ),
        "chemistry_registry": registry_manifest(),
    }


def write_schema_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = schema_manifest()
    (directory / "universal_representation_schema.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "feature_index": np.arange(len(universal_ohe_feature_names())),
            "feature_name": universal_ohe_feature_names(),
            "representation": "universal_ohe_raw",
        }
    ).to_csv(directory / "universal_ohe_raw_feature_names.csv", index=False)
    pd.DataFrame(
        {
            "feature_index": np.arange(len(universal_enhanced_feature_names())),
            "feature_name": universal_enhanced_feature_names(),
            "representation": "universal_ohe_enhanced",
        }
    ).to_csv(
        directory / "universal_ohe_enhanced_feature_names.csv",
        index=False,
    )
