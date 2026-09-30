"""Normalized experimental-context features for oligonucleotide models.

The module keeps three concepts separate:

* biological/chemical molecule features;
* experimental covariates that may be used by a conditional model; and
* provenance/group identifiers, which must never become predictor features.

All parsers are conservative. Values without a recoverable unit remain missing,
and ambiguous source joins are expected to be resolved before this module is
called rather than guessed here.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Mapping

import numpy as np
import pandas as pd


NUMERIC_CONTEXT_COLUMNS = (
    "dose_um",
    "exposure_hours",
    "cell_density_per_well",
)

CATEGORICAL_CONTEXT_COLUMNS = (
    "cell_line",
    "species",
    "delivery",
    "route",
    "assay",
)

# These values occur in the ASOptimizer cell column but describe a readout or
# normalizer rather than a biological cell system.
KNOWN_NON_CELL_LABELS = frozenset(
    {
        "tubulin",
        "angptl2 actin",
        "oscillation",
    }
)

_MISSING_TEXT = frozenset(
    {
        "",
        "na",
        "n a",
        "nan",
        "none",
        "null",
        "undefined",
        "not available",
        "not reported",
    }
)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _MISSING_TEXT


def normalize_category(value: object) -> str:
    """Return a stable lowercase category, or an empty string when missing."""

    if _is_missing(value):
        return ""
    text = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
    text = re.sub(r"[^a-z0-9+.-]+", " ", text)
    return " ".join(text.split())


def clean_cell_line(value: object) -> tuple[str, str]:
    """Return a cleaned cell category and an explicit quality status."""

    normalized = normalize_category(value)
    if not normalized:
        return "", "missing"
    if normalized in KNOWN_NON_CELL_LABELS:
        return "", "non_cell_label"
    aliases = {
        "a 431": "a431",
        "a-431": "a431",
        "hela": "hela",
        "lipofectamine2000": "lipofectamine 2000",
    }
    return aliases.get(normalized, normalized), "recorded"


def _first_or_range_midpoint(text: str) -> float | None:
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    if len(numbers) >= 2 and re.search(r"\d\s*(?:-|to)\s*\d", text):
        return float((numbers[0] + numbers[1]) / 2.0)
    return float(numbers[0])


def parse_concentration_um(value: object, *, unit_hint: str | None = None) -> float:
    """Parse a concentration and normalize it to micromolar.

    Numeric values require ``unit_hint``. This prevents an undocumented number
    from silently being treated as nM or uM.
    """

    if _is_missing(value):
        return float("nan")
    hint = normalize_category(unit_hint)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        unit_text = hint
    else:
        text = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
        parsed = _first_or_range_midpoint(text)
        if parsed is None:
            return float("nan")
        number = parsed
        unit_text = text if re.search(r"(?:pm|nm|um|mm|mol)", text) else hint
    if not math.isfinite(number) or number < 0:
        return float("nan")
    compact = unit_text.replace(" ", "")
    if "pm" in compact:
        return number / 1_000_000.0
    if "nm" in compact:
        return number / 1_000.0
    if "um" in compact:
        return number
    if "mm" in compact:
        return number * 1_000.0
    return float("nan")


def parse_duration_hours(value: object, *, unit_hint: str | None = None) -> float:
    """Parse an exposure or observation duration and normalize it to hours."""

    if _is_missing(value):
        return float("nan")
    hint = normalize_category(unit_hint)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        unit_text = hint
    else:
        text = str(value).strip().lower()
        parsed = _first_or_range_midpoint(text)
        if parsed is None:
            return float("nan")
        number = parsed
        unit_text = text if re.search(r"(?:min|hour|hr|day|week)", text) else hint
    if not math.isfinite(number) or number < 0:
        return float("nan")
    if "week" in unit_text:
        return number * 24.0 * 7.0
    if "day" in unit_text:
        return number * 24.0
    if "min" in unit_text:
        return number / 60.0
    if "hour" in unit_text or "hr" in unit_text or unit_text in {"h", "hours"}:
        return number
    return float("nan")


def parse_cell_density(value: object) -> float:
    """Parse cells/well, using a range midpoint when the source reports one."""

    if _is_missing(value):
        return float("nan")
    parsed = _first_or_range_midpoint(str(value).replace(",", "").lower())
    if parsed is None or not math.isfinite(parsed) or parsed < 0:
        return float("nan")
    return float(parsed)


def normalize_context_record(record: Mapping[str, object]) -> dict[str, object]:
    """Normalize one record that already follows the canonical field names."""

    cell_line, cell_status = clean_cell_line(record.get("cell_line"))
    dose_unit = record.get("dose_unit")
    exposure_unit = record.get("exposure_time_unit")
    normalized: dict[str, object] = {
        "dose_um": parse_concentration_um(
            record.get("dose"),
            unit_hint="" if _is_missing(dose_unit) else str(dose_unit),
        ),
        "exposure_hours": parse_duration_hours(
            record.get("exposure_time"),
            unit_hint="" if _is_missing(exposure_unit) else str(exposure_unit),
        ),
        "cell_density_per_well": parse_cell_density(record.get("cell_density")),
        "cell_line": cell_line,
        "cell_line_status": cell_status,
        "species": normalize_category(record.get("species")),
        "delivery": normalize_category(record.get("delivery")),
        "route": normalize_category(record.get("route")),
        "assay": normalize_category(record.get("assay")),
        "source_id": normalize_category(record.get("source_id")),
        "target_group": normalize_category(record.get("target_group")),
    }
    available = 0
    for name in NUMERIC_CONTEXT_COLUMNS:
        is_available = bool(pd.notna(normalized[name]))
        normalized[f"{name}_available"] = int(is_available)
        available += int(is_available)
    for name in CATEGORICAL_CONTEXT_COLUMNS:
        is_available = bool(normalized[name])
        normalized[f"{name}_available"] = int(is_available)
        available += int(is_available)
    normalized["context_fields_available"] = available
    return normalized


def context_feature_dict(
    normalized: Mapping[str, object],
    *,
    include_numeric: bool = True,
    categorical_columns: tuple[str, ...] = CATEGORICAL_CONTEXT_COLUMNS,
) -> dict[str, float]:
    """Build a sparse, named feature dictionary from normalized context."""

    features: dict[str, float] = {}
    if include_numeric:
        for name in NUMERIC_CONTEXT_COLUMNS:
            value = normalized.get(name)
            available = bool(pd.notna(value))
            features[f"ctx__{name}__available"] = float(available)
            features[f"ctx__{name}__log1p"] = (
                float(np.log1p(max(float(value), 0.0))) if available else 0.0
            )
    for name in categorical_columns:
        if name not in CATEGORICAL_CONTEXT_COLUMNS:
            raise ValueError(f"unsupported context category: {name}")
        value = normalize_category(normalized.get(name))
        features[f"ctx__{name}__available"] = float(bool(value))
        if value:
            features[f"ctx__{name}={value}"] = 1.0
    features["ctx__available_field_count"] = float(
        normalized.get("context_fields_available", 0) or 0
    )
    features["ctx__cell_non_cell_label"] = float(
        normalized.get("cell_line_status") == "non_cell_label"
    )
    return features


def experimental_context_id(normalized: Mapping[str, object]) -> str:
    """Hash model-eligible context fields; source and target IDs are excluded."""

    values = []
    for name in (*NUMERIC_CONTEXT_COLUMNS, *CATEGORICAL_CONTEXT_COLUMNS):
        value = normalized.get(name)
        values.append("" if _is_missing(value) else str(value))
    values.append(str(normalized.get("cell_line_status") or ""))
    payload = "\0".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
