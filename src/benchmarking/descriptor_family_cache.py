"""Validated on-disk cache for assembled descriptor-family matrices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_family_cache(
    path: Path,
    x_values: pd.Series,
    families: dict[str, pd.DataFrame],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": x_values.astype(str).reset_index(drop=True).tolist(),
        "families": {name: frame.reset_index(drop=True) for name, frame in families.items()},
    }
    pd.to_pickle(payload, path)


def load_family_cache(
    path: Path,
    x_values: pd.Series,
) -> dict[str, pd.DataFrame] | None:
    if not path.exists():
        return None
    payload = pd.read_pickle(path)
    expected = x_values.astype(str).reset_index(drop=True).tolist()
    if not isinstance(payload, dict) or payload.get("x") != expected:
        raise ValueError(f"descriptor family cache row contract mismatch: {path}")
    families = payload.get("families")
    if not isinstance(families, dict) or "core_baseline" not in families:
        raise ValueError(f"invalid descriptor family cache: {path}")
    return {name: frame.reset_index(drop=True) for name, frame in families.items()}

