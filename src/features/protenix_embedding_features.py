"""Utilities for turning cached Protenix representations into tabular features.

The repository currently has pilot Protenix `.npz` artifacts with:
  - single: per-token representation, usually [L, 384]
  - pair: pair representation, usually [L, L, 128]
  - sequence: base sequence used for extraction

This module deliberately does not run Protenix. It only consumes cached
artifacts so training scripts can report honest coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


N_SINGLE = 384
PAIR_SUMMARY_NAMES = (
    "pair_mean",
    "pair_std",
    "pair_abs_mean",
    "pair_offdiag_mean",
    "pair_offdiag_abs_mean",
)


def normalize_sequence(seq: str) -> str:
    """Normalize DNA/RNA sequence for exact cache matching."""
    return (seq or "").upper().replace("U", "T")


def feature_names() -> list[str]:
    names = [f"protenix_single_mean_{i}" for i in range(N_SINGLE)]
    names += [f"protenix_single_std_{i}" for i in range(N_SINGLE)]
    names += list(PAIR_SUMMARY_NAMES)
    names += ["protenix_missing"]
    return names


def _squeeze_single(single: np.ndarray) -> np.ndarray:
    arr = np.asarray(single, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.reshape((-1, arr.shape[-1]))
    if arr.ndim != 2:
        raise ValueError(f"single representation must be 2D/3D, got {arr.shape}")
    if arr.shape[1] != N_SINGLE:
        raise ValueError(f"single representation dim must be {N_SINGLE}, got {arr.shape}")
    return arr


def _squeeze_pair(pair: np.ndarray) -> np.ndarray:
    arr = np.asarray(pair, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    if arr.ndim == 3:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"pair representation must be 3D/4D, got {arr.shape}")
    return arr


def pool_npz(path: Path) -> tuple[str, np.ndarray]:
    """Return `(sequence, pooled_features)` from one Protenix `.npz` artifact."""
    data = np.load(path, allow_pickle=True)
    if "note" in data and "fail" in str(data["note"]).lower():
        raise ValueError(f"{path} contains failed placeholder features")
    seq = normalize_sequence(str(data["sequence"]))
    single = _squeeze_single(data["single"])
    if "pair" in data:
        pair = _squeeze_pair(data["pair"])
        pair_flat = pair.reshape(-1)
        if pair.shape[0] == pair.shape[1]:
            mask = ~np.eye(pair.shape[0], dtype=bool)
            offdiag = pair[mask]
        else:
            offdiag = pair_flat
        pair_summary = np.array(
            [
                float(pair_flat.mean()),
                float(pair_flat.std()),
                float(np.abs(pair_flat).mean()),
                float(offdiag.mean()) if offdiag.size else 0.0,
                float(np.abs(offdiag).mean()) if offdiag.size else 0.0,
            ],
            dtype=np.float32,
        )
    elif "pair_summary" in data:
        pair_summary = np.asarray(data["pair_summary"], dtype=np.float32).reshape(-1)
        if pair_summary.shape[0] != len(PAIR_SUMMARY_NAMES):
            raise ValueError(f"pair_summary dim must be {len(PAIR_SUMMARY_NAMES)}, got {pair_summary.shape}")
    else:
        raise ValueError(f"{path} contains neither pair nor pair_summary")

    single_mean = single.mean(axis=0)
    single_std = single.std(axis=0)
    vec = np.concatenate([single_mean, single_std, pair_summary, np.array([0.0], dtype=np.float32)])
    return seq, vec.astype(np.float32)


def load_feature_cache(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    """Load all usable cached Protenix artifacts keyed by normalized sequence."""
    cache: dict[str, np.ndarray] = {}
    for path in paths:
        try:
            seq, vec = pool_npz(Path(path))
        except Exception:
            continue
        cache[seq] = vec
    return cache


def discover_feature_cache(root: Path, extra_dirs: Iterable[Path] = ()) -> dict[str, np.ndarray]:
    """Discover cached Protenix `.npz` files from standard output locations."""
    dirs = [root / "outputs", root / "outputs" / "protenix_features_validation", *extra_dirs]
    files: list[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*protenix*.npz")))
            files.extend(sorted(d.glob("*.npz")))
    return load_feature_cache(files)


def matrix_for_sequences(sequences: Iterable[str], cache: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a feature matrix and coverage metadata for input sequences."""
    names = feature_names()
    missing_vec = np.zeros(len(names), dtype=np.float32)
    missing_vec[-1] = 1.0

    rows = []
    meta = []
    for seq in sequences:
        norm = normalize_sequence(seq)
        vec = cache.get(norm)
        hit = vec is not None
        rows.append(vec if hit else missing_vec.copy())
        meta.append({"sequence": norm, "protenix_cache_hit": int(hit)})
    return np.vstack(rows).astype(np.float32), pd.DataFrame(meta)


def dataframe_for_sequences(sequences: Iterable[str], cache: dict[str, np.ndarray]) -> pd.DataFrame:
    X, meta = matrix_for_sequences(sequences, cache)
    df = pd.DataFrame(X, columns=feature_names())
    return pd.concat([meta, df], axis=1)
