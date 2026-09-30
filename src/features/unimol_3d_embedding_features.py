"""Cached Uni-Mol 3D embedding utilities for modified oligos.

This module consumes `.npz` files created from explicit RDKit conformers:
HELM -> stereo-aware SMILES/RDKit Mol -> 3D coordinates -> Uni-Mol cls_repr.
The cache is keyed by a stable SHA1 hash of the HELM string, because unlike
the current Protenix sequence cache, Uni-Mol 3D features depend on chemistry.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def helm_key(helm: str) -> str:
    return hashlib.sha1((helm or "").encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir: Path, helm: str) -> Path:
    return Path(cache_dir) / f"unimol3d_{helm_key(helm)}.npz"


def load_feature_cache(cache_dir: Path) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return cache
    for path in sorted(cache_dir.glob("unimol3d_*.npz")):
        try:
            data = np.load(path, allow_pickle=True)
            status = str(data.get("status", "ok"))
            if status != "ok":
                continue
            helm = str(data["helm"])
            emb = np.asarray(data["embedding"], dtype=np.float32).reshape(-1)
            cache[helm_key(helm)] = emb
        except Exception:
            continue
    return cache


def feature_names(dim: int) -> list[str]:
    return [f"unimol3d_{i}" for i in range(dim)] + ["unimol3d_missing"]


def matrix_for_helms(helms: Iterable[str], cache: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    helms = list(helms)
    dim = len(next(iter(cache.values()))) if cache else 0
    rows: list[np.ndarray] = []
    meta: list[dict] = []
    for helm in helms:
        key = helm_key(helm)
        vec = cache.get(key)
        hit = vec is not None
        if dim == 0:
            row = np.ones(1, dtype=np.float32)
        elif hit:
            row = np.concatenate([vec.astype(np.float32), np.array([0.0], dtype=np.float32)])
        else:
            row = np.zeros(dim + 1, dtype=np.float32)
            row[-1] = 1.0
        rows.append(row)
        meta.append({"helm_key": key, "unimol3d_cache_hit": int(hit)})
    return np.vstack(rows).astype(np.float32), pd.DataFrame(meta)


def dataframe_for_helms(helms: Iterable[str], cache: dict[str, np.ndarray]) -> pd.DataFrame:
    X, meta = matrix_for_helms(helms, cache)
    dim = X.shape[1] - 1
    return pd.concat([meta, pd.DataFrame(X, columns=feature_names(dim))], axis=1)
