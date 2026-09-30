"""Feature loaders for Protenix custom-CCD oligo outputs.

These caches are keyed by HELM, not plain sequence, because modified-residue
CCD inputs encode backbone/sugar/base chemistry and produce atom-level tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.unimol_3d_embedding_features import helm_key


COMPACT_DIM = 128
CONTACT_THRESHOLDS = (4.0, 6.0, 8.0, 10.0)
ELEMENTS = ("C", "N", "O", "P", "S", "H")


def _string_scalar(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def load_compact_embedding_cache(cache_dir: Path) -> dict[str, np.ndarray]:
    """Load `compact_128` vectors keyed by HELM hash."""
    cache: dict[str, np.ndarray] = {}
    emb_dir = Path(cache_dir) / "embeddings"
    if not emb_dir.exists():
        return cache
    for path in sorted(emb_dir.glob("protenix_customccd_*.npz")):
        try:
            data = np.load(path, allow_pickle=True)
            key = _string_scalar(data["helm_key"]) if "helm_key" in data else helm_key(_string_scalar(data["helm"]))
            vec = np.asarray(data["compact_128"], dtype=np.float32).reshape(-1)
            if vec.shape[0] == COMPACT_DIM:
                cache[key] = vec
        except Exception:
            continue
    return cache


def compact_feature_names(dim: int = COMPACT_DIM) -> list[str]:
    return [f"protenix_customccd_compact_{i}" for i in range(dim)] + ["protenix_customccd_missing"]


def matrix_for_helms(helms: Iterable[str], cache: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    """Return compact custom-CCD embedding matrix and coverage metadata."""
    rows: list[np.ndarray] = []
    meta: list[dict[str, object]] = []
    for helm in helms:
        key = helm_key(str(helm))
        vec = cache.get(key)
        hit = vec is not None
        if hit:
            row = np.concatenate([vec.astype(np.float32), np.array([0.0], dtype=np.float32)])
        else:
            row = np.zeros(COMPACT_DIM + 1, dtype=np.float32)
            row[-1] = 1.0
        rows.append(row)
        meta.append({"helm_key": key, "protenix_customccd_cache_hit": int(hit)})
    return np.vstack(rows).astype(np.float32), pd.DataFrame(meta)


def _atom_site_rows(cif_path: Path) -> tuple[list[str], np.ndarray]:
    """Parse element symbols and coordinates from a Protenix prediction CIF."""
    elements: list[str] = []
    coords: list[list[float]] = []
    for line in cif_path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM "):
            continue
        parts = line.split()
        if len(parts) < 17:
            continue
        try:
            elements.append(parts[1])
            coords.append([float(parts[14]), float(parts[15]), float(parts[16])])
        except Exception:
            continue
    return elements, np.asarray(coords, dtype=np.float32)


def _empty_descriptor(prefix: str = "pccd3d", status: str = "missing") -> dict[str, float | str]:
    record: dict[str, float | str] = {
        f"{prefix}_status": status,
        f"{prefix}_n_atoms": 0.0,
        f"{prefix}_radius_gyration": 0.0,
        f"{prefix}_max_radius": 0.0,
        f"{prefix}_bbox_x": 0.0,
        f"{prefix}_bbox_y": 0.0,
        f"{prefix}_bbox_z": 0.0,
        f"{prefix}_pca_lambda1": 0.0,
        f"{prefix}_pca_lambda2": 0.0,
        f"{prefix}_pca_lambda3": 0.0,
        f"{prefix}_pca_linearity": 0.0,
        f"{prefix}_pca_planarity": 0.0,
        f"{prefix}_distance_mean": 0.0,
        f"{prefix}_distance_std": 0.0,
        f"{prefix}_distance_q10": 0.0,
        f"{prefix}_distance_q25": 0.0,
        f"{prefix}_distance_q50": 0.0,
        f"{prefix}_distance_q75": 0.0,
        f"{prefix}_distance_q90": 0.0,
        f"{prefix}_distance_max": 0.0,
    }
    for element in ELEMENTS:
        record[f"{prefix}_element_frac_{element}"] = 0.0
    for threshold in CONTACT_THRESHOLDS:
        tag = str(threshold).replace(".", "p")
        record[f"{prefix}_contact_count_{tag}A"] = 0.0
        record[f"{prefix}_contact_density_{tag}A"] = 0.0
    return record


def descriptors_from_cif(cif_path: Path, prefix: str = "pccd3d") -> dict[str, float | str]:
    """Compute graph-free 3D shape/contact descriptors from a saved CIF."""
    try:
        elements, coords = _atom_site_rows(cif_path)
        if coords.shape[0] < 2:
            return _empty_descriptor(prefix=prefix, status="failed_empty_cif")
        record = _empty_descriptor(prefix=prefix, status="ok")
        n_atoms = float(coords.shape[0])
        centered = coords - coords.mean(axis=0, keepdims=True)
        radii = np.sqrt((centered * centered).sum(axis=1))
        bbox = coords.max(axis=0) - coords.min(axis=0)
        cov = np.cov(centered.T)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eig = np.maximum(eig, 0.0)
        denom = max(float(eig[0]), 1e-8)

        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))
        tri = np.triu_indices(coords.shape[0], k=1)
        dvals = dist[tri]
        ddenom = max(float(len(dvals)), 1.0)

        record.update(
            {
                f"{prefix}_n_atoms": n_atoms,
                f"{prefix}_radius_gyration": float(np.sqrt((radii * radii).mean())),
                f"{prefix}_max_radius": float(radii.max()),
                f"{prefix}_bbox_x": float(bbox[0]),
                f"{prefix}_bbox_y": float(bbox[1]),
                f"{prefix}_bbox_z": float(bbox[2]),
                f"{prefix}_pca_lambda1": float(eig[0]),
                f"{prefix}_pca_lambda2": float(eig[1]) if len(eig) > 1 else 0.0,
                f"{prefix}_pca_lambda3": float(eig[2]) if len(eig) > 2 else 0.0,
                f"{prefix}_pca_linearity": float((eig[0] - eig[1]) / denom) if len(eig) > 1 else 0.0,
                f"{prefix}_pca_planarity": float((eig[1] - eig[2]) / denom) if len(eig) > 2 else 0.0,
                f"{prefix}_distance_mean": float(dvals.mean()),
                f"{prefix}_distance_std": float(dvals.std()),
                f"{prefix}_distance_q10": float(np.quantile(dvals, 0.10)),
                f"{prefix}_distance_q25": float(np.quantile(dvals, 0.25)),
                f"{prefix}_distance_q50": float(np.quantile(dvals, 0.50)),
                f"{prefix}_distance_q75": float(np.quantile(dvals, 0.75)),
                f"{prefix}_distance_q90": float(np.quantile(dvals, 0.90)),
                f"{prefix}_distance_max": float(dvals.max()),
            }
        )
        counts = {element: elements.count(element) for element in ELEMENTS}
        for element, count in counts.items():
            record[f"{prefix}_element_frac_{element}"] = float(count / n_atoms)
        for threshold in CONTACT_THRESHOLDS:
            tag = str(threshold).replace(".", "p")
            count = float((dvals <= threshold).sum())
            record[f"{prefix}_contact_count_{tag}A"] = count
            record[f"{prefix}_contact_density_{tag}A"] = count / ddenom
        return record
    except Exception as exc:
        record = _empty_descriptor(prefix=prefix, status="failed")
        record[f"{prefix}_error"] = f"{type(exc).__name__}: {exc}"
        return record


def load_cif_descriptor_cache(cache_dir: Path, prefix: str = "pccd3d") -> pd.DataFrame:
    """Compute descriptors for all saved custom-CCD Protenix CIFs."""
    rows: list[dict[str, object]] = []
    emb_dir = Path(cache_dir) / "embeddings"
    if not emb_dir.exists():
        return pd.DataFrame(columns=["helm_key"])
    for npz_path in sorted(emb_dir.glob("protenix_customccd_*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)
            key = _string_scalar(data["helm_key"])
            cif = Path(_string_scalar(data["structure_cif"]))
            row: dict[str, object] = {"helm_key": key, "structure_cif": str(cif)}
            row.update(descriptors_from_cif(cif, prefix=prefix) if cif.exists() else _empty_descriptor(prefix=prefix))
            rows.append(row)
        except Exception as exc:
            rows.append({"helm_key": npz_path.stem.replace("protenix_customccd_", ""), f"{prefix}_status": "failed", f"{prefix}_error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows).drop_duplicates("helm_key", keep="last")


def descriptor_matrix_for_helms(
    helms: Iterable[str],
    descriptor_cache: pd.DataFrame,
    prefix: str = "pccd3d",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = pd.DataFrame({"helm_key": [helm_key(str(h)) for h in helms]})
    merged = keys.merge(descriptor_cache, on="helm_key", how="left")
    status_col = f"{prefix}_status"
    meta = pd.DataFrame(
        {
            "helm_key": keys["helm_key"],
            "protenix_customccd_cif_hit": merged[status_col].notna().astype(int) if status_col in merged else 0,
            "protenix_customccd_cif_ok": (merged[status_col] == "ok").astype(int) if status_col in merged else 0,
        }
    )
    numeric = merged.select_dtypes(include=[np.number]).fillna(0.0)
    numeric = numeric.reset_index(drop=True)
    numeric["protenix_customccd_cif_missing"] = 1.0 - meta["protenix_customccd_cif_hit"].to_numpy(dtype=float)
    return numeric.astype(float), meta
