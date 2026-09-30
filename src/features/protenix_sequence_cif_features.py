"""Descriptors from sequence-only Protenix CIF structures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.protenix_customccd_features import descriptors_from_cif
from src.features.protenix_embedding_features import normalize_sequence


def _empty_descriptor(prefix: str = "pseq3d", status: str = "missing") -> dict[str, float | str]:
    probe = descriptors_from_cif(Path("__missing__.cif"), prefix=prefix)
    out = {k: (0.0 if isinstance(v, float) else v) for k, v in probe.items()}
    out[f"{prefix}_status"] = status
    return out


def _sequence_from_archive_dir(path: Path) -> str:
    parts = path.name.split("_")
    if len(parts) < 3:
        return ""
    seq = parts[-1].upper().replace("U", "T")
    return seq if re.fullmatch("[ACGT]+", seq) else ""


def load_sequence_cif_descriptor_cache(
    archive_dir: Path,
    extra_npz_dir: Path | None = None,
    prefix: str = "pseq3d",
) -> pd.DataFrame:
    """Load descriptors keyed by plain normalized sequence."""
    rows: list[dict[str, object]] = []
    archive_dir = Path(archive_dir)
    if archive_dir.exists():
        for item in sorted(archive_dir.glob("protenix_*")):
            cif = item / "structure.cif"
            seq = _sequence_from_archive_dir(item)
            if not seq or not cif.exists():
                continue
            row: dict[str, object] = {"sequence": seq, "structure_cif": str(cif)}
            row.update(descriptors_from_cif(cif, prefix=prefix))
            rows.append(row)

    if extra_npz_dir is not None and Path(extra_npz_dir).exists():
        for path in sorted(Path(extra_npz_dir).glob("*.npz")):
            try:
                data = np.load(path, allow_pickle=True)
                seq = normalize_sequence(str(np.asarray(data["sequence"]).item()))
                cif_text = str(np.asarray(data["structure_cif"]).item()) if "structure_cif" in data else ""
                if not cif_text:
                    continue
                cif = Path(cif_text)
                if not cif.is_file():
                    continue
                row = {"sequence": seq, "structure_cif": str(cif)}
                row.update(descriptors_from_cif(cif, prefix=prefix))
                rows.append(row)
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(columns=["sequence"])
    return pd.DataFrame(rows).drop_duplicates("sequence", keep="last")


def descriptor_matrix_for_sequences(
    sequences: Iterable[str],
    descriptor_cache: pd.DataFrame,
    prefix: str = "pseq3d",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = pd.DataFrame({"sequence": [normalize_sequence(str(s)) for s in sequences]})
    merged = keys.merge(descriptor_cache, on="sequence", how="left")
    status_col = f"{prefix}_status"
    meta = pd.DataFrame(
        {
            "sequence": keys["sequence"],
            "protenix_sequence_cif_hit": merged[status_col].notna().astype(int) if status_col in merged else 0,
            "protenix_sequence_cif_ok": (merged[status_col] == "ok").astype(int) if status_col in merged else 0,
        }
    )
    numeric = merged.select_dtypes(include=[np.number]).fillna(0.0).reset_index(drop=True)
    numeric["protenix_sequence_cif_missing"] = 1.0 - meta["protenix_sequence_cif_hit"].to_numpy(dtype=float)
    return numeric.astype(float), meta
