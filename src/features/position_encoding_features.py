"""Alternative positional encodings for variable-length oligonucleotides.

The baseline featurizer mostly uses 5'-anchored absolute positions. This block
adds complementary encodings so models can compare whether toxicity is better
explained by terminal, center-relative, or gap-relative location.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.features.chemistry_features import gap_linkage_profile
from src.features.featurizer import parse_helm
from src.features.sequence_chemistry_interaction_features import MOTIFS, _motif_positions


MOD_CODES = ("L", "M", "O", "D", "R")


def _normalize_mods(mod_pattern: str, n: int) -> str:
    out = (mod_pattern or "")[:n]
    return out + ("?" * max(0, n - len(out)))


def _region(idx: int, n: int, gap_start: int | None, gap_end: int | None) -> str:
    if gap_start is not None and gap_end is not None and gap_start <= idx < gap_end:
        if idx < gap_start + 3:
            return "gap_first3"
        if idx >= gap_end - 3:
            return "gap_last3"
        return "gap_mid"
    if gap_start is not None and idx < gap_start:
        return "wing5"
    if gap_end is not None and idx >= gap_end:
        return "wing3"
    return "other"


def _summary(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_abs_mean": 0.0,
            f"{prefix}_count": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_abs_mean": float(np.abs(arr).mean()),
        f"{prefix}_count": float(len(arr)),
    }


def features_for_helm(
    helm: str,
    max_pos: int = 30,
    max_center_offset: int = 15,
) -> dict[str, float]:
    """Compute position encodings anchored at 5', 3', center, and DNA gap."""
    mod_pattern, bases = parse_helm(str(helm))
    seq = bases.upper().replace("U", "T")
    n = len(seq)
    mods = _normalize_mods(mod_pattern, n)
    gap = gap_linkage_profile(str(helm))
    gap_start = gap.get("gap_start")
    gap_end = gap.get("gap_end")
    if gap_start is None or gap_end is None or gap_end <= gap_start:
        wing = min(5, max(n // 4, 1))
        gap_start, gap_end = wing, max(wing, n - wing)
    gap_center = (float(gap_start) + float(gap_end - 1)) / 2.0 if n else 0.0
    seq_center = (n - 1) / 2.0 if n else 0.0
    right_center_anchor = n // 2
    denom_abs = max(n - 1, 1)
    denom_center = max(seq_center, 1.0)
    gap_len = max(gap_end - gap_start, 1)

    feats: dict[str, float] = {
        "posenc_seq_len": float(n),
        "posenc_gap_start_5p_norm": float(gap_start / max(n, 1)),
        "posenc_gap_end_5p_norm": float(gap_end / max(n, 1)),
        "posenc_gap_center_5p_norm": float(gap_center / max(n - 1, 1)),
        "posenc_center_anchor_is_right_middle": float(n % 2 == 0),
    }

    # 5'-anchored and 3'-mirrored fixed slots. These expose terminal asymmetry.
    for i in range(max_pos):
        mod5 = mods[i] if i < n else "?"
        base5 = seq[i] if i < n else "?"
        j = n - 1 - i
        mod3 = mods[j] if 0 <= j < n else "?"
        base3 = seq[j] if 0 <= j < n else "?"
        feats[f"posenc_5p_{i+1:02d}_exists"] = float(i < n)
        feats[f"posenc_3p_{i+1:02d}_exists"] = float(0 <= j < n)
        feats[f"posenc_5p_{i+1:02d}_modified"] = float(mod5 in "LMO")
        feats[f"posenc_3p_{i+1:02d}_modified"] = float(mod3 in "LMO")
        feats[f"posenc_5p_{i+1:02d}_in_gap"] = float(i < n and gap_start <= i < gap_end)
        feats[f"posenc_3p_{i+1:02d}_in_gap"] = float(0 <= j < n and gap_start <= j < gap_end)
        for code in MOD_CODES:
            feats[f"posenc_5p_{i+1:02d}_{code}"] = float(mod5 == code)
            feats[f"posenc_3p_{i+1:02d}_{code}"] = float(mod3 == code)
        for base in ("A", "C", "G", "T"):
            feats[f"posenc_5p_{i+1:02d}_base_{base}"] = float(base5 == base)
            feats[f"posenc_3p_{i+1:02d}_base_{base}"] = float(base3 == base)

    # Center-relative fixed slots. For even lengths, offset -1/0 are the two
    # middle residues; for odd lengths, offset 0 is the exact middle residue.
    for offset in range(-max_center_offset, max_center_offset + 1):
        idx = right_center_anchor + offset
        tag = f"m{abs(offset):02d}" if offset < 0 else f"p{offset:02d}"
        mod = mods[idx] if 0 <= idx < n else "?"
        base = seq[idx] if 0 <= idx < n else "?"
        feats[f"posenc_center_{tag}_exists"] = float(0 <= idx < n)
        feats[f"posenc_center_{tag}_modified"] = float(mod in "LMO")
        feats[f"posenc_center_{tag}_in_gap"] = float(0 <= idx < n and gap_start <= idx < gap_end)
        feats[f"posenc_center_{tag}_center_norm"] = float((idx - seq_center) / denom_center) if 0 <= idx < n else 0.0
        for code in MOD_CODES:
            feats[f"posenc_center_{tag}_{code}"] = float(mod == code)
        for base_name in ("A", "C", "G", "T"):
            feats[f"posenc_center_{tag}_base_{base_name}"] = float(base == base_name)

    # Continuous summaries are length-normalized and therefore more comparable
    # across 13mer, 16mer, 19mer, and 20mer examples.
    for code in MOD_CODES:
        positions = [idx for idx, mod in enumerate(mods) if mod == code]
        feats.update(_summary([idx / denom_abs for idx in positions], f"posenc_{code}_5p_norm"))
        feats.update(_summary([(n - 1 - idx) / denom_abs for idx in positions], f"posenc_{code}_3p_norm"))
        feats.update(_summary([(idx - seq_center) / denom_center for idx in positions], f"posenc_{code}_center_norm"))
        feats.update(_summary([abs(idx - gap_start) / gap_len for idx in positions], f"posenc_{code}_dist_gap_start"))
        feats.update(_summary([abs(idx - (gap_end - 1)) / gap_len for idx in positions], f"posenc_{code}_dist_gap_end"))
        feats.update(_summary([abs(idx - gap_center) / gap_len for idx in positions], f"posenc_{code}_dist_gap_center"))
        for region_name in ("wing5", "gap_first3", "gap_mid", "gap_last3", "wing3", "other"):
            count = sum(1 for idx in positions if _region(idx, n, gap_start, gap_end) == region_name)
            feats[f"posenc_{code}_{region_name}_count"] = float(count)
            feats[f"posenc_{code}_{region_name}_fraction"] = float(count / max(len(positions), 1))

    modified_positions = [idx for idx, mod in enumerate(mods) if mod in "LMO"]
    feats.update(_summary([(idx - seq_center) / denom_center for idx in modified_positions], "posenc_modified_center_norm"))
    feats.update(_summary([abs(idx - gap_center) / gap_len for idx in modified_positions], "posenc_modified_dist_gap_center"))

    # Motif location summaries: not just whether a motif exists, but whether it
    # is terminal, central, or gap-centered after normalizing for length.
    for motif in MOTIFS:
        starts = _motif_positions(seq, motif)
        tag = motif.replace(" ", "_")
        centers = [start + (len(motif) - 1) / 2.0 for start in starts]
        feats.update(_summary([(c - seq_center) / denom_center for c in centers], f"posenc_motif_{tag}_center_norm"))
        feats.update(_summary([abs(c - gap_center) / gap_len for c in centers], f"posenc_motif_{tag}_dist_gap_center"))
        feats[f"posenc_motif_{tag}_in_gap_count"] = float(
            sum(gap_start <= start < gap_end for start in starts)
        )
        feats[f"posenc_motif_{tag}_near_center_count"] = float(
            sum(abs(c - seq_center) <= 2.0 for c in centers)
        )
        feats[f"posenc_motif_{tag}_near_5p_count"] = float(sum(start < 5 for start in starts))
        feats[f"posenc_motif_{tag}_near_3p_count"] = float(sum((n - (start + len(motif))) < 5 for start in starts))

    return feats


def dataframe_for_helms(
    helms: Iterable[str],
    max_pos: int = 30,
    max_center_offset: int = 15,
) -> pd.DataFrame:
    records = []
    for helm in helms:
        try:
            records.append(features_for_helm(str(helm), max_pos=max_pos, max_center_offset=max_center_offset))
        except Exception:
            records.append({})
    return pd.DataFrame(records).fillna(0.0).astype(float)

