"""Position-aware PPB/protein-binding descriptors.

The existing PPB rule block summarizes global PS density, protein-attribution
scores, and site-specific backbone rules. This block keeps the same evidence
source but exposes where those signals occur: 5' end, 3' end, DNA gap, center,
and motif-near-PS contexts.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.chemistry_features import compute_all_features, gap_linkage_profile
from src.features.featurizer import parse_helm
from src.features.helm_to_smiles import helm_to_smiles
from src.features.sequence_chemistry_interaction_features import MOTIFS, _motif_positions


LINKAGE_CLASSES = ("PS", "PO", "MP", "MOP", "MSPA")
PROTEIN_KEYS = ("HSA", "aagp", "factorH", "lipoprotein")


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


def _safe_bundle(helm: str, sequence: str | None, extra_name: str) -> dict:
    try:
        smiles = helm_to_smiles(str(helm))
        mol = None
        if smiles:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
        return compute_all_features(str(helm), mol, sequence=sequence, extra_name=extra_name)
    except Exception:
        return {}


def _region_for_link(link_idx: int, n_links: int, gap_start: int | None, gap_end: int | None) -> str:
    if gap_start is not None and gap_end is not None and gap_start <= link_idx < max(gap_start, gap_end - 1):
        rel = link_idx - gap_start + 1
        if rel <= 3:
            return "gap5"
        if link_idx >= max(gap_start, gap_end - 4):
            return "gap3"
        return "gap_mid"
    if link_idx < 5:
        return "end5"
    if link_idx >= max(0, n_links - 5):
        return "end3"
    return "other"


def features_for_helm(
    helm: str,
    sequence: str | None = None,
    extra_name: str = "",
    max_link_pos: int = 30,
    max_gap_rel: int = 12,
) -> dict[str, float]:
    """Compute position-aware PPB/protein-binding descriptors for one HELM."""

    mod_pattern, bases = parse_helm(str(helm))
    seq = (sequence or bases).upper().replace("U", "T")
    n_res = len(seq)
    profile = gap_linkage_profile(str(helm))
    linkages = list(profile.get("linkages", []))
    n_links = len(linkages)
    counts = Counter(linkages)
    gap_start = profile.get("gap_start")
    gap_end = profile.get("gap_end")
    gap_center = 0.0
    if gap_start is not None and gap_end is not None and gap_end > gap_start:
        gap_center = (float(gap_start) + float(gap_end - 1)) / 2.0
    seq_center = (n_res - 1) / 2.0 if n_res else 0.0
    denom_link = max(n_links - 1, 1)
    denom_center = max(seq_center, 1.0)
    gap_len = max(int(profile.get("gap_length") or 0), 1)
    bundle = _safe_bundle(str(helm), seq, extra_name)
    attribution = bundle.get("protein_attribution", {}) if isinstance(bundle, dict) else {}
    site_scores = (bundle.get("site_specific_backbone", {}) or {}).get("scores", {}) if isinstance(bundle, dict) else {}

    feats: dict[str, float] = {
        "ppbpos_seq_len": float(n_res),
        "ppbpos_n_linkages": float(n_links),
        "ppbpos_gap_start_norm": float((gap_start or 0) / max(n_res, 1)),
        "ppbpos_gap_end_norm": float((gap_end or 0) / max(n_res, 1)),
        "ppbpos_gap_length": float(profile.get("gap_length", 0) or 0),
    }

    for cls in LINKAGE_CLASSES:
        positions = [idx for idx, link in enumerate(linkages) if link == cls]
        feats[f"ppbpos_{cls}_count"] = float(counts.get(cls, 0))
        feats[f"ppbpos_{cls}_fraction"] = float(counts.get(cls, 0) / max(n_links, 1))
        feats.update(_summary([idx / denom_link for idx in positions], f"ppbpos_{cls}_5p_norm"))
        feats.update(_summary([(n_links - 1 - idx) / denom_link for idx in positions], f"ppbpos_{cls}_3p_norm"))
        feats.update(_summary([(idx - seq_center) / denom_center for idx in positions], f"ppbpos_{cls}_center_norm"))
        feats.update(_summary([abs(idx - gap_center) / gap_len for idx in positions], f"ppbpos_{cls}_dist_gap_center"))
        for region in ("end5", "end3", "gap5", "gap_mid", "gap3", "other"):
            region_count = sum(1 for idx in positions if _region_for_link(idx, n_links, gap_start, gap_end) == region)
            feats[f"ppbpos_{cls}_{region}_count"] = float(region_count)
            feats[f"ppbpos_{cls}_{region}_fraction"] = float(region_count / max(len(positions), 1))

    by_gap_rel = {int(x["gap_relative_position"]): str(x["linkage"]) for x in profile.get("gap_linkages", [])}
    for rel in range(1, max_gap_rel + 1):
        link = by_gap_rel.get(rel, "")
        for cls in LINKAGE_CLASSES:
            feats[f"ppbpos_gap_rel_{rel:02d}_{cls}"] = float(link == cls)

    ps_positions = [idx for idx, link in enumerate(linkages) if link == "PS"]
    ps_set = set(ps_positions)
    ps_adjacent_pairs = sum(1 for idx in ps_positions if idx + 1 in ps_set)
    feats["ppbpos_PS_adjacent_pair_count"] = float(ps_adjacent_pairs)
    feats["ppbpos_PS_longest_run"] = float(_longest_run(ps_positions))
    feats["ppbpos_PS_terminal_imbalance"] = float(feats["ppbpos_PS_end5_count"] - feats["ppbpos_PS_end3_count"])
    feats["ppbpos_PS_gap5_vs_gap3"] = float(feats["ppbpos_PS_gap5_count"] - feats["ppbpos_PS_gap3_count"])

    for motif in MOTIFS:
        starts = _motif_positions(seq, motif)
        tag = motif.replace(" ", "_")
        near_ps = 0
        in_gap_near_ps = 0
        for start in starts:
            motif_link_window = set(range(max(0, start - 1), min(n_links, start + len(motif))))
            hit = bool(motif_link_window & ps_set)
            near_ps += int(hit)
            if hit and gap_start is not None and gap_end is not None and gap_start <= start < gap_end:
                in_gap_near_ps += 1
        feats[f"ppbpos_motif_{tag}_near_PS_count"] = float(near_ps)
        feats[f"ppbpos_motif_{tag}_in_gap_near_PS_count"] = float(in_gap_near_ps)

    for protein in PROTEIN_KEYS:
        score = float(attribution.get(protein, 0.0))
        feats[f"ppbpos_{protein}_score"] = score
        feats[f"ppbpos_{protein}_x_PS_gap5"] = score * feats["ppbpos_PS_gap5_count"]
        feats[f"ppbpos_{protein}_x_PS_centered"] = score * feats["ppbpos_PS_dist_gap_center_abs_mean"]
        feats[f"ppbpos_{protein}_x_PS_run"] = score * feats["ppbpos_PS_longest_run"]

    for key, value in site_scores.items():
        feats[f"ppbpos_site_{key}"] = float(value)
        feats[f"ppbpos_site_{key}_x_PS_gap5"] = float(value) * feats["ppbpos_PS_gap5_count"]

    return feats


def _longest_run(positions: list[int]) -> int:
    if not positions:
        return 0
    longest = 1
    current = 1
    prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
        prev = pos
    return max(longest, current)


def dataframe_for_helms(
    helms: Iterable[str],
    sequences: Iterable[str] | None = None,
    extra_names: Iterable[str] | None = None,
    max_link_pos: int = 30,
    max_gap_rel: int = 12,
) -> pd.DataFrame:
    helm_list = [str(h) for h in helms]
    seq_list = [None] * len(helm_list) if sequences is None else [str(s) for s in sequences]
    name_list = [""] * len(helm_list) if extra_names is None else [str(n) for n in extra_names]
    records = []
    for helm, seq, name in zip(helm_list, seq_list, name_list):
        try:
            records.append(
                features_for_helm(
                    helm,
                    sequence=seq,
                    extra_name=name,
                    max_link_pos=max_link_pos,
                    max_gap_rel=max_gap_rel,
                )
            )
        except Exception:
            records.append({})
    return pd.DataFrame(records).fillna(0.0).astype(float)
