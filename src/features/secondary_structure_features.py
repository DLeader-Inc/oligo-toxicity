"""ViennaRNA secondary-structure features for ASO toxicity models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.featurizer import parse_helm


MAX_POS_DEFAULT = 30
HB_WEIGHTS = {
    frozenset(("G", "C")): 3.0,
    frozenset(("A", "U")): 2.0,
    frozenset(("A", "T")): 2.0,
    frozenset(("G", "U")): 1.0,
    frozenset(("G", "T")): 1.0,
}
MOD_CODES = ("L", "M", "O", "D", "R")
STRUCTURE_CONTEXTS = ("paired", "stem", "loop", "unpaired", "long_range_pair")


@dataclass(frozen=True)
class FoldResult:
    sequence: str
    dotbracket: str
    mfe: float
    pair_map: dict[int, int]
    pair_probs: dict[tuple[int, int], float]


def normalize_base_sequence(seq: str) -> str:
    """Return an RNA-style A/C/G/U sequence for ViennaRNA."""
    out = []
    for base in str(seq).upper():
        if base == "T":
            out.append("U")
        elif base in "ACGU":
            out.append(base)
        else:
            out.append("N")
    return "".join(out)


def sequence_from_helm(helm: str) -> str:
    """Extract base sequence from HELM and normalize T to U."""
    _, bases = parse_helm(helm)
    return normalize_base_sequence(bases)


def parse_dotbracket(dotbracket: str) -> dict[int, int]:
    """Parse simple dot-bracket notation into 0-based paired positions."""
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for idx, char in enumerate(dotbracket):
        if char == "(":
            stack.append(idx)
        elif char == ")" and stack:
            left = stack.pop()
            pairs[left] = idx
            pairs[idx] = left
    return pairs


def fold_sequence(seq: str, with_pair_probs: bool = True) -> FoldResult:
    """Fold a sequence with ViennaRNA and return MFE plus optional pair probs."""
    import RNA

    rna_seq = normalize_base_sequence(seq)
    fc = RNA.fold_compound(rna_seq)
    dotbracket, mfe = fc.mfe()
    pair_probs: dict[tuple[int, int], float] = {}
    if with_pair_probs:
        try:
            fc.pf()
            bpp = fc.bpp()
            for i in range(1, len(rna_seq) + 1):
                for j in range(i + 1, len(rna_seq) + 1):
                    p = float(bpp[i][j])
                    if p > 0.0:
                        pair_probs[(i - 1, j - 1)] = p
        except Exception:
            pair_probs = {}
    return FoldResult(
        sequence=rna_seq,
        dotbracket=dotbracket,
        mfe=float(mfe),
        pair_map=parse_dotbracket(dotbracket),
        pair_probs=pair_probs,
    )


def _hb_weight(a: str, b: str) -> float:
    return HB_WEIGHTS.get(frozenset((a, b)), 0.0)


def _stem_lengths(pair_map: dict[int, int]) -> dict[int, int]:
    stems: dict[int, int] = {}
    seen: set[int] = set()
    for i, j in sorted((i, j) for i, j in pair_map.items() if i < j):
        if i in seen or j in seen:
            continue
        run = [(i, j)]
        a, b = i + 1, j - 1
        while pair_map.get(a) == b:
            run.append((a, b))
            a += 1
            b -= 1
        length = len(run)
        for x, y in run:
            stems[x] = length
            stems[y] = length
            seen.add(x)
            seen.add(y)
    return stems


def _hairpin_count(dotbracket: str) -> int:
    count = 0
    pairs = parse_dotbracket(dotbracket)
    for i, j in pairs.items():
        if i < j and all(c == "." for c in dotbracket[i + 1 : j]):
            count += 1
    return count


def _stem_runs(pair_map: dict[int, int]) -> list[list[tuple[int, int]]]:
    """Return contiguous helix runs as paired-index lists."""
    runs: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for i, j in sorted((i, j) for i, j in pair_map.items() if i < j):
        if (i, j) in seen:
            continue
        run: list[tuple[int, int]] = []
        a, b = i, j
        while pair_map.get(a) == b:
            run.append((a, b))
            seen.add((a, b))
            a += 1
            b -= 1
        runs.append(run)
    return runs


def _unpaired_runs(dotbracket: str) -> list[tuple[int, int]]:
    """Return inclusive-exclusive intervals of consecutive unpaired bases."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, char in enumerate(dotbracket):
        if char == "." and start is None:
            start = idx
        elif char != "." and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(dotbracket)))
    return runs


def _loop_intervals(dotbracket: str, pair_items: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Approximate loop/bulge intervals as unpaired runs enclosed by a pair."""
    unpaired = _unpaired_runs(dotbracket)
    intervals: list[tuple[int, int]] = []
    for start, end in unpaired:
        if any(i < start and end <= j for i, j in pair_items):
            intervals.append((start, end))
    return intervals


def _pair_only_component_count(pair_items: list[tuple[int, int]]) -> int:
    """Count connected components in the base-pair graph without backbone edges."""
    if not pair_items:
        return 0
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pair_items:
        union(i, j)
    return len({find(x) for x in parent})


def features_from_fold(fold: FoldResult, max_pos: int = MAX_POS_DEFAULT) -> dict[str, float]:
    """Convert a FoldResult into global and fixed-position features."""
    seq = fold.sequence
    n = len(seq)
    pair_items = [(i, j) for i, j in fold.pair_map.items() if i < j]
    pair_count = len(pair_items)
    hb_total = sum(_hb_weight(seq[i], seq[j]) for i, j in pair_items)
    stems = _stem_lengths(fold.pair_map)
    stem_runs = _stem_runs(fold.pair_map)
    unpaired_runs = _unpaired_runs(fold.dotbracket)
    loop_intervals = _loop_intervals(fold.dotbracket, pair_items)
    loop_lengths = [end - start for start, end in loop_intervals]
    pair_distances = [j - i for i, j in pair_items]
    norm_pair_distances = [d / max(n - 1, 1) for d in pair_distances]
    long_range_threshold = max(8, n // 2)
    enclosed = set()
    for i, j in pair_items:
        enclosed.update(range(i + 1, j))

    feats: dict[str, float] = {
        "ss_mfe": fold.mfe,
        "ss_mfe_per_nt": fold.mfe / max(n, 1),
        "ss_pair_count": float(pair_count),
        "ss_pair_fraction": float(2 * pair_count / max(n, 1)),
        "ss_hb_count": float(hb_total),
        "ss_hb_per_nt": float(hb_total / max(n, 1)),
        "ss_max_stem_len": float(max(stems.values(), default=0)),
        "ss_stem_count": float(len(stem_runs)),
        "ss_mean_stem_len": float(np.mean([len(run) for run in stem_runs])) if stem_runs else 0.0,
        "ss_hairpin_count": float(_hairpin_count(fold.dotbracket)),
        "ss_loop_count": float(len(loop_intervals)),
        "ss_mean_loop_len": float(np.mean(loop_lengths)) if loop_lengths else 0.0,
        "ss_max_loop_len": float(max(loop_lengths, default=0)),
        "ss_unpaired_count": float(sum(1 for char in fold.dotbracket if char == ".")),
        "ss_unpaired_fraction": float(sum(1 for char in fold.dotbracket if char == ".") / max(n, 1)),
        "ss_longest_unpaired_run": float(max((end - start for start, end in unpaired_runs), default=0)),
        "ss_mean_pair_distance_norm": float(np.mean(norm_pair_distances)) if norm_pair_distances else 0.0,
        "ss_max_pair_distance_norm": float(max(norm_pair_distances, default=0.0)),
        "ss_contact_order": float(np.mean(norm_pair_distances)) if norm_pair_distances else 0.0,
        "ss_long_range_pair_count": float(sum(d >= long_range_threshold for d in pair_distances)),
        "ss_pair_graph_components": float(_pair_only_component_count(pair_items)),
        "ss_mean_pair_prob": float(np.mean(list(fold.pair_probs.values()))) if fold.pair_probs else 0.0,
        "ss_max_pair_prob": float(max(fold.pair_probs.values(), default=0.0)),
        "ss_truncated_flag": float(n > max_pos),
    }

    for i in range(max_pos):
        paired = i in fold.pair_map
        partner = fold.pair_map.get(i, -1)
        hb = _hb_weight(seq[i], seq[partner]) if paired and i < n and partner < n else 0.0
        feats[f"ss_pos_{i+1:02d}_paired"] = float(paired)
        feats[f"ss_pos_{i+1:02d}_partner_norm"] = float((partner + 1) / max(n, 1)) if paired else 0.0
        feats[f"ss_pos_{i+1:02d}_pair_distance_norm"] = float(abs(partner - i) / max(n - 1, 1)) if paired else 0.0
        feats[f"ss_pos_{i+1:02d}_hb_weight"] = hb
        feats[f"ss_pos_{i+1:02d}_stem_flag"] = float(stems.get(i, 0) > 1)
        feats[f"ss_pos_{i+1:02d}_loop_flag"] = float((i < n) and (not paired) and (i in enclosed))
    return feats


def chemistry_structure_interaction_features(
    mod_pattern: str,
    fold: FoldResult,
    max_pos: int = MAX_POS_DEFAULT,
) -> dict[str, float]:
    """Describe where chemical modifications fall in the 2D fold.

    These features join two positional representations:
    - chemistry context from HELM (`L/M/O/D/R` per nucleotide)
    - structure context from ViennaRNA dot-bracket and pair map
    """
    mods = (mod_pattern or "")[: len(fold.sequence)]
    n = len(fold.sequence)
    if len(mods) < n:
        mods = mods + ("?" * (n - len(mods)))

    pair_items = [(i, j) for i, j in fold.pair_map.items() if i < j]
    stems = _stem_lengths(fold.pair_map)
    enclosed = set()
    pair_distances = {}
    for i, j in pair_items:
        enclosed.update(range(i + 1, j))
        pair_distances[i] = j - i
        pair_distances[j] = j - i
    long_range_threshold = max(8, n // 2)

    paired_positions = {idx for pair in pair_items for idx in pair}
    stem_positions = {idx for idx, length in stems.items() if length > 1}
    loop_positions = {idx for idx in range(n) if idx not in paired_positions and idx in enclosed}
    unpaired_positions = {idx for idx in range(n) if idx not in paired_positions}
    long_range_positions = {idx for idx, dist in pair_distances.items() if dist >= long_range_threshold}

    context_positions = {
        "paired": paired_positions,
        "stem": stem_positions,
        "loop": loop_positions,
        "unpaired": unpaired_positions,
        "long_range_pair": long_range_positions,
    }
    modified_positions = {i for i, mod in enumerate(mods) if mod in "LMO"}

    feats: dict[str, float] = {}
    for context, positions in context_positions.items():
        denom = max(len(positions), 1)
        mod_in_context = positions & modified_positions
        feats[f"sschem_{context}_modified_count"] = float(len(mod_in_context))
        feats[f"sschem_{context}_modified_fraction"] = float(len(mod_in_context) / denom)
        for code in MOD_CODES:
            count = sum(1 for idx in positions if idx < len(mods) and mods[idx] == code)
            total_code = max(mods.count(code), 1)
            feats[f"sschem_{context}_{code}_count"] = float(count)
            feats[f"sschem_{context}_{code}_within_context_fraction"] = float(count / denom)
            feats[f"sschem_{context}_{code}_within_mod_fraction"] = float(count / total_code)

    paired_modified = sum(1 for idx in paired_positions if idx in modified_positions)
    feats["sschem_modified_paired_fraction"] = float(paired_modified / max(len(modified_positions), 1))
    feats["sschem_modified_loop_fraction"] = float(len(loop_positions & modified_positions) / max(len(modified_positions), 1))
    feats["sschem_modified_unpaired_fraction"] = float(len(unpaired_positions & modified_positions) / max(len(modified_positions), 1))
    feats["sschem_modified_stem_fraction"] = float(len(stem_positions & modified_positions) / max(len(modified_positions), 1))

    # Pair-composition features expose whether paired bases bridge the DNA gap,
    # modified wings, or mixed chemistry contexts.
    pair_combo_counts: dict[str, int] = {}
    mismatched_pairs = 0
    modified_modified_pairs = 0
    modified_dna_pairs = 0
    for i, j in pair_items:
        a, b = mods[i], mods[j]
        combo = "_".join(sorted((a, b)))
        pair_combo_counts[combo] = pair_combo_counts.get(combo, 0) + 1
        if a != b:
            mismatched_pairs += 1
        if a in "LMO" and b in "LMO":
            modified_modified_pairs += 1
        if (a in "LMO" and b == "D") or (b in "LMO" and a == "D"):
            modified_dna_pairs += 1
    for a in MOD_CODES:
        for b in MOD_CODES:
            if a <= b:
                combo = f"{a}_{b}"
                feats[f"sschem_pair_{combo}_count"] = float(pair_combo_counts.get(combo, 0))
                feats[f"sschem_pair_{combo}_fraction"] = float(pair_combo_counts.get(combo, 0) / max(len(pair_items), 1))
    feats["sschem_pair_mod_mismatch_fraction"] = float(mismatched_pairs / max(len(pair_items), 1))
    feats["sschem_pair_modified_modified_fraction"] = float(modified_modified_pairs / max(len(pair_items), 1))
    feats["sschem_pair_modified_dna_fraction"] = float(modified_dna_pairs / max(len(pair_items), 1))

    for i in range(max_pos):
        mod = mods[i] if i < len(mods) else "?"
        paired = i in paired_positions
        loop = i in loop_positions
        stem = i in stem_positions
        for code in MOD_CODES:
            is_code = mod == code
            feats[f"sschem_pos_{i+1:02d}_{code}_paired"] = float(is_code and paired)
            feats[f"sschem_pos_{i+1:02d}_{code}_loop"] = float(is_code and loop)
            feats[f"sschem_pos_{i+1:02d}_{code}_stem"] = float(is_code and stem)

    return feats


def features_for_sequence(seq: str, max_pos: int = MAX_POS_DEFAULT, with_pair_probs: bool = True) -> dict[str, float]:
    return features_from_fold(fold_sequence(seq, with_pair_probs=with_pair_probs), max_pos=max_pos)


def dataframe_for_helms(
    helms: Iterable[str],
    max_pos: int = MAX_POS_DEFAULT,
    with_pair_probs: bool = True,
) -> pd.DataFrame:
    records = []
    for helm in helms:
        mod_pattern, _ = parse_helm(str(helm))
        seq = sequence_from_helm(str(helm))
        fold = fold_sequence(seq, with_pair_probs=with_pair_probs)
        row = {
            "ss_sequence": seq,
            "ss_dotbracket": fold.dotbracket,
            **features_from_fold(fold, max_pos=max_pos),
            **chemistry_structure_interaction_features(mod_pattern, fold, max_pos=max_pos),
        }
        records.append(row)
    return pd.DataFrame(records).fillna(0)
