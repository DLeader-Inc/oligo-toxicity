"""
Universal HELM featurizer for ASO toxicity prediction.

Parses HELM notation into base sequence + per-position chemistry, then
computes:
  - K-mer frequencies (1/2/3-mer)
  - Modification pattern features (LNA, MOE, 2'OMe, DNA counts/fractions)
  - Per-position chemistry flags
  - Toxicity motif features (CpG, TCC/TGC, G-quadruplex, etc.)

All features are derived from sequence + modification only — NO Tm or other
lab measurements are used. This makes the featurizer applicable to any
unscreened ASO for which only sequence + chemistry is known.
"""

from dataclasses import dataclass
from itertools import product
import re
import pandas as pd
from typing import Tuple, Dict

from src.features.chemistry_registry import (
    MODIFIED_SUGAR_CODES,
    canonical_base,
    sugar_descriptor_code,
)


# Single-letter codes for each chemistry: L=LNA, M=MOE, O=2'OMe, D=DNA, R=RNA
PATTERN_MAP = {
    "[lna]": "L",
    "[moe]": "M",
    "m": "O",
    "d": "D",
    "r": "R",
}


@dataclass(frozen=True)
class HelmResidue:
    """One residue parsed without requiring a closed monomer vocabulary."""

    raw_sugar: str
    raw_base: str
    raw_linkage: str
    modification: str
    base: str
    linkage: str


@dataclass(frozen=True)
class ReconciledOligoSequence:
    """Keep source sequence completeness separate from HELM chemistry certainty."""

    helm_sequence: str
    source_sequence: str
    complete_sequence: str
    modification_pattern: str
    linkage_pattern: tuple[str, ...]
    relation: str
    appended_source_bases: str
    appended_modifications: str
    appended_linkages: tuple[str, ...]


_HELM_RNA_RE = re.compile(r"(?:^|\|)RNA\d+\{(.*?)\}(?=\||\$)")
_GENERIC_MONOMER_RE = re.compile(
    r"^(?P<sugar>\[[^\]]+\]|[A-Za-z][A-Za-z0-9_+\-]*)"
    r"\((?P<base>\[[^\]]+\]|[^)]+)\)"
    r"(?P<linkage>.*)$"
)
_LINKAGE_CLASS = {
    "": "TERMINAL",
    "p": "PO",
    "sp": "PS",
    "mp": "MP",
    "mop": "MOP",
    "mspa": "MSPA",
}


def _strip_brackets(token: str) -> str:
    value = str(token).strip()
    return value[1:-1] if value.startswith("[") and value.endswith("]") else value


def _canonical_base(raw_base: str) -> str:
    token = _strip_brackets(raw_base)
    registered = canonical_base(token)
    if registered != "N":
        return registered
    aliases = {"M5U": "T", "5MU": "T", "5MT": "T"}
    return aliases.get(token.upper(), "N")


def _normalized_linkage(raw_linkage: str, *, terminal: bool) -> str:
    if terminal:
        return "TERMINAL"
    token = _strip_brackets(raw_linkage).strip().lower()
    if not token:
        return "PO"
    return _LINKAGE_CLASS.get(token, "OTHER")


def parse_helm_components(helm: str) -> tuple[HelmResidue, ...]:
    """Parse RNA HELM into fixed semantic classes plus raw fallback tokens.

    Unknown monomers remain present as ``OTHER``/``N`` rather than being
    silently dropped. The raw tokens are retained so a fixed hashing fallback
    can distinguish novel chemistry without fitting a dataset-specific
    vocabulary.
    """

    match = _HELM_RNA_RE.search(str(helm).strip())
    if not match:
        raise ValueError(f"Not a recognized RNA HELM string: {str(helm)[:80]!r}")
    units = match.group(1).split(".")
    parsed: list[tuple[str, str, str]] = []
    for unit in units:
        monomer = _GENERIC_MONOMER_RE.match(unit.strip())
        if not monomer:
            raise ValueError(f"Could not parse HELM monomer: {unit!r}")
        parsed.append(
            (
                monomer.group("sugar"),
                monomer.group("base"),
                monomer.group("linkage"),
            )
        )

    residues: list[HelmResidue] = []
    for index, (raw_sugar, raw_base, raw_linkage) in enumerate(parsed):
        residues.append(
            HelmResidue(
                raw_sugar=_strip_brackets(raw_sugar),
                raw_base=_strip_brackets(raw_base),
                raw_linkage=_strip_brackets(raw_linkage),
                modification=sugar_descriptor_code(_strip_brackets(raw_sugar)),
                base=_canonical_base(raw_base),
                linkage=_normalized_linkage(
                    raw_linkage,
                    terminal=index == len(parsed) - 1,
                ),
            )
        )
    return tuple(residues)


def parse_helm(helm: str) -> Tuple[str, str]:
    """Parse HELM notation into modification pattern and base sequence.

    Args:
        helm: HELM string, e.g. ``"RNA1{[lna](T)[sp].d(A)[sp].[moe](G)}$$$$"``

    Returns:
        ``(mod_pattern, bases)`` where:
          - ``mod_pattern`` uses ``L``/``M``/``O``/``D``/``R`` per position
          - ``bases`` is the nucleotide sequence (``ATCGN``, with ``U`` mapped to ``T``)
    """
    residues = parse_helm_components(helm)
    return (
        "".join(residue.modification for residue in residues),
        "".join(residue.base for residue in residues),
    )


def reconcile_helm_and_source_sequence(
    helm: str,
    source_sequence: object,
) -> ReconciledOligoSequence:
    """Reconcile source bases with HELM without inventing missing chemistry.

    HELM is authoritative for instantiated sugar/linkage chemistry.  A source
    sequence may contain terminal bases omitted during upstream conversion. In
    that case the bases are retained, while absent sugar and linkage values are
    represented explicitly as ``X`` and ``UNKNOWN`` rather than guessed.
    """

    residues = parse_helm_components(helm)
    helm_sequence = "".join(residue.base for residue in residues)
    source = "" if source_sequence is None or pd.isna(source_sequence) else str(source_sequence)
    source = "".join(
        char for char in source.upper().replace("U", "T") if char in "ACGTN"
    )
    modifications = "".join(residue.modification for residue in residues)
    linkages = tuple(residue.linkage for residue in residues[:-1])

    if not source:
        relation = "source_missing"
        complete = helm_sequence
        suffix = ""
    elif source == helm_sequence:
        relation = "exact"
        complete = helm_sequence
        suffix = ""
    elif source.startswith(helm_sequence):
        suffix = source[len(helm_sequence) :]
        relation = f"source_equals_helm_plus_3prime_suffix:{suffix}"
        complete = source
    else:
        relation = "non_terminal_mismatch"
        complete = helm_sequence
        suffix = ""

    # Each appended residue needs one preceding inter-residue bond. Neither
    # sugar nor bond chemistry is present in the packaged HELM, so retain
    # explicit unknown values until row-level source evidence resolves them.
    appended_modifications = "X" * len(suffix)
    appended_linkages = ("UNKNOWN",) * len(suffix)
    return ReconciledOligoSequence(
        helm_sequence=helm_sequence,
        source_sequence=source,
        complete_sequence=complete,
        modification_pattern=modifications + appended_modifications,
        linkage_pattern=linkages + appended_linkages,
        relation=relation,
        appended_source_bases=suffix,
        appended_modifications=appended_modifications,
        appended_linkages=appended_linkages,
    )


def count_motif(seq: str, motif: str) -> int:
    """Count overlapping occurrences of ``motif`` in ``seq``."""
    return sum(1 for i in range(len(seq) - len(motif) + 1)
               if seq[i:i + len(motif)] == motif)


def get_motif_features(bases: str, mod_pattern: str, max_pos: int = 20) -> Dict[str, float]:
    """Compute toxicity motif + modification features.

    Args:
        bases: Nucleotide sequence (ATCGN).
        mod_pattern: Per-position modification string (LMODR).
        max_pos: Number of per-position chemistry flags to emit.

    Returns:
        Dict of feature name → value.
    """
    f = {}

    # ── Toxicity motifs (literature-derived) ──────────────────────────
    f["n_CpG"] = count_motif(bases, "CG")             # B1: TLR9 activation
    f["n_TCC"] = count_motif(bases, "TCC")            # B2: hepatocellular protein binding
    f["n_TGC"] = count_motif(bases, "TGC")            # B2: same mechanism
    f["n_toxic_trinuc"] = f["n_TCC"] + f["n_TGC"]
    f["has_G4"] = 1 if re.search(r"G{4,}", bases) else 0   # B3: G-quadruplex
    f["has_G3"] = 1 if re.search(r"G{3,}", bases) else 0
    f["n_GG"] = count_motif(bases, "GG")
    f["n_TT"] = count_motif(bases, "TT")              # TLR7/8 (UU proxy)
    f["n_TG"] = count_motif(bases, "TG")              # TLR7/8 (UG proxy)
    f["gc_content"] = (bases.count("G") + bases.count("C")) / max(len(bases), 1)

    # ── Modification counts (chemistry-agnostic) ──────────────────────
    f["n_LNA"] = mod_pattern.count("L")
    f["n_MOE"] = mod_pattern.count("M")
    f["n_OMe"] = mod_pattern.count("O")
    f["n_DNA"] = mod_pattern.count("D")
    f["n_RNA"] = mod_pattern.count("R")
    f["n_cEt"] = mod_pattern.count("E")
    f["seq_len"] = len(bases)

    L = max(len(mod_pattern), 1)
    f["frac_LNA"] = f["n_LNA"] / L
    f["frac_MOE"] = f["n_MOE"] / L
    f["frac_OMe"] = f["n_OMe"] / L
    f["frac_DNA"] = f["n_DNA"] / L
    f["frac_cEt"] = f["n_cEt"] / L

    # ── Wing/gap structure (gapmer architecture) ──────────────────────
    f["mod_5prime_count"] = sum(1 for c in mod_pattern[:3] if c in MODIFIED_SUGAR_CODES)
    f["mod_3prime_count"] = sum(1 for c in mod_pattern[-3:] if c in MODIFIED_SUGAR_CODES)
    dna_runs = re.findall(r"D+", mod_pattern)
    f["gap_length"] = max((len(r) for r in dna_runs), default=0)
    modified_pattern = "".join("1" if c in MODIFIED_SUGAR_CODES else "0" for c in mod_pattern)
    mod_runs = re.findall(r"1+", modified_pattern)
    f["max_mod_run"] = max((len(r) for r in mod_runs), default=0)

    # ── Per-position chemistry flag (1 if modified, 0 if DNA) ─────────
    for pos in range(min(max_pos, len(mod_pattern))):
        f[f"mod_pos_{pos}"] = 1 if mod_pattern[pos] in MODIFIED_SUGAR_CODES else 0

    return f


def get_kmer_features(seq: str, k_values: Tuple[int, ...] = (1, 2, 3)) -> Dict[str, float]:
    """Compute normalized k-mer frequency features.

    K-mers containing ``N`` are skipped. Each k-mer count is divided by the
    total number of valid k-mers at that ``k``.
    """
    features = {}
    for k in k_values:
        kmers = {}
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if "N" in kmer:
                continue
            kmers[kmer] = kmers.get(kmer, 0) + 1
        total = max(sum(kmers.values()), 1)
        for kmer, count in kmers.items():
            features[f"kmer_{kmer}"] = count / total
    return features


MOTIF_FEATURE_NAMES = (
    "n_CpG",
    "n_TCC",
    "n_TGC",
    "n_toxic_trinuc",
    "has_G4",
    "has_G3",
    "n_GG",
    "n_TT",
    "n_TG",
    "gc_content",
)
CHEMISTRY_FEATURE_NAMES = (
    "n_LNA",
    "n_MOE",
    "n_OMe",
    "n_DNA",
    "n_RNA",
    "n_cEt",
    "seq_len",
    "frac_LNA",
    "frac_MOE",
    "frac_OMe",
    "frac_DNA",
    "frac_cEt",
    "mod_5prime_count",
    "mod_3prime_count",
    "gap_length",
    "max_mod_run",
)


def universal_core_feature_names(
    k_values: Tuple[int, ...] = (1, 2, 3),
    max_pos: int = 30,
) -> tuple[str, ...]:
    """Return the frozen global Core schema used by universal_v1."""

    if max_pos < 1:
        raise ValueError("max_pos must be positive")
    names: list[str] = []
    for k in k_values:
        names.extend(
            "kmer_" + "".join(token)
            for token in product("ACGT", repeat=int(k))
        )
    names.extend(MOTIF_FEATURE_NAMES)
    names.extend(CHEMISTRY_FEATURE_NAMES)
    names.extend(f"mod_pos_{position}" for position in range(max_pos))
    if len(names) != len(set(names)):
        raise RuntimeError("universal Core schema contains duplicate names")
    return tuple(names)


def featurize_dataset(df: pd.DataFrame,
                       helm_col: str = "x",
                       k_values: Tuple[int, ...] = (1, 2, 3),
                       max_pos: int = 20,
                       fixed_schema: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the full featurizer to a DataFrame containing a HELM column.

    Args:
        df: Input DataFrame with a HELM column.
        helm_col: Name of the HELM column. Default ``"x"`` matches OligoGym.
        k_values: K-mer sizes to compute. Default ``(1, 2, 3)``.
        max_pos: Max per-position chemistry flags.
        fixed_schema: Reindex to the frozen all-k-mer/all-position schema.

    Returns:
        ``(X, df_with_parsed)`` where:
          - ``X`` is the feature matrix (numeric, NaN-filled with 0)
          - ``df_with_parsed`` is a copy of the input with extra columns
            ``mod_pattern`` and ``bases`` for downstream traceability.
    """
    df = df.copy()
    df["mod_pattern"], df["bases"] = zip(*df[helm_col].apply(parse_helm))

    motif_features = df.apply(
        lambda row: get_motif_features(row["bases"], row["mod_pattern"], max_pos),
        axis=1,
        result_type="expand",
    )
    kmer_features = pd.DataFrame(
        df["bases"].apply(lambda s: get_kmer_features(s, k_values)).tolist()
    ).fillna(0)

    X = pd.concat([kmer_features, motif_features], axis=1).fillna(0)
    if fixed_schema:
        X = X.reindex(
            columns=universal_core_feature_names(k_values, max_pos),
            fill_value=0.0,
        )
    return X, df


def categorize_feature(feat: str) -> str:
    """Bucket a feature name into one of: K-mer, Per-position, Modification, Motif."""
    if feat.startswith("kmer_"):
        k = len(feat.replace("kmer_", ""))
        return f"K-mer ({k}-mer)"
    if feat.startswith("mod_pos_"):
        return "Per-position chemistry"
    modification_features = {
        "n_LNA", "n_MOE", "n_OMe", "n_DNA", "n_RNA", "seq_len",
        "frac_LNA", "frac_MOE", "frac_OMe", "frac_DNA",
        "mod_5prime_count", "mod_3prime_count", "gap_length", "max_mod_run",
    }
    if feat in modification_features:
        return "Modification"
    motif_features = {
        "n_CpG", "n_TCC", "n_TGC", "n_toxic_trinuc",
        "has_G4", "has_G3", "n_GG", "n_TT", "n_TG", "gc_content",
    }
    if feat in motif_features:
        return "Toxicity motif"
    return "Other"
