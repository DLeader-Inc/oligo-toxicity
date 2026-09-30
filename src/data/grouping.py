"""Canonical grouping contracts for leakage-safe oligonucleotide validation.

The OligoGym MOE endpoint contains a systematic representation discrepancy:
the source ``fasta`` field has one more terminal base than the packaged
HELM/SMILES molecule. Source-patent checks show that this is an upstream
conversion loss, not a disposable FASTA suffix. Grouping therefore preserves
an exact or terminally extended source sequence while retaining an explicit
warning that the appended residue chemistry is absent from HELM.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd

from src.features.featurizer import parse_helm, reconcile_helm_and_source_sequence


VALID_BASES = frozenset("ACGTN")


def normalize_base_sequence(value: object) -> str:
    """Normalize a nucleotide string for grouping without inventing bases."""

    if value is None or pd.isna(value):
        return ""
    text = str(value).upper().replace("U", "T")
    return "".join(char for char in text if char in VALID_BASES)


def stable_row_id(dataset_key: str, row_index: int, helm: object) -> str:
    """Return a deterministic, non-identifying row ID for split manifests."""

    payload = f"{dataset_key}\0{row_index}\0{helm}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_sequence_relation(molecular_sequence: str, source_fasta: str) -> str:
    """Classify how the source FASTA relates to the instantiated molecule."""

    if not source_fasta:
        return "source_fasta_missing"
    if molecular_sequence == source_fasta:
        return "exact"
    if source_fasta.startswith(molecular_sequence):
        suffix = source_fasta[len(molecular_sequence) :]
        return f"source_fasta_equals_molecule_plus_3prime_suffix:{suffix}"
    if source_fasta.endswith(molecular_sequence):
        prefix = source_fasta[: -len(molecular_sequence)] if molecular_sequence else source_fasta
        return f"source_fasta_equals_5prime_prefix_plus_molecule:{prefix}"
    if molecular_sequence.startswith(source_fasta):
        suffix = molecular_sequence[len(source_fasta) :]
        return f"molecule_equals_source_fasta_plus_3prime_suffix:{suffix}"
    if molecular_sequence.endswith(source_fasta):
        prefix = molecular_sequence[: -len(source_fasta)] if source_fasta else molecular_sequence
        return f"molecule_equals_5prime_prefix_plus_source_fasta:{prefix}"
    return "non_terminal_mismatch"


def build_sequence_contract(
    frame: pd.DataFrame,
    *,
    dataset_key: str,
    helm_col: str = "x",
    fasta_col: str = "fasta",
    parsed_sequence_col: str = "bases",
) -> pd.DataFrame:
    """Build row-level sequence provenance and the canonical group ID.

    Exact source/HELM matches and simple source terminal extensions use the
    complete source sequence. Other disagreements fall back to HELM so an
    arbitrary source annotation cannot silently redefine the molecule.
    """

    if helm_col not in frame:
        raise KeyError(f"Missing HELM column: {helm_col}")

    if parsed_sequence_col in frame:
        molecular = frame[parsed_sequence_col].map(normalize_base_sequence)
    else:
        molecular = frame[helm_col].map(lambda value: normalize_base_sequence(parse_helm(str(value))[1]))
    source_fasta = (
        frame[fasta_col].map(normalize_base_sequence)
        if fasta_col in frame
        else pd.Series([""] * len(frame), index=frame.index, dtype=object)
    )

    rows = []
    for position, (index, helm, molecule, fasta) in enumerate(
        zip(frame.index, frame[helm_col], molecular, source_fasta)
    ):
        relation = classify_sequence_relation(molecule, fasta)
        reconciled = reconcile_helm_and_source_sequence(str(helm), fasta)
        canonical = reconciled.complete_sequence
        rows.append(
            {
                "dataset": dataset_key,
                "row_position": position,
                "source_index": str(index),
                "row_id": stable_row_id(dataset_key, position, helm),
                "canonical_sequence_group_id": canonical,
                "group_basis": (
                    "source_complete_sequence_with_helm_chemistry_warning"
                    if canonical != molecule
                    else "helm_source_exact_sequence"
                ),
                "molecular_sequence": molecule,
                "source_fasta": fasta,
                "molecular_length": len(molecule),
                "source_fasta_length": len(fasta),
                "length_delta_source_minus_molecule": len(fasta) - len(molecule),
                "sequence_relation": relation,
                "appended_source_bases": reconciled.appended_source_bases,
                "appended_chemistry_status": (
                    "unknown_not_present_in_packaged_helm"
                    if reconciled.appended_source_bases
                    else "not_applicable"
                ),
                "representation_warning": "" if relation == "exact" else relation,
            }
        )
    return pd.DataFrame(rows)


def canonical_sequence_groups(frame: pd.DataFrame, *, dataset_key: str = "unknown") -> pd.Series:
    """Return the canonical HELM-derived sequence groups for a data frame."""

    contract = build_sequence_contract(frame, dataset_key=dataset_key)
    return contract["canonical_sequence_group_id"].reset_index(drop=True)


def split_manifest_rows(
    contract: pd.DataFrame,
    fold_ids: Iterable[int],
    *,
    validation: str,
) -> pd.DataFrame:
    """Attach deterministic fold assignments to a sequence contract."""

    result = contract.copy()
    result["validation"] = validation
    result["fold"] = list(fold_ids)
    return result
