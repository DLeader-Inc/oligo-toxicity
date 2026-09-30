"""Numeric PPB/protein-binding rule descriptors for endpoint models."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from rdkit import Chem

from src.features.chemistry_features import compute_all_features
from src.features.helm_to_smiles import helm_to_smiles


def _safe_mol_from_helm(helm: str):
    try:
        smiles = helm_to_smiles(str(helm))
        if not smiles:
            return None
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _flatten_features(bundle: dict) -> dict[str, float]:
    ps = bundle.get("ps_density", {})
    linkage = bundle.get("linkage_profile", {})
    linkage_counts = linkage.get("counts", {})
    methyl = bundle.get("methyl", {})
    conj = bundle.get("conjugation", {})
    hydro = bundle.get("hydrophobicity", {})
    attrib = bundle.get("protein_attribution", {})
    site = bundle.get("site_specific_backbone", {})
    site_scores = site.get("scores", {})

    out: dict[str, float] = {
        "ppbrule_n_ps": float(ps.get("n_ps", 0)),
        "ppbrule_n_total_linkages": float(ps.get("n_total", 0)),
        "ppbrule_ps_fraction": float(ps.get("ps_fraction", 0.0)),
        "ppbrule_ps_at_5_end": float(ps.get("ps_at_5_end", 0)),
        "ppbrule_ps_at_3_end": float(ps.get("ps_at_3_end", 0)),
        "ppbrule_gap_length": float(site.get("gap_length", 0) or 0),
        "ppbrule_n_5mC": float(methyl.get("n_5mC", 0)),
        "ppbrule_n_5mU": float(methyl.get("n_5mU", 0)),
        "ppbrule_any_methyl_pyrimidine": float(methyl.get("any_methyl_pyrimidine", 0)),
        "ppbrule_logP": float(hydro.get("logP", 0.0)),
        "ppbrule_mr": float(hydro.get("mr", 0.0)),
        "ppbrule_logP_per_atom": float(hydro.get("logP_per_atom", 0.0)),
        "ppbrule_HSA_score": float(attrib.get("HSA", 0.0)),
        "ppbrule_aagp_score": float(attrib.get("aagp", 0.0)),
        "ppbrule_factorH_score": float(attrib.get("factorH", 0.0)),
        "ppbrule_lipoprotein_score": float(attrib.get("lipoprotein", 0.0)),
        "ppbrule_n_risk_flags": float(len(bundle.get("risk_flags", []))),
    }
    for key in ("PS", "PO", "MP", "MOP", "MSPA"):
        out[f"ppbrule_linkage_count_{key}"] = float(linkage_counts.get(key, 0))
    for key in ("none", "GalNAc", "cholesterol", "fatty_acid", "peptide", "lipid_NP", "PEG"):
        out[f"ppbrule_conj_{key}"] = float(conj.get(key, 0))
    for key, value in site_scores.items():
        out[f"ppbrule_{key}"] = float(value)
    return out


def dataframe_for_helms(
    helms: Iterable[str],
    sequences: Iterable[str] | None = None,
    extra_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return numeric PPB/protein-binding rule features for each HELM row."""
    helm_list = [str(h) for h in helms]
    seq_list = [None] * len(helm_list) if sequences is None else [str(s) for s in sequences]
    name_list = [""] * len(helm_list) if extra_names is None else [str(n) for n in extra_names]
    records = []
    for helm, seq, name in zip(helm_list, seq_list, name_list):
        mol = _safe_mol_from_helm(helm)
        try:
            bundle = compute_all_features(helm, mol, sequence=seq, extra_name=name)
            records.append(_flatten_features(bundle))
        except Exception:
            records.append({k: 0.0 for k in _flatten_features({}).keys()})
    return pd.DataFrame(records).fillna(0.0).astype(float)


def explanation_bundle(helm: str, sequence: str | None = None, extra_name: str = "") -> dict:
    """Return the original nested rule bundle for human-readable examples."""
    mol = _safe_mol_from_helm(helm)
    return compute_all_features(str(helm), mol, sequence=sequence, extra_name=extra_name)
