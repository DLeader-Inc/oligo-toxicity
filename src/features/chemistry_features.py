"""Chemistry-level features for plasma-protein-binding (PPB) attribution.

Layer 2 of the universal predictor: given a HELM/SMILES and optional
stereo annotation, compute a small set of physically-grounded features
that drive plasma-protein-binding behavior, then convert them to a
per-protein attribution dict {HSA, aagp, factorH, lipoprotein}.

These features are deliberately rule-based (not learned) — the goal is
explainability, not pure prediction accuracy. The rules come from
Gupta et al. 2025 (J. Med. Chem. 68:9848) which catalogues which
plasma proteins are responsible for which oligonucleotide toxicity
patterns.
"""
import re
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Crippen

from src.features.helm_to_smiles import parse_helm


# ─── Sequence / HELM parsers ────────────────────────────────────────

def ps_density(helm: str, stereo_list: Optional[list] = None) -> dict:
    """Count phosphorothioate (PS) linkages and their positions.

    Returns:
        {
          "n_ps": int,             # total PS bridges
          "n_total": int,          # total internucleotide bridges
          "ps_fraction": float,    # n_ps / n_total
          "positions": [int],      # 0-indexed positions of PS bridges
          "ps_at_5_end": int,      # PS count in first 5 positions
          "ps_at_3_end": int,      # PS count in last 5 positions
        }

    A position is counted PS if the HELM marker `[sp]` is present OR
    if stereo_list at that position is one of {Rp, Sp, racemate}.
    """
    n_ps = 0
    positions = []
    n_total = 0

    if "RNA1{" in helm:
        body = helm.split("RNA1{")[1].split("}")[0]
        residues = body.split(".")
        n_total = max(0, len(residues) - 1)
        for i in range(n_total):
            # Linkage marker is after the i-th residue
            res = residues[i]
            if "[sp]" in res:
                n_ps += 1
                positions.append(i)

    if stereo_list is not None and n_total > 0:
        # Override based on stereo_list — Rp / Sp / racemate all mean PS
        n_ps = sum(1 for s in stereo_list if s in ("Rp", "Sp", "racemate"))
        positions = [i for i, s in enumerate(stereo_list)
                     if s in ("Rp", "Sp", "racemate")]

    return {
        "n_ps": n_ps,
        "n_total": n_total,
        "ps_fraction": n_ps / max(n_total, 1),
        "positions": positions,
        "ps_at_5_end": sum(1 for p in positions if p < 5),
        "ps_at_3_end": sum(1 for p in positions if p >= n_total - 5),
    }


def linkage_profile(helm: str, stereo_list: Optional[list] = None) -> dict:
    """Parse per-linkage backbone chemistry from HELM.

    Linkage positions are 0-indexed and refer to the bridge after residue i.
    The supported v1 linkage classes are:
      PS, PO, MP, MOP, MSPA.

    Stereo labels {Rp, Sp, racemate} are treated as evidence that a linkage
    is PS only when the HELM linkage itself is PO/unknown; explicit MP/MOP/MSPA
    tokens remain their chemically specified class.
    """
    if not helm:
        return {
            "monomers": [],
            "linkages": [],
            "n_total": 0,
            "counts": {"PS": 0, "PO": 0, "MP": 0, "MOP": 0, "MSPA": 0},
        }

    try:
        monomers, linkages = parse_helm(helm)
    except Exception:
        return {
            "monomers": [],
            "linkages": [],
            "n_total": 0,
            "counts": {"PS": 0, "PO": 0, "MP": 0, "MOP": 0, "MSPA": 0},
            "parse_error": 1,
        }

    normalized = [link.upper() for link in linkages]
    if stereo_list is not None:
        for idx, stereo in enumerate(stereo_list[:len(normalized)]):
            if stereo in ("Rp", "Sp", "racemate") and normalized[idx] == "PO":
                normalized[idx] = "PS"

    counts = {key: 0 for key in ("PS", "PO", "MP", "MOP", "MSPA")}
    for link in normalized:
        counts[link] = counts.get(link, 0) + 1

    return {
        "monomers": monomers,
        "linkages": normalized,
        "n_total": len(normalized),
        "counts": counts,
    }


def _longest_dna_gap(monomers: list[tuple[str, str]]) -> tuple[int, int] | None:
    best_start = None
    best_len = 0
    cur_start = None
    cur_len = 0
    for idx, (sugar, _) in enumerate(monomers):
        if sugar == "DNA":
            if cur_start is None:
                cur_start = idx
            cur_len += 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_start = None
            cur_len = 0
    if best_start is None:
        return None
    return best_start, best_start + best_len


def gap_linkage_profile(helm: str, stereo_list: Optional[list] = None) -> dict:
    """Return linkage chemistry inside the longest DNA gap.

    `gap_start` is inclusive and `gap_end` is exclusive in residue coordinates.
    `gap_relative_position` is 1-indexed from the 5' side and counts internal
    linkages within the DNA gap. For a 5-10-5 gapmer, the first internal gap
    linkage connects DNA gap bases 1 and 2.
    """
    profile = linkage_profile(helm, stereo_list)
    monomers = profile.get("monomers", [])
    linkages = profile.get("linkages", [])
    gap = _longest_dna_gap(monomers)
    if gap is None:
        return {
            **profile,
            "gap_start": None,
            "gap_end": None,
            "gap_length": 0,
            "gap_linkages": [],
        }

    gap_start, gap_end = gap
    gap_links = []
    for link_idx in range(gap_start, max(gap_start, gap_end - 1)):
        if link_idx >= len(linkages):
            continue
        gap_links.append({
            "absolute_position": link_idx,
            "gap_relative_position": link_idx - gap_start + 1,
            "linkage": linkages[link_idx],
        })

    return {
        **profile,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "gap_length": gap_end - gap_start,
        "gap_linkages": gap_links,
    }


def site_specific_backbone_rules(gap_profile: dict) -> dict:
    """Evidence rules from Migawa et al. 2019 and Anderson et al. 2021.

    These are explainability scores, not calibrated probabilities. They capture
    intracellular PS-protein-binding risk around the 5' DNA gap junction and
    the protective/tuning evidence from neutral phosphonates and MsPA.
    """
    gap_links = gap_profile.get("gap_linkages", [])
    counts = gap_profile.get("counts", {})
    by_rel = {x["gap_relative_position"]: x["linkage"] for x in gap_links}
    sensitive = {2, 3}
    mspa_window = {1, 2, 3, 4, 5}
    neutral = {"MP", "MOP"}

    sensitive_ps = sorted(pos for pos in sensitive if by_rel.get(pos) == "PS")
    sensitive_neutral = sorted(pos for pos in sensitive if by_rel.get(pos) in neutral)
    sensitive_mspa = sorted(pos for pos in sensitive if by_rel.get(pos) == "MSPA")
    mspa_5prime = sorted(pos for pos in mspa_window if by_rel.get(pos) == "MSPA")
    ps_in_gap = sum(1 for x in gap_links if x["linkage"] == "PS")
    gap_len = max(int(gap_profile.get("gap_length") or 0), 1)

    intracellular_risk = min(1.0, 0.55 * (ps_in_gap / gap_len)
                             + 0.225 * len(sensitive_ps))
    neutral_protection = min(1.0, 0.5 * len(sensitive_neutral))
    mspa_protection = min(1.0, 0.25 * len(mspa_5prime)
                          + 0.2 * len(sensitive_mspa))
    duration_bonus = min(1.0, 0.18 * counts.get("MSPA", 0))

    evidence = []
    counterfactuals = []
    if sensitive_ps:
        evidence.append(
            "PS at 5' DNA-gap linkage position(s) "
            + ",".join(str(p) for p in sensitive_ps)
            + " matches NAR 2019 intracellular protein-binding risk pattern"
        )
        counterfactuals.append(
            "Consider replacing gap 5' linkage position 2 or 3 with MOP/MP "
            "or tuned MsPA to reduce P54nrb/P21/caspase-linked risk"
        )
    if sensitive_neutral:
        evidence.append(
            "Neutral phosphonate at sensitive 5' gap position(s) "
            + ",".join(str(p) for p in sensitive_neutral)
            + " is protective in the NAR 2019 design rule"
        )
    if mspa_5prime:
        evidence.append(
            "MsPA in 5' DNA-gap position(s) "
            + ",".join(str(p) for p in mspa_5prime)
            + " supports the NAR 2021 protein-binding/pro-inflammatory tuning rule"
        )
    if counts.get("MSPA", 0) >= 3:
        evidence.append(
            "Multiple MsPA linkages support a nuclease-stability/duration benefit signal"
        )
    if not evidence:
        evidence.append("No 5' DNA-gap position 2/3 backbone risk signal detected")

    return {
        "scores": {
            "intracellular_ps_binding_risk": round(intracellular_risk, 3),
            "gap_5prime_ps_2_3_risk": round(min(1.0, 0.5 * len(sensitive_ps)), 3),
            "neutral_phosphonate_protection": round(neutral_protection, 3),
            "mspa_tuned_binding_protection": round(mspa_protection, 3),
            "duration_stability_bonus": round(duration_bonus, 3),
        },
        "evidence": evidence,
        "counterfactuals": counterfactuals,
    }


def methyl_pyrimidine_flag(helm: str) -> dict:
    """Count 5-methyl-C and 5-methyl-U residues.

    Gupta 2025 §5.1: 'all the above ASO drugs except PMOs have 5-methyl
    cytosine and 5-methyl uridine incorporated for mitigating
    unfavorable immune responses.' So this is a TLR-mitigation signal.

    Looks for `[m5C]`, `[m5U]`, `[5mT]`, `mC`, `mU` markers in the HELM
    string. Current v0 also counts plain thymine `T` as the DNA methyl-
    pyrimidine analogue for this mitigation flag; preserve that behavior
    when reproducing existing PPB/TLR-risk outputs.
    """
    n_mC = len(re.findall(r"\[(?:m5C|5mC)\]|mC", helm))
    n_mU = len(re.findall(r"\[(?:m5U|5mU|5mT|T)\]|mU", helm))
    return {
        "n_5mC": n_mC,
        "n_5mU": n_mU,
        "any_methyl_pyrimidine": int((n_mC + n_mU) > 0),
    }


CONJUGATION_TOKENS = [
    # token list aligns with terms used in Gupta 2025 §3.2 / §5
    ("none", []),
    ("GalNAc",      ["GalNAc", "tri-GalNAc", "ASGPR"]),
    ("cholesterol", ["cholesterol", "chol", "alpha-tocopherol", "tocopherol"]),
    ("fatty_acid",  ["palmitic", "docosanoic", "myristic", "C16", "C18",
                     "PalmA", "DocA"]),
    ("peptide",     ["peptide", "CPP", "Pip", "ILFQY", "Tat"]),
    ("lipid_NP",    ["LNP", "lipid_NP", "lipid-NP", "DLin"]),
    ("PEG",         ["PEG", "pegol"]),
]


def conjugation_flag(input_str: str) -> dict:
    """Detect a conjugation moiety in the HELM string, SMILES, or name.

    Returns a one-hot dict over CONJUGATION_TOKENS keys.
    """
    s = (input_str or "").lower()
    out = {tok: 0 for tok, _ in CONJUGATION_TOKENS}
    found_any = False
    for tok, kws in CONJUGATION_TOKENS:
        if tok == "none":
            continue
        for kw in kws:
            if kw.lower() in s:
                out[tok] = 1
                found_any = True
                break
    out["none"] = 0 if found_any else 1
    return out


def crippen_hydrophobicity(mol) -> dict:
    """Bulk hydrophobicity via Crippen logP.

    Returns:
        {"logP": float, "mr": float (molar refractivity),
         "logP_per_atom": float}
    """
    if mol is None:
        return {"logP": 0.0, "mr": 0.0, "logP_per_atom": 0.0}
    try:
        # Use canonical form (Crippen sometimes errors on chirality-tagged mols)
        smi = Chem.MolToSmiles(mol)
        m2 = Chem.MolFromSmiles(smi)
        if m2 is None:
            m2 = mol
        logp = float(Crippen.MolLogP(m2))
        mr = float(Crippen.MolMR(m2))
        n_heavy = max(m2.GetNumHeavyAtoms(), 1)
    except Exception:
        logp, mr = 0.0, 0.0
        n_heavy = max(mol.GetNumHeavyAtoms(), 1)
    return {
        "logP": logp,
        "mr": mr,
        "logP_per_atom": logp / n_heavy,
    }


# ─── Per-protein attribution rules ─────────────────────────────────

def attribution_rules(ps: dict, methyl: dict, conj: dict, hydro: dict,
                       sequence: Optional[str] = None) -> dict:
    """Compute a per-protein attribution score in [0, 1].

    The four channels correspond to the four major plasma binders the
    Gupta paper identifies. Higher score = stronger predicted binding
    to that protein.

    Rules:
        HSA          ~ PS density × hydrophobicity (Henry, Geary, refs 8, 35)
        α-1-AGP      ~ basic/cationic conjugation flags (peptide, guanidinium)
        Factor H     ~ PS density × G-doublet presence (ref 52 — monkeys ACP)
        Lipoprotein  ~ hydrophobic conjugation flag (cholesterol/fatty acid)
    """
    # HSA — proportional to PS fraction and bulk hydrophobicity.
    # Logp normalised to [-5, +5] → [0, 1]
    logp_norm = max(0.0, min(1.0, (hydro["logP"] + 50.0) / 100.0))
    # 50% PS = ~0.8 HSA, 100% PS = 1.0
    hsa_score = 0.4 * min(1.0, ps["ps_fraction"] * 2) + 0.4 * logp_norm \
                + 0.2 * (1 - conj["GalNAc"])  # GalNAc redirects to ASGPR

    # α-1-AGP — basic-drug carrier
    aagp_score = 0.6 * conj["peptide"] + 0.4 * conj.get("phosphoryl_guan", 0)

    # Factor H — PS density × G-doublet/G-tract presence
    g_doublet = 0
    if sequence is not None:
        u = sequence.upper().replace("U", "T")
        g_doublet = 1 if "GG" in u else 0
        # extra weight if G-tract (GGG+)
        if "GGG" in u:
            g_doublet = 2
    factorH_score = min(1.0,
                         0.6 * min(1.0, ps["n_ps"] / 15.0)
                         + 0.4 * (g_doublet / 2.0))
    # Only fire if at least 10 PS linkages
    if ps["n_ps"] < 10:
        factorH_score *= 0.3  # weaker for short PS

    # Lipoprotein — hydrophobic conjugates
    lipo_score = 0.5 * conj["cholesterol"] + 0.4 * conj["fatty_acid"] \
                 + 0.3 * conj["lipid_NP"]
    lipo_score = min(1.0, lipo_score)

    return {
        "HSA": round(min(1.0, hsa_score), 3),
        "aagp": round(min(1.0, aagp_score), 3),
        "factorH": round(min(1.0, factorH_score), 3),
        "lipoprotein": round(min(1.0, lipo_score), 3),
    }


# ─── Convenience: bundle everything for one molecule ───────────────

def compute_all_features(helm: str, mol, sequence: Optional[str] = None,
                          stereo_list: Optional[list] = None,
                          extra_name: str = "") -> dict:
    """One-shot wrapper: HELM + RDKit Mol → all chemistry features +
    attribution dict. `extra_name` is appended to the conjugation
    detection (for drugs whose name carries the moiety like
    'Eplontersen GalNAc')."""
    ps = ps_density(helm, stereo_list)
    methyl = methyl_pyrimidine_flag(helm)
    conj_input = (helm or "") + " " + (extra_name or "")
    conj = conjugation_flag(conj_input)
    hydro = crippen_hydrophobicity(mol)
    attrib = attribution_rules(ps, methyl, conj, hydro, sequence=sequence)
    gap_profile = gap_linkage_profile(helm, stereo_list)
    site_rules = site_specific_backbone_rules(gap_profile)
    risk_flags = []
    if attrib["factorH"] >= 0.5:
        risk_flags.append(
            f"Factor H risk: {ps['n_ps']} PS + G-doublet (ACP-activation)")
    if attrib["HSA"] >= 0.8 and not conj["GalNAc"]:
        risk_flags.append(
            f"High HSA binding ({attrib['HSA']:.2f}) without GalNAc — "
            "high systemic dose typical")
    if conj["cholesterol"] and ps["n_ps"] > 8:
        risk_flags.append(
            "Cholesterol + heavy PS — Wave-class mortality risk (ref 55)")
    if methyl["any_methyl_pyrimidine"] == 0 and ps["n_ps"] >= 10:
        risk_flags.append(
            "Heavy PS without 5-methyl-C/U — TLR response not mitigated")
    site_scores = site_rules["scores"]
    if site_scores["gap_5prime_ps_2_3_risk"] > 0:
        risk_flags.append(
            "5' DNA-gap PS position 2/3 — intracellular protein-binding "
            "risk (P54nrb/P21/caspase pattern)")
    if site_scores["neutral_phosphonate_protection"] > 0:
        risk_flags.append(
            "Neutral phosphonate at 5' gap position 2/3 — protective "
            "backbone-tuning signal")
    if site_scores["mspa_tuned_binding_protection"] >= 0.5:
        risk_flags.append(
            "MsPA-enriched 5' DNA gap — reduced PS-protein-binding / "
            "pro-inflammatory signal")
    return {
        "ps_density": ps,
        "linkage_profile": {
            "counts": gap_profile.get("counts", {}),
            "n_total": gap_profile.get("n_total", 0),
        },
        "site_specific_backbone": {
            "gap_start": gap_profile.get("gap_start"),
            "gap_end": gap_profile.get("gap_end"),
            "gap_length": gap_profile.get("gap_length", 0),
            "gap_linkages": gap_profile.get("gap_linkages", []),
            **site_rules,
        },
        "methyl": methyl,
        "conjugation": conj,
        "hydrophobicity": hydro,
        "protein_attribution": attrib,
        "risk_flags": risk_flags,
    }
