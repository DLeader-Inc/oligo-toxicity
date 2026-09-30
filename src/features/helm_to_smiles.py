"""
Convert HELM notation (used by OligoGym) into atomistic SMILES.

The HELM format used by OligoGym is a restricted subset of the Pistoia
HELM standard:

    RNA1{<MONOMER>[<LINKAGE>].<MONOMER>[<LINKAGE>]....<MONOMER>}$$$$

Where:
  MONOMER is one of:
    d(A), d(T), d(C), d(G), d([m5C])           # DNA (deoxyribose)
    r(A), r(U), r(C), r(G)                     # RNA (ribose)
    m(A), m(U), m(C), m(G)                     # 2'-O-Methyl
    [moe](A), [moe](T), [moe](C), [moe](G),
    [moe]([m5C])                                # 2'-O-Methoxyethyl
    [lna](A), [lna](T), [lna](C), [lna](G)     # Locked Nucleic Acid

  LINKAGE is one of:
    [sp]     # phosphorothioate (P=S)
    p        # phosphodiester (P=O), default if omitted
    [mp]     # methylphosphonate
    [mop]    # methoxypropyl phosphonate
    [mspa]   # mesyl phosphoramidate

The polymer is built using RDKit's RWMol API: each nucleoside is parsed
as a standalone Mol with attachment atoms tagged via atom-map labels
(5'-O = 5, 3'-O = 3), then a phosphate (or phosphorothioate) is inserted
between consecutive nucleosides by adding explicit bonds.

Output is a fully-bonded SMILES of the entire oligonucleotide
(~300-500 heavy atoms for typical ASOs), suitable for atom-level GNNs
like Chemprop, Uni-Mol, or MACE-OFF.
"""

from __future__ import annotations
import re
from typing import Optional

try:
    from rdkit import Chem
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "RDKit is required for HELM-to-SMILES conversion. "
        "Install with: pip install rdkit"
    ) from e


# ─────────────────────────────────────────────────────────────────────────
# Per-nucleoside SMILES with attachment-atom tags
#
# Each nucleoside SMILES contains exactly two oxygens marked with atom-map
# labels:
#   [O:5]   = the 5' hydroxyl oxygen
#   [O:3]   = the 3' hydroxyl oxygen
# We locate them by ``GetAtomMapNum()`` after parsing, then bond them to
# the phosphate that joins this nucleoside to its neighbor.
# ─────────────────────────────────────────────────────────────────────────

# Base SMILES fragments. The token ``[N9]`` marks the atom that bonds to
# C1' of the sugar (purines: N9; pyrimidines: N1). We replace it with the
# attached sugar atom at build time.
_BASE_SMILES = {
    # Purines (sugar attaches via N9)
    "A":   "[N9]1cnc2c(ncnc12)N",                  # adenine
    "G":   "[N9]1cnc2c1nc(N)[nH]c2=O",             # guanine
    # Pyrimidines (sugar attaches via N1)
    "C":   "[N9]1ccc(N)nc1=O",                     # cytosine
    "T":   "[N9]1cc(C)c(=O)[nH]c1=O",              # thymine
    "U":   "[N9]1ccc(=O)[nH]c1=O",                 # uracil
    "m5C": "[N9]1cc(C)c(N)nc1=O",                  # 5-methyl-cytosine
}


# Sugar SMILES — note the ``{BASE}`` token where the base attaches, and
# [O:5]/[O:3] tags. Each SMILES is a self-contained ring system; we
# render them as separate Mols then bond into a polymer.
# [OH:5] / [OH:3] — terminal 5'/3' hydroxyls. Explicit H + atom map.
# When joined via a phosphate the H is dropped programmatically.
_SUGAR_SMILES = {
    "DNA":   "[OH:5]C[C@H]1O[C@@H]({BASE})C[C@@H]1[OH:3]",
    "RNA":   "[OH:5]C[C@H]1O[C@@H]({BASE})[C@H](O)[C@@H]1[OH:3]",
    "2OMe":  "[OH:5]C[C@H]1O[C@@H]({BASE})[C@H](OC)[C@@H]1[OH:3]",
    "MOE":   "[OH:5]C[C@H]1O[C@@H]({BASE})[C@H](OCCOC)[C@@H]1[OH:3]",
    # LNA: 4'-C,2'-O methylene bridge. The furanose is ring 1; the bridge
    # forms ring 2. Ring numbers 1/2 don't conflict with bases (which use
    # higher numbers via the [N9] → n([*:98]) substitution path).
    "LNA":   "[OH:5]C[C@]12CO[C@@H]1[C@H]([C@H]2{BASE})[OH:3]",
}


# ─────────────────────────────────────────────────────────────────────────
# HELM parser
# ─────────────────────────────────────────────────────────────────────────

_SUGAR_MAP = {
    "d":      "DNA",
    "[d]":    "DNA",
    "r":      "RNA",
    "m":      "2OMe",
    "[moe]":  "MOE",
    "[lna]":  "LNA",
}

# Match one monomer like d(A), [lna](T), m([m5C])
_MONOMER_RE = re.compile(r"^(\[(?:moe|lna|d)\]|m|d|r)\((\[(?:m5C|5meC)\]|[ATCGU])\)(.*)$")


def parse_helm(helm: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Parse HELM string → (monomers, linkages).

    Returns
    -------
    monomers : list of (sugar_code, base_code)
        sugar_code ∈ {DNA, RNA, 2OMe, MOE, LNA}
        base_code ∈ {A, T, C, G, U, m5C}
    linkages : list of {PO, PS}
        len(linkages) == len(monomers) - 1
    """
    m = re.match(r"^RNA1\{(.+)\}\$\$\$\$\s*$", helm.strip())
    if not m:
        raise ValueError(f"Not a recognized RNA HELM string: {helm[:80]!r}")
    body = m.group(1)
    units = body.split(".")

    monomers: list[tuple[str, str]] = []
    linkages: list[str] = []
    for unit in units:
        mm = _MONOMER_RE.match(unit)
        if not mm:
            raise ValueError(f"Could not parse HELM monomer: {unit!r}")
        sugar_prefix, base, rest = mm.group(1), mm.group(2), mm.group(3)
        sugar = _SUGAR_MAP[sugar_prefix]
        base_code = base.strip("[]") if base.startswith("[") else base
        if base_code == "5meC":
            base_code = "m5C"
        monomers.append((sugar, base_code))
        if rest:
            token = rest.lower()
            if token == "[sp]":
                linkages.append("PS")
            elif token == "p":
                linkages.append("PO")
            elif token == "[mp]":
                linkages.append("MP")  # methylphosphonate
            elif token == "[mop]":
                linkages.append("MOP")  # methoxypropyl phosphonate
            elif token == "[mspa]":
                linkages.append("MSPA")  # mesyl phosphoramidate
            else:
                raise ValueError(f"Unknown linkage token: {rest!r}")
        else:
            # An omitted token before the next dot is a standard PO linkage.
            linkages.append("PO")

    if len(linkages) == len(monomers):
        # The terminal residue contributes no outgoing linkage.
        linkages = linkages[:-1]
    if len(linkages) != len(monomers) - 1:
        raise ValueError(
            f"Linkage count mismatch: {len(linkages)} linkages for "
            f"{len(monomers)} monomers in {helm[:80]!r}"
        )
    return monomers, linkages


# ─────────────────────────────────────────────────────────────────────────
# Molecule builder using RDKit RWMol
# ─────────────────────────────────────────────────────────────────────────

def _build_nucleoside_mol(
    sugar: str,
    base: str,
    residue_index: int | None = None,
) -> "Chem.Mol":
    """Return an RDKit Mol of one free nucleoside, with the 5'-O atom and
    3'-O atom tagged via atom-map numbers 5 and 3 respectively. The atoms
    are returned in a Mol that contains both sugar + base bonded at N1/N9.
    """
    if sugar not in _SUGAR_SMILES:
        raise ValueError(f"Unknown sugar: {sugar!r}")
    if base not in _BASE_SMILES:
        raise ValueError(f"Unknown base: {base!r}")

    # Build sugar with a dummy atom in place of the base attachment.
    sugar_smi = _SUGAR_SMILES[sugar].format(BASE="[*:99]")
    sugar_mol = Chem.MolFromSmiles(sugar_smi)
    if sugar_mol is None:
        raise ValueError(f"Bad sugar SMILES (sugar={sugar}): {sugar_smi!r}")

    # Build base with a dummy atom in place of the sugar attachment.
    base_smi = _BASE_SMILES[base].replace("[N9]", "n([*:98])")
    base_mol = Chem.MolFromSmiles(base_smi)
    if base_mol is None:
        raise ValueError(f"Bad base SMILES (base={base}): {base_smi!r}")

    # Properties survive CombineMols and atom deletion, giving downstream
    # atom models a reversible atom-to-residue/component map.
    for atom in sugar_mol.GetAtoms():
        atom.SetProp("oligo_component", "sugar")
        atom.SetProp("oligo_sugar_code", sugar)
        atom.SetProp("oligo_base_code", base)
        if residue_index is not None:
            atom.SetIntProp("oligo_residue_index", int(residue_index))
    for atom in base_mol.GetAtoms():
        atom.SetProp("oligo_component", "base")
        atom.SetProp("oligo_sugar_code", sugar)
        atom.SetProp("oligo_base_code", base)
        if residue_index is not None:
            atom.SetIntProp("oligo_residue_index", int(residue_index))

    # Combine and join via dummy atoms 98 and 99.
    combo = Chem.RWMol(Chem.CombineMols(sugar_mol, base_mol))
    dummy_99 = next(a.GetIdx() for a in combo.GetAtoms() if a.GetAtomMapNum() == 99)
    dummy_98 = next(a.GetIdx() for a in combo.GetAtoms() if a.GetAtomMapNum() == 98)
    # Find the heavy-atom neighbor of each dummy
    sugar_attach = next(
        n.GetIdx() for n in combo.GetAtomWithIdx(dummy_99).GetNeighbors()
    )
    base_attach = next(
        n.GetIdx() for n in combo.GetAtomWithIdx(dummy_98).GetNeighbors()
    )
    # Add bond and remove dummies
    combo.AddBond(sugar_attach, base_attach, Chem.BondType.SINGLE)
    # Remove higher idx first
    for idx in sorted([dummy_99, dummy_98], reverse=True):
        combo.RemoveAtom(idx)
    return combo.GetMol()


def _add_phosphate_between(
    mol: "Chem.RWMol",
    o3_idx_prev: int,
    o5_idx_next: int,
    linkage: str,
    stereo: str = "racemate",
    linkage_index: int | None = None,
    left_residue_index: int | None = None,
    right_residue_index: int | None = None,
) -> tuple[int, int]:
    """Add a phosphate (PO) or phosphorothioate (PS) atom and bond it
    between the previous nucleoside's 3'-O and the next nucleoside's 5'-O.

    For PS linkages with stereo in {"Rp", "Sp"}, sets the P atom's
    chirality tag so that RDKit's canonical SMILES emits [P@] / [P@@],
    which propagates into Chemprop's atom feature pipeline.

    Mutates `mol` in place. Returns the indices of the new P and pendant
    O/S atoms.
    """
    p_idx = mol.AddAtom(Chem.Atom("P"))
    if linkage == "PO":
        pendant_doublebond = mol.AddAtom(Chem.Atom("O"))
        pendant_single = mol.AddAtom(Chem.Atom("O"))
        pendant_single_h = 1
    elif linkage == "PS":
        pendant_doublebond = mol.AddAtom(Chem.Atom("S"))
        pendant_single = mol.AddAtom(Chem.Atom("O"))
        pendant_single_h = 1
    elif linkage == "MP":
        pendant_doublebond = mol.AddAtom(Chem.Atom("O"))
        pendant_single = mol.AddAtom(Chem.Atom("C"))
        pendant_single_h = 3
    elif linkage == "MOP":
        pendant_doublebond = mol.AddAtom(Chem.Atom("O"))
        pendant_single = mol.AddAtom(Chem.Atom("C"))
        c2 = mol.AddAtom(Chem.Atom("C"))
        c3 = mol.AddAtom(Chem.Atom("C"))
        o4 = mol.AddAtom(Chem.Atom("O"))
        c5 = mol.AddAtom(Chem.Atom("C"))
        pendant_single_h = 2
    elif linkage == "MSPA":
        pendant_doublebond = mol.AddAtom(Chem.Atom("O"))
        pendant_single = mol.AddAtom(Chem.Atom("N"))
        sulfonyl_s = mol.AddAtom(Chem.Atom("S"))
        sulfonyl_o1 = mol.AddAtom(Chem.Atom("O"))
        sulfonyl_o2 = mol.AddAtom(Chem.Atom("O"))
        mesyl_c = mol.AddAtom(Chem.Atom("C"))
        pendant_single_h = 1
    else:
        raise ValueError(f"Unknown linkage: {linkage!r}")

    linkage_atoms = [p_idx, pendant_doublebond, pendant_single]
    if linkage == "MOP":
        linkage_atoms.extend([c2, c3, o4, c5])
    elif linkage == "MSPA":
        linkage_atoms.extend([sulfonyl_s, sulfonyl_o1, sulfonyl_o2, mesyl_c])
    for atom_idx in linkage_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetProp("oligo_component", "linkage")
        atom.SetProp("oligo_linkage_code", linkage)
        if linkage_index is not None:
            atom.SetIntProp("oligo_linkage_index", int(linkage_index))
        if left_residue_index is not None:
            atom.SetIntProp("oligo_left_residue_index", int(left_residue_index))
        if right_residue_index is not None:
            atom.SetIntProp("oligo_right_residue_index", int(right_residue_index))

    # Bonds — ORDER MATTERS for chirality assignment because RDKit
    # interprets chirality based on neighbour order. We add neighbours
    # in fixed order: O3'(prev), pendant_double, pendant_single, O5'(next).
    mol.AddBond(p_idx, pendant_doublebond, Chem.BondType.DOUBLE)
    mol.AddBond(p_idx, pendant_single, Chem.BondType.SINGLE)
    mol.AddBond(o3_idx_prev, p_idx, Chem.BondType.SINGLE)
    mol.AddBond(p_idx, o5_idx_next, Chem.BondType.SINGLE)

    if linkage == "MOP":
        mol.AddBond(pendant_single, c2, Chem.BondType.SINGLE)
        mol.AddBond(c2, c3, Chem.BondType.SINGLE)
        mol.AddBond(c3, o4, Chem.BondType.SINGLE)
        mol.AddBond(o4, c5, Chem.BondType.SINGLE)
        mol.GetAtomWithIdx(c2).SetNumExplicitHs(2)
        mol.GetAtomWithIdx(c3).SetNumExplicitHs(2)
        mol.GetAtomWithIdx(c5).SetNumExplicitHs(3)
    elif linkage == "MSPA":
        mol.AddBond(pendant_single, sulfonyl_s, Chem.BondType.SINGLE)
        mol.AddBond(sulfonyl_s, sulfonyl_o1, Chem.BondType.DOUBLE)
        mol.AddBond(sulfonyl_s, sulfonyl_o2, Chem.BondType.DOUBLE)
        mol.AddBond(sulfonyl_s, mesyl_c, Chem.BondType.SINGLE)
        mol.GetAtomWithIdx(mesyl_c).SetNumExplicitHs(3)

    mol.GetAtomWithIdx(o3_idx_prev).SetNumExplicitHs(0)
    mol.GetAtomWithIdx(o5_idx_next).SetNumExplicitHs(0)
    mol.GetAtomWithIdx(pendant_single).SetNumExplicitHs(pendant_single_h)

    # Apply chirality tag if this is a stereo-defined PS bond
    if linkage == "PS" and stereo in ("Rp", "Sp"):
        atom = mol.GetAtomWithIdx(p_idx)
        # Bond order around P now: =S(pendant_doublebond), -OH(pendant_single),
        # bridging O3' (prev), bridging O5' (next)
        # We map Rp → CCW, Sp → CW (arbitrary convention; the point is
        # that Rp and Sp produce DIFFERENT chiral tags so SMILES + atom
        # features differ.) The CIP-correct mapping depends on element
        # priorities (S > O5' > O3' > O); we'll use a consistent rule.
        if stereo == "Rp":
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        else:  # Sp
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

    return p_idx, pendant_doublebond


def _find_attachment(mol: "Chem.Mol", map_num: int) -> int:
    """Return the atom index whose atom-map number == map_num."""
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom.GetIdx()
    raise KeyError(f"No atom with map number {map_num} in {Chem.MolToSmiles(mol)}")


def _build_oligonucleotide(monomers, linkages, stereo_list=None) -> "Chem.Mol":
    """Construct the full ASO Mol by combining nucleosides + phosphates.

    Args:
        monomers, linkages: as produced by parse_helm()
        stereo_list: optional list of per-bond stereo labels matching
            `linkages`. Each entry is "Rp", "Sp", or "racemate". When
            provided AND the corresponding linkage is "PS", the P atom
            gets a chirality tag → SMILES output differs.
    """
    if not monomers:
        raise ValueError("Empty monomer list")

    if stereo_list is None:
        stereo_list = ["racemate"] * len(linkages)
    if len(stereo_list) != len(linkages):
        raise ValueError(f"stereo_list length {len(stereo_list)} != "
                          f"linkages length {len(linkages)}")

    # Build first nucleoside
    first = _build_nucleoside_mol(*monomers[0], residue_index=1)
    combined = Chem.RWMol(first)
    o3_idx = _find_attachment(combined, 3)

    for i, link in enumerate(linkages):
        next_nuc = _build_nucleoside_mol(*monomers[i + 1], residue_index=i + 2)
        offset = combined.GetNumAtoms()
        merged = Chem.RWMol(Chem.CombineMols(combined, next_nuc))
        # The atom-map numbers carry through CombineMols; find them again.
        next_o5_idx = _find_attachment(merged, 5)
        # Need to find next_o5_idx that lives at offset+... not the prev one.
        # Search starting at `offset`:
        next_o5_idx = None
        for a in merged.GetAtoms():
            if a.GetIdx() >= offset and a.GetAtomMapNum() == 5:
                next_o5_idx = a.GetIdx()
                break
        if next_o5_idx is None:
            raise RuntimeError("Couldn't locate next 5'-O after merge")
        # Likewise, the previous 3'-O retains its map number 3, but there
        # are now two atoms with map 3 (prev nucleoside + new one). We
        # want the FIRST one (the previous 3'-end). Re-find it:
        first_o3 = None
        for a in merged.GetAtoms():
            if a.GetIdx() < offset and a.GetAtomMapNum() == 3:
                first_o3 = a.GetIdx()
                break
        if first_o3 is None:
            raise RuntimeError("Couldn't locate previous 3'-O after merge")

        # Add the phosphate between prev 3'-O and next 5'-O.
        _add_phosphate_between(
            merged,
            first_o3,
            next_o5_idx,
            link,
            stereo=stereo_list[i],
            linkage_index=i + 1,
            left_residue_index=i + 1,
            right_residue_index=i + 2,
        )

        # Clear atom-map labels we just used so they don't pollute later joins.
        merged.GetAtomWithIdx(first_o3).SetAtomMapNum(0)
        merged.GetAtomWithIdx(next_o5_idx).SetAtomMapNum(0)

        combined = merged

    # Clear remaining attachment labels (the 5'-O of the first nucleoside
    # and the 3'-O of the last nucleoside both stay as free -OH groups).
    for atom in combined.GetAtoms():
        atom.SetAtomMapNum(0)

    mol = combined.GetMol()
    Chem.SanitizeMol(mol)
    return mol


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

def helm_to_mol(helm: str) -> "Chem.Mol":
    """Convert HELM RNA notation into an RDKit Mol."""
    monomers, linkages = parse_helm(helm)
    return _build_oligonucleotide(monomers, linkages)


def helm_to_smiles(helm: str, canonical: bool = True) -> str:
    """Convert HELM RNA notation into a canonical SMILES of the full ASO."""
    mol = helm_to_mol(helm)
    return Chem.MolToSmiles(mol, canonical=canonical)


def helm_to_mol_with_stereo(helm: str, stereo_list: list) -> "Chem.Mol":
    """Build the Mol with per-PS-bond stereo labels applied.

    Args:
        helm: standard HELM string
        stereo_list: list of strings, one per internucleotide bond,
            each in {"Rp", "Sp", "racemate", "none"}. Only "Rp" and
            "Sp" cause a stereo tag to be set on the corresponding
            P atom; "racemate" and "none" leave it untagged.
    """
    monomers, linkages = parse_helm(helm)
    if len(stereo_list) != len(linkages):
        raise ValueError(
            f"stereo_list length {len(stereo_list)} does not match "
            f"linkages length {len(linkages)} (HELM has {len(linkages)} bonds)"
        )
    return _build_oligonucleotide(monomers, linkages, stereo_list=stereo_list)


def helm_to_smiles_with_stereo(helm: str, stereo_list: list,
                                 canonical: bool = True) -> str:
    """Convert HELM + per-bond stereo labels into canonical SMILES with
    P chirality flags ([P@] / [P@@]) where the stereo is defined.

    For positions tagged "Rp" or "Sp", the SMILES output includes a
    chirality flag at that P atom. For "racemate" or "none" positions,
    no flag is emitted — the SMILES output for those positions matches
    plain helm_to_smiles().

    This means:
        helm_to_smiles_with_stereo(helm, ["racemate"] * n) == helm_to_smiles(helm)
    """
    mol = helm_to_mol_with_stereo(helm, stereo_list)
    return Chem.MolToSmiles(mol, canonical=canonical)


def helm_to_mol_with_residues(helm: str):
    """Build the ASO Mol and return (mol_canonical, residue_per_atom) so
    callers can group atoms by residue (nucleotide index).

    Convention: residue i contains the i-th nucleoside (sugar+base) plus
    the *trailing* linker atoms (phosphate group connecting residue i to
    residue i+1). Residue N-1 has no trailing linker.

    Returns:
        mol_canonical: the molecule as produced by Chem.MolFromSmiles on
            the canonical SMILES (same object compute_bond_attributions
            sees), so atom indices line up with bond_attribution outputs.
        residue_per_atom: list[int] of length mol_canonical.GetNumAtoms()
            mapping each canonical atom to its residue index (0..N-1).
    """
    monomers, linkages = parse_helm(helm)
    if not monomers:
        raise ValueError("Empty monomer list")

    # Build with residue tagging via custom int property "_residx"
    first = _build_nucleoside_mol(*monomers[0])
    combined = Chem.RWMol(first)
    for a in combined.GetAtoms():
        a.SetIntProp("_residx", 0)

    for i, link in enumerate(linkages):
        next_nuc = _build_nucleoside_mol(*monomers[i + 1])
        offset = combined.GetNumAtoms()
        merged = Chem.RWMol(Chem.CombineMols(combined, next_nuc))
        # tag the newly-added nucleoside atoms with residue i+1
        for a in merged.GetAtoms():
            if a.GetIdx() >= offset:
                a.SetIntProp("_residx", i + 1)

        next_o5_idx = None
        for a in merged.GetAtoms():
            if a.GetIdx() >= offset and a.GetAtomMapNum() == 5:
                next_o5_idx = a.GetIdx()
                break
        first_o3 = None
        for a in merged.GetAtoms():
            if a.GetIdx() < offset and a.GetAtomMapNum() == 3:
                first_o3 = a.GetIdx()
                break
        if next_o5_idx is None or first_o3 is None:
            raise RuntimeError("Couldn't locate attachment Os after merge")

        n_before_phos = merged.GetNumAtoms()
        _add_phosphate_between(merged, first_o3, next_o5_idx, link)
        # Tag phosphate atoms with residue i (the LINKER trails residue i)
        for a in merged.GetAtoms():
            if a.GetIdx() >= n_before_phos:
                a.SetIntProp("_residx", i)

        merged.GetAtomWithIdx(first_o3).SetAtomMapNum(0)
        merged.GetAtomWithIdx(next_o5_idx).SetAtomMapNum(0)
        combined = merged

    for atom in combined.GetAtoms():
        atom.SetAtomMapNum(0)
    mol_built = combined.GetMol()
    Chem.SanitizeMol(mol_built)

    # Get residue assignment in build-order then map to canonical SMILES order
    residue_built = [int(a.GetProp("_residx")) for a in mol_built.GetAtoms()]
    canonical_smi = Chem.MolToSmiles(mol_built, canonical=True)
    mol_canonical = Chem.MolFromSmiles(canonical_smi)
    # match[i] = canonical_atom_idx for built atom i
    match = mol_canonical.GetSubstructMatch(mol_built)
    if len(match) != mol_built.GetNumAtoms():
        # Fall back to identity if substructure match degenerate
        return mol_canonical, residue_built
    residue_canonical = [0] * mol_canonical.GetNumAtoms()
    for built_i, canon_i in enumerate(match):
        residue_canonical[canon_i] = residue_built[built_i]
    return mol_canonical, residue_canonical


def count_atoms(helm: str, include_hydrogens: bool = False) -> int:
    """Return the heavy-atom count (or all-atom count if requested)."""
    mol = helm_to_mol(helm)
    if include_hydrogens:
        mol = Chem.AddHs(mol)
        return mol.GetNumAtoms()
    return mol.GetNumHeavyAtoms()
