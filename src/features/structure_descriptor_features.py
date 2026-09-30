"""Explicit 2D/3D structural descriptors for modified oligonucleotides.

These features are intentionally model-agnostic: HELM is converted to a full
atomistic SMILES, a conformer is generated with Open Babel by default, then
shape/distance/contact descriptors are computed from the resulting coordinates.
RDKit ETKDG is retained only as a small-molecule/debug fallback because it was
empirically unreliable for full-length modified ASOs.
"""

from __future__ import annotations

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Descriptors3D, rdMolDescriptors

from src.features.helm_to_smiles import helm_to_smiles


CONTACT_THRESHOLDS = (4.0, 6.0, 8.0, 10.0)
USR_LEN = 12
USRCAT_LEN = 60


def _empty_record(status: str, prefix: str = "sd3d") -> dict[str, float | str]:
    record: dict[str, float | str] = {
        f"{prefix}_status": status,
        f"{prefix}_n_atoms": 0.0,
        f"{prefix}_n_heavy_atoms": 0.0,
        f"{prefix}_mol_wt": 0.0,
        f"{prefix}_logp": 0.0,
        f"{prefix}_tpsa": 0.0,
        f"{prefix}_rotatable_bonds": 0.0,
        f"{prefix}_ring_count": 0.0,
        f"{prefix}_hetero_atom_fraction": 0.0,
        f"{prefix}_radius_gyration": 0.0,
        f"{prefix}_asphericity": 0.0,
        f"{prefix}_eccentricity": 0.0,
        f"{prefix}_inertial_shape_factor": 0.0,
        f"{prefix}_npr1": 0.0,
        f"{prefix}_npr2": 0.0,
        f"{prefix}_pmi1": 0.0,
        f"{prefix}_pmi2": 0.0,
        f"{prefix}_pmi3": 0.0,
        f"{prefix}_spherocity_index": 0.0,
        f"{prefix}_distance_mean": 0.0,
        f"{prefix}_distance_std": 0.0,
        f"{prefix}_distance_q10": 0.0,
        f"{prefix}_distance_q25": 0.0,
        f"{prefix}_distance_q50": 0.0,
        f"{prefix}_distance_q75": 0.0,
        f"{prefix}_distance_q90": 0.0,
        f"{prefix}_distance_max": 0.0,
    }
    for threshold in CONTACT_THRESHOLDS:
        tag = str(threshold).replace(".", "p")
        record[f"{prefix}_contact_count_{tag}A"] = 0.0
        record[f"{prefix}_contact_density_{tag}A"] = 0.0
    for idx in range(USR_LEN):
        record[f"{prefix}_usr_{idx:02d}"] = 0.0
    for idx in range(USRCAT_LEN):
        record[f"{prefix}_usrcat_{idx:02d}"] = 0.0
    return record


def _embed_rdkit(
    smiles: str,
    seed: int = 42,
    timeout_sec: int = 30,
    save_sdf_path: str | Path | None = None,
) -> Chem.Mol:
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        raise ValueError("RDKit MolFromSmiles failed")
    mol = Chem.AddHs(mol0)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    params.maxIterations = 500
    if timeout_sec > 0:
        params.timeout = int(timeout_sec)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise TimeoutError("RDKit ETKDG embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=100)
    if save_sdf_path is not None:
        out_path = Path(save_sdf_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(out_path))
        writer.write(mol)
        writer.close()
    return mol


def _embed_openbabel(
    smiles: str,
    timeout_sec: int = 0,
    save_sdf_path: str | Path | None = None,
) -> Chem.Mol:
    with tempfile.TemporaryDirectory(prefix="obabel_struct_desc_") as tmp:
        tmp_path = Path(tmp)
        smi_path = tmp_path / "input.smi"
        sdf_path = tmp_path / "output.sdf"
        smi_path.write_text(f"{smiles}\tquery\n", encoding="utf-8")
        subprocess.run(
            ["obabel", "-ismi", str(smi_path), "-osdf", "-O", str(sdf_path), "--gen3d"],
            capture_output=True,
            text=True,
            timeout=None if timeout_sec <= 0 else timeout_sec,
            check=True,
        )
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = next((m for m in suppl if m is not None), None)
        if mol is None or mol.GetNumConformers() == 0:
            raise RuntimeError("Open Babel did not produce a readable conformer")
        if save_sdf_path is not None:
            out_path = Path(save_sdf_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sdf_path, out_path)
        return mol


def _distance_values(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    coords = np.asarray(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
        dtype=float,
    )
    if len(coords) < 2:
        return np.asarray([], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    tri = np.triu_indices(len(coords), k=1)
    return dist[tri]


def descriptors_from_conformer(mol: Chem.Mol, prefix: str = "sd3d") -> dict[str, float | str]:
    """Compute scalar, contact, and USR descriptors from a molecule conformer."""
    record = _empty_record("ok", prefix=prefix)
    heavy = Chem.RemoveHs(mol)
    n_atoms = float(mol.GetNumAtoms())
    n_heavy = float(heavy.GetNumAtoms())
    hetero = sum(1 for atom in heavy.GetAtoms() if atom.GetAtomicNum() not in (1, 6))

    record.update(
        {
            f"{prefix}_n_atoms": n_atoms,
            f"{prefix}_n_heavy_atoms": n_heavy,
            f"{prefix}_mol_wt": float(Descriptors.MolWt(heavy)),
            f"{prefix}_logp": float(Crippen.MolLogP(heavy)),
            f"{prefix}_tpsa": float(rdMolDescriptors.CalcTPSA(heavy)),
            f"{prefix}_rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(heavy)),
            f"{prefix}_ring_count": float(rdMolDescriptors.CalcNumRings(heavy)),
            f"{prefix}_hetero_atom_fraction": float(hetero / max(n_heavy, 1.0)),
        }
    )

    try:
        record.update(
            {
                f"{prefix}_radius_gyration": float(Descriptors3D.RadiusOfGyration(mol)),
                f"{prefix}_asphericity": float(Descriptors3D.Asphericity(mol)),
                f"{prefix}_eccentricity": float(Descriptors3D.Eccentricity(mol)),
                f"{prefix}_inertial_shape_factor": float(Descriptors3D.InertialShapeFactor(mol)),
                f"{prefix}_npr1": float(Descriptors3D.NPR1(mol)),
                f"{prefix}_npr2": float(Descriptors3D.NPR2(mol)),
                f"{prefix}_pmi1": float(Descriptors3D.PMI1(mol)),
                f"{prefix}_pmi2": float(Descriptors3D.PMI2(mol)),
                f"{prefix}_pmi3": float(Descriptors3D.PMI3(mol)),
                f"{prefix}_spherocity_index": float(Descriptors3D.SpherocityIndex(mol)),
            }
        )
    except Exception:
        record[f"{prefix}_status"] = "partial_3d_descriptor_error"

    distances = _distance_values(mol)
    if len(distances):
        record.update(
            {
                f"{prefix}_distance_mean": float(np.mean(distances)),
                f"{prefix}_distance_std": float(np.std(distances)),
                f"{prefix}_distance_q10": float(np.quantile(distances, 0.10)),
                f"{prefix}_distance_q25": float(np.quantile(distances, 0.25)),
                f"{prefix}_distance_q50": float(np.quantile(distances, 0.50)),
                f"{prefix}_distance_q75": float(np.quantile(distances, 0.75)),
                f"{prefix}_distance_q90": float(np.quantile(distances, 0.90)),
                f"{prefix}_distance_max": float(np.max(distances)),
            }
        )
        denom = max(len(distances), 1)
        for threshold in CONTACT_THRESHOLDS:
            tag = str(threshold).replace(".", "p")
            count = float(np.sum(distances <= threshold))
            record[f"{prefix}_contact_count_{tag}A"] = count
            record[f"{prefix}_contact_density_{tag}A"] = count / denom

    try:
        usr = rdMolDescriptors.GetUSR(mol)
        for idx, value in enumerate(usr[:USR_LEN]):
            record[f"{prefix}_usr_{idx:02d}"] = float(value)
    except Exception:
        pass

    try:
        usrcat = rdMolDescriptors.GetUSRCAT(mol)
        for idx, value in enumerate(usrcat[:USRCAT_LEN]):
            record[f"{prefix}_usrcat_{idx:02d}"] = float(value)
    except Exception:
        pass

    return record


def descriptors_for_smiles(
    smiles: str,
    backend: str = "openbabel",
    seed: int = 42,
    timeout_sec: int = 30,
    prefix: str = "sd3d",
    save_sdf_path: str | Path | None = None,
) -> dict[str, float | str]:
    """Generate a conformer and compute explicit 3D descriptors for SMILES.

    Use ``backend="openbabel"`` for full-length modified ASOs. RDKit ETKDG is
    available for smoke tests but can hang/fail on flexible 20mer gapmers.
    """
    try:
        if save_sdf_path is not None:
            existing = Path(save_sdf_path)
            if existing.exists() and existing.stat().st_size > 0:
                suppl = Chem.SDMolSupplier(str(existing), removeHs=False)
                mol = next((m for m in suppl if m is not None), None)
                if mol is not None and mol.GetNumConformers() > 0:
                    record = descriptors_from_conformer(mol, prefix=prefix)
                    record[f"{prefix}_conformer_sdf"] = str(existing)
                    record[f"{prefix}_status"] = "ok"
                    return record
        if backend == "rdkit":
            mol = _embed_rdkit(smiles, seed=seed, timeout_sec=timeout_sec, save_sdf_path=save_sdf_path)
        elif backend == "openbabel":
            mol = _embed_openbabel(smiles, timeout_sec=timeout_sec, save_sdf_path=save_sdf_path)
        else:
            raise ValueError(f"Unknown conformer backend: {backend}")
        record = descriptors_from_conformer(mol, prefix=prefix)
        if save_sdf_path is not None and Path(save_sdf_path).exists():
            record[f"{prefix}_conformer_sdf"] = str(Path(save_sdf_path))
        return record
    except Exception as exc:
        record = _empty_record("failed", prefix=prefix)
        record[f"{prefix}_error"] = f"{type(exc).__name__}: {exc}"
        return record


def descriptors_for_helm(
    helm: str,
    backend: str = "openbabel",
    seed: int = 42,
    timeout_sec: int = 30,
    prefix: str = "sd3d",
    save_sdf_path: str | Path | None = None,
) -> dict[str, float | str]:
    """Convert HELM to atomistic SMILES, then compute explicit 3D descriptors."""
    try:
        smiles = helm_to_smiles(helm)
    except Exception as exc:
        record = _empty_record("failed_helm_to_smiles", prefix=prefix)
        record[f"{prefix}_error"] = f"{type(exc).__name__}: {exc}"
        return record
    return descriptors_for_smiles(
        smiles,
        backend=backend,
        seed=seed,
        timeout_sec=timeout_sec,
        prefix=prefix,
        save_sdf_path=save_sdf_path,
    )


def dataframe_for_helms(
    helms: Iterable[str],
    backend: str = "openbabel",
    seed: int = 42,
    timeout_sec: int = 30,
    prefix: str = "sd3d",
    save_sdf_paths: Iterable[str | Path | None] | None = None,
) -> pd.DataFrame:
    """Compute 3D descriptors for HELM strings and return one row per input."""
    helms = list(helms)
    paths = list(save_sdf_paths) if save_sdf_paths is not None else [None] * len(helms)
    rows = [
        descriptors_for_helm(
            str(helm),
            backend=backend,
            seed=seed,
            timeout_sec=timeout_sec,
            prefix=prefix,
            save_sdf_path=path,
        )
        for helm, path in zip(helms, paths, strict=True)
    ]
    return pd.DataFrame(rows).fillna(0)


def numeric_descriptor_frame(df: pd.DataFrame, prefix: str = "sd3d") -> pd.DataFrame:
    """Return numeric molecular descriptors from one descriptor namespace.

    Cache tables also contain paths, hashes, dataset labels, and extraction
    timing. Coercing those columns to numbers silently turns them into
    constant-zero pseudo-features, while timing can encode the extraction
    environment rather than molecular structure. Keep only descriptor-prefixed
    values and explicitly remove status/error/path metadata.
    """
    descriptor_cols = [
        col
        for col in df.columns
        if col.startswith(f"{prefix}_")
        and not col.startswith(f"{prefix}_status")
        and not col.startswith(f"{prefix}_error")
        and not col.endswith("_conformer_sdf")
    ]
    numeric = df[descriptor_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    return numeric.fillna(0.0)
