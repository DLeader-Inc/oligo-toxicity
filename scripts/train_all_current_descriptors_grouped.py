#!/usr/bin/env python
"""Train grouped-CV models with all currently implemented descriptor blocks.

This is a fixed all-descriptor model, not a recipe selector. It combines:
baseline sequence/chemistry, sequence-chemistry interaction, position encoding,
secondary-structure descriptors, PPB/rule descriptors, explicit OpenBabel 3D
descriptors, cached Uni-Mol3D embeddings, cached Protenix sequence embeddings,
Protenix sequence-only CIF descriptors, Protenix custom-CCD compact embeddings,
and Protenix custom-CCD CIF descriptors when caches are available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_structure_descriptor_ablation import _numeric_secondary, _sanitize_feature_frame
from src.data.datasets import load_oligogym, get_dataset
from src.features.featurizer import featurize_dataset
from src.features.position_encoding_features import dataframe_for_helms as position_for_helms
from src.features.ppb_rule_features import dataframe_for_helms as ppb_rules_for_helms
from src.features.protenix_customccd_features import (
    descriptor_matrix_for_helms as customccd_cif_matrix,
    load_cif_descriptor_cache as load_customccd_cif_descriptors,
    load_compact_embedding_cache as load_customccd_embedding_cache,
    matrix_for_helms as customccd_embedding_matrix,
)
from src.features.protenix_embedding_features import (
    discover_feature_cache as discover_protenix_embedding_cache,
    matrix_for_sequences as protenix_embedding_matrix,
)
from src.features.protenix_sequence_cif_features import (
    descriptor_matrix_for_sequences as sequence_cif_matrix,
    load_sequence_cif_descriptor_cache,
)
from src.features.secondary_structure_features import dataframe_for_helms as secondary2d_for_helms
from src.features.sequence_chemistry_interaction_features import dataframe_for_helms as seqchem_for_helms
from src.features.structure_descriptor_features import numeric_descriptor_frame
from src.features.unimol_3d_embedding_features import (
    feature_names as unimol_feature_names,
    load_feature_cache as load_unimol_cache,
    matrix_for_helms as unimol_matrix,
)


DEFAULT_DATASETS = [
    "cytotoxicity",
    "neurotoxicity_lna",
    "neurotoxicity_moe",
    "tlr7_immunotox",
    "tlr8_immunotox",
]
TARGET_ELIGIBLE = {"neurotoxicity_moe", "tlr7_immunotox", "tlr8_immunotox"}


def _metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    sp, _ = stats.spearmanr(y, pred)
    pe, _ = stats.pearsonr(y, pred)
    return {
        "spearman": float(sp),
        "pearson": float(pe),
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }


def _model(seed: int, n_estimators: int, max_depth: int, learning_rate: float) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        verbosity=0,
        n_jobs=-1,
    )


def _groups(parsed: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "sequence":
        return parsed["bases"].astype(str).reset_index(drop=True)
    if mode == "target":
        return parsed["targets"].astype(str).reset_index(drop=True)
    raise ValueError(mode)


def _run_grouped(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    seed: int,
    folds: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> dict[str, float]:
    X = _sanitize_feature_frame(X).reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    n_splits = min(folds, int(groups.nunique()))
    splitter = GroupKFold(n_splits=n_splits)
    preds = np.zeros(len(y), dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = _model(seed + fold_idx, n_estimators, max_depth, learning_rate)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X.iloc[test_idx])
    out = _metric(y, preds)
    out["n_splits"] = float(n_splits)
    return out


def _align_structure3d(parsed: pd.DataFrame, cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cache_path.exists():
        missing = pd.DataFrame({"sd3d_missing": np.ones(len(parsed), dtype=float)}, index=parsed.index)
        return missing, pd.DataFrame(
            {"structure3d_cache_hit": np.zeros(len(parsed), dtype=int), "structure3d_ok": np.zeros(len(parsed), dtype=int)}
        )
    cache = pd.read_csv(cache_path)
    merged = pd.DataFrame({"helm": parsed["x"].astype(str)}).merge(cache, on="helm", how="left")
    status_col = next((col for col in merged.columns if col.endswith("_status")), None)
    meta = pd.DataFrame(
        {
            "structure3d_cache_hit": merged[status_col].notna().astype(int) if status_col else 0,
            "structure3d_ok": (merged[status_col] == "ok").astype(int) if status_col else 0,
        }
    )
    X = numeric_descriptor_frame(merged.drop(columns=["helm"], errors="ignore"))
    X["sd3d_missing"] = 1.0 - meta["structure3d_ok"].to_numpy(dtype=float)
    return X.reset_index(drop=True), meta


def _prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out.columns = [f"{prefix}__{col}" for col in out.columns]
    return out


def _build_all_features(
    parsed: pd.DataFrame,
    X_base: pd.DataFrame,
    args: argparse.Namespace,
    caches: dict[str, object],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    helms = parsed["x"].astype(str)
    seqs = parsed["bases"].astype(str)
    blocks: list[tuple[str, pd.DataFrame]] = []
    coverage: list[dict[str, object]] = []

    blocks.append(("baseline", X_base.reset_index(drop=True)))
    coverage.append({"block": "baseline", "n_features": X_base.shape[1], "n_hits": len(parsed), "n_rows": len(parsed)})

    X_seqchem = seqchem_for_helms(helms, max_pos=args.max_pos)
    blocks.append(("seqchem", X_seqchem))
    coverage.append({"block": "seqchem", "n_features": X_seqchem.shape[1], "n_hits": len(parsed), "n_rows": len(parsed)})

    X_position = position_for_helms(helms, max_pos=args.max_pos)
    blocks.append(("position", X_position))
    coverage.append({"block": "position", "n_features": X_position.shape[1], "n_hits": len(parsed), "n_rows": len(parsed)})

    X_2d = _numeric_secondary(secondary2d_for_helms(helms, max_pos=args.max_pos, with_pair_probs=not args.no_pair_probs))
    blocks.append(("secondary2d", X_2d))
    coverage.append({"block": "secondary2d", "n_features": X_2d.shape[1], "n_hits": len(parsed), "n_rows": len(parsed)})

    X_ppb = ppb_rules_for_helms(helms, sequences=seqs)
    blocks.append(("ppb_rules", X_ppb))
    coverage.append({"block": "ppb_rules", "n_features": X_ppb.shape[1], "n_hits": len(parsed), "n_rows": len(parsed)})

    X_3d, meta3d = _align_structure3d(parsed, Path(args.structure3d_cache))
    if X_3d.shape[1] > 0:
        blocks.append(("openbabel3d_desc", X_3d))
    coverage.append(
        {
            "block": "openbabel3d_desc",
            "n_features": X_3d.shape[1],
            "n_hits": int(meta3d["structure3d_ok"].sum()),
            "n_rows": len(parsed),
        }
    )

    uni_X, uni_meta = unimol_matrix(helms, caches["unimol"])
    if uni_X.shape[1] > 1:
        X_uni = pd.DataFrame(uni_X, columns=unimol_feature_names(uni_X.shape[1] - 1))
        blocks.append(("unimol3d_embedding", X_uni))
    coverage.append(
        {
            "block": "unimol3d_embedding",
            "n_features": int(uni_X.shape[1]),
            "n_hits": int(uni_meta["unimol3d_cache_hit"].sum()),
            "n_rows": len(parsed),
        }
    )

    prot_X, prot_meta = protenix_embedding_matrix(seqs, caches["protenix_embedding"])
    X_prot = pd.DataFrame(prot_X)
    blocks.append(("protenix_sequence_embedding", X_prot))
    coverage.append(
        {
            "block": "protenix_sequence_embedding",
            "n_features": X_prot.shape[1],
            "n_hits": int(prot_meta["protenix_cache_hit"].sum()),
            "n_rows": len(parsed),
        }
    )

    X_seq_cif, seq_cif_meta = sequence_cif_matrix(seqs, caches["sequence_cif"])
    blocks.append(("protenix_sequence_cif_desc", X_seq_cif))
    coverage.append(
        {
            "block": "protenix_sequence_cif_desc",
            "n_features": X_seq_cif.shape[1],
            "n_hits": int(seq_cif_meta["protenix_sequence_cif_ok"].sum()),
            "n_rows": len(parsed),
        }
    )

    custom_emb_X, custom_emb_meta = customccd_embedding_matrix(helms, caches["customccd_embedding"])
    X_custom_emb = pd.DataFrame(custom_emb_X)
    blocks.append(("protenix_customccd_embedding", X_custom_emb))
    coverage.append(
        {
            "block": "protenix_customccd_embedding",
            "n_features": X_custom_emb.shape[1],
            "n_hits": int(custom_emb_meta["protenix_customccd_cache_hit"].sum()),
            "n_rows": len(parsed),
        }
    )

    X_custom_cif, custom_cif_meta = customccd_cif_matrix(helms, caches["customccd_cif"])
    blocks.append(("protenix_customccd_cif_desc", X_custom_cif))
    coverage.append(
        {
            "block": "protenix_customccd_cif_desc",
            "n_features": X_custom_cif.shape[1],
            "n_hits": int(custom_cif_meta["protenix_customccd_cif_ok"].sum()),
            "n_rows": len(parsed),
        }
    )

    X_all = pd.concat([_prefix(frame, name) for name, frame in blocks], axis=1)
    X_all = _sanitize_feature_frame(X_all)
    return X_all, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/all_current_descriptors_grouped"))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-pos", type=int, default=30)
    parser.add_argument("--no-pair-probs", action="store_true")
    parser.add_argument("--structure3d-cache", default=str(ROOT / "outputs/structure_descriptor_ablation_openbabel_full/structure3d_descriptors.csv"))
    parser.add_argument("--unimol-cache-dir", default=str(ROOT / "outputs/unimol3d_full_cache_openbabel_cuda"))
    parser.add_argument("--protenix-cache-dir", default=str(ROOT / "outputs/protenix_full_cache"))
    parser.add_argument("--customccd-protenix-cache-dir", default=str(ROOT / "outputs/protenix_customccd_full_cache"))
    parser.add_argument("--protenix-sequence-cif-archive-dir", default=str(ROOT / "outputs/protenix_structure_archive"))
    parser.add_argument("--protenix-sequence-cif-extra-npz-dir", default=str(ROOT / "outputs/protenix_plain_sequence_structure_cache"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caches = {
        "unimol": load_unimol_cache(Path(args.unimol_cache_dir)),
        "protenix_embedding": discover_protenix_embedding_cache(ROOT, extra_dirs=[Path(args.protenix_cache_dir)]),
        "sequence_cif": load_sequence_cif_descriptor_cache(
            Path(args.protenix_sequence_cif_archive_dir),
            extra_npz_dir=Path(args.protenix_sequence_cif_extra_npz_dir),
        ),
        "customccd_embedding": load_customccd_embedding_cache(Path(args.customccd_protenix_cache_dir)),
        "customccd_cif": load_customccd_cif_descriptors(Path(args.customccd_protenix_cache_dir)),
    }

    metric_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for dataset_key in args.datasets:
        ds = get_dataset(dataset_key)
        df = load_oligogym(ds.oligogym_key)
        X_base, parsed = featurize_dataset(df)
        y = parsed["y"].to_numpy()
        print(f"{dataset_key}: building all descriptors", flush=True)
        X_all, coverage = _build_all_features(parsed, X_base, args, caches)
        feature_rows.append({"dataset": dataset_key, "rows": len(parsed), "all_descriptor_features": X_all.shape[1]})
        for row in coverage:
            row = dict(row)
            row["dataset"] = dataset_key
            coverage_rows.append(row)
        modes = ["sequence"] + (["target"] if dataset_key in TARGET_ELIGIBLE else [])
        for mode in modes:
            groups = _groups(parsed, mode)
            metrics = _run_grouped(
                X_all,
                y,
                groups,
                args.seed,
                args.folds,
                args.n_estimators,
                args.max_depth,
                args.learning_rate,
            )
            metric_rows.append(
                {
                    "dataset": dataset_key,
                    "validation": mode,
                    "rows": len(parsed),
                    "groups": int(groups.nunique()),
                    "n_features": int(X_all.shape[1]),
                    **metrics,
                }
            )
            pd.DataFrame(metric_rows).to_csv(out_dir / "all_current_descriptors_metrics_partial.csv", index=False)

    metrics_df = pd.DataFrame(metric_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    feature_df = pd.DataFrame(feature_rows)
    metrics_df.to_csv(out_dir / "all_current_descriptors_metrics.csv", index=False)
    coverage_df.to_csv(out_dir / "all_current_descriptors_coverage.csv", index=False)
    feature_df.to_csv(out_dir / "all_current_descriptors_feature_counts.csv", index=False)
    lines = [
        "# All Current Descriptors Grouped Validation",
        "",
        "Fixed model with all currently implemented descriptor blocks concatenated.",
        "",
        "## Metrics",
        "",
        metrics_df.round(4).to_markdown(index=False),
        "",
        "## Coverage",
        "",
        coverage_df.to_markdown(index=False),
        "",
    ]
    (out_dir / "ALL_CURRENT_DESCRIPTORS_GROUPED.md").write_text("\n".join(lines), encoding="utf-8")
    print(metrics_df.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
