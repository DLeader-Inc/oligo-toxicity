#!/usr/bin/env python
"""Grouped CV for current descriptor families.

Each family is trained as `baseline + family-specific descriptors` using its
own XGBoost model. The ensemble is a fixed unweighted combination of family
predictions; it does not concatenate all descriptors into one giant model.
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

from scripts.train_all_current_descriptors_grouped import _align_structure3d
from scripts.train_structure_descriptor_ablation import _numeric_secondary, _sanitize_feature_frame
from src.data.datasets import get_dataset, load_oligogym
from src.data.grouping import canonical_sequence_groups
from src.features.featurizer import featurize_dataset
from src.features.position_encoding_features import dataframe_for_helms as position_for_helms
from src.features.position_reencoding_features import (
    dataframes_for_helms as position_reencoded_for_helms,
    token_ohe_dataframe_for_helms,
)
from src.features.ppb_position_features import dataframe_for_helms as ppb_position_for_helms
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


def _groups(
    parsed: pd.DataFrame,
    mode: str,
    *,
    dataset_key: str | None = None,
    similarity_cluster_file: Path | None = None,
    similarity_threshold: float | None = None,
) -> pd.Series:
    if mode == "sequence":
        return canonical_sequence_groups(parsed, dataset_key=dataset_key or "unknown")
    if mode == "target":
        return parsed["targets"].astype(str).reset_index(drop=True)
    if mode == "similarity":
        if dataset_key is None or similarity_cluster_file is None or similarity_threshold is None:
            raise ValueError("similarity grouping requires dataset_key, similarity_cluster_file, and similarity_threshold")
        clusters = pd.read_csv(similarity_cluster_file)
        clusters = clusters[
            clusters["dataset"].astype(str).eq(str(dataset_key))
            & clusters["threshold"].astype(float).eq(float(similarity_threshold))
        ].copy()
        if clusters.empty:
            raise ValueError(f"No similarity clusters for dataset={dataset_key!r}, threshold={similarity_threshold}")
        cluster_map = clusters.drop_duplicates("sequence").set_index("sequence")["cluster_id"].astype(str).to_dict()
        groups = parsed["bases"].astype(str).map(cluster_map)
        missing = int(groups.isna().sum())
        if missing:
            raise ValueError(f"Similarity cluster map missing {missing}/{len(parsed)} sequences for {dataset_key}")
        return groups.reset_index(drop=True)
    raise ValueError(mode)


def _prefixed(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out.columns = [f"{prefix}__{col}" for col in out.columns]
    return out


def _family_frame(X_base: pd.DataFrame, X_desc: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if X_desc is None or X_desc.shape[1] == 0:
        return _prefixed(X_base, "baseline")
    return pd.concat([_prefixed(X_base, "baseline"), _prefixed(X_desc, name)], axis=1)


def _build_families(
    parsed: pd.DataFrame,
    X_base: pd.DataFrame,
    args: argparse.Namespace,
    caches: dict[str, object],
    dataset_key: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    helms = parsed["x"].astype(str)
    seqs = parsed["bases"].astype(str)
    selected_families = set(args.families or [])

    def wants(name: str) -> bool:
        return not selected_families or name in selected_families or name == "core_baseline"

    raw_blocks: dict[str, pd.DataFrame | None] = {"core_baseline": None}
    coverage_rows: list[dict[str, object]] = [
        {"family": "core_baseline", "n_features_added": 0, "n_hits": len(parsed), "n_rows": len(parsed)}
    ]

    def add_block(name: str, frame: pd.DataFrame, hits: int | None = None) -> None:
        raw_blocks[name] = frame.reset_index(drop=True)
        coverage_rows.append(
            {
                "family": name,
                "n_features_added": int(frame.shape[1]),
                "n_hits": int(len(parsed) if hits is None else hits),
                "n_rows": len(parsed),
            }
        )

    if wants("seqchem"):
        add_block("seqchem", seqchem_for_helms(helms, max_pos=args.max_pos))
    if wants("position"):
        add_block("position", position_for_helms(helms, max_pos=args.max_pos))
    reencoded_names = {
        "position_token_ohe",
        "position_token_ohe_leftpad",
        "position_token_ohe_centerpad",
        "position_relative",
        "position_interactions",
        "position_token_relative",
        "position_reencoded_all",
    }
    if any(wants(name) for name in reencoded_names):
        reencoded = position_reencoded_for_helms(helms, max_pos=args.max_pos)
        if wants("position_token_ohe"):
            add_block("position_token_ohe", reencoded["position_token_ohe"])
        if wants("position_token_ohe_leftpad"):
            add_block(
                "position_token_ohe_leftpad",
                token_ohe_dataframe_for_helms(
                    helms, max_pos=args.max_pos, alignment="left"
                ),
            )
        if wants("position_token_ohe_centerpad"):
            add_block(
                "position_token_ohe_centerpad",
                token_ohe_dataframe_for_helms(
                    helms, max_pos=args.max_pos, alignment="center"
                ),
            )
        if wants("position_relative"):
            add_block("position_relative", reencoded["position_relative"])
        if wants("position_interactions"):
            add_block("position_interactions", reencoded["position_interactions"])
        if wants("position_token_relative"):
            add_block(
                "position_token_relative",
                pd.concat(
                    [reencoded["position_token_ohe"], reencoded["position_relative"]],
                    axis=1,
                ),
            )
        if wants("position_reencoded_all"):
            add_block("position_reencoded_all", pd.concat(list(reencoded.values()), axis=1))
    if wants("secondary2d"):
        add_block(
            "secondary2d",
            _numeric_secondary(secondary2d_for_helms(helms, max_pos=args.max_pos, with_pair_probs=not args.no_pair_probs)),
        )
    if wants("ppb_rules"):
        add_block("ppb_rules", ppb_rules_for_helms(helms, sequences=seqs))
    if wants("ppb_position"):
        add_block("ppb_position", ppb_position_for_helms(helms, sequences=seqs))

    if wants("openbabel3d_desc"):
        X_3d, meta3d = _align_structure3d(parsed, Path(args.structure3d_cache))
        add_block("openbabel3d_desc", X_3d, int(meta3d["structure3d_ok"].sum()) if "structure3d_ok" in meta3d else 0)

    if wants("unimol3d_embedding"):
        uni_X, uni_meta = unimol_matrix(helms, caches["unimol"])
        X_uni = pd.DataFrame(uni_X, columns=unimol_feature_names(max(uni_X.shape[1] - 1, 0)))
        add_block("unimol3d_embedding", X_uni, int(uni_meta["unimol3d_cache_hit"].sum()))

    if wants("protenix_sequence_embedding"):
        prot_X, prot_meta = protenix_embedding_matrix(seqs, caches["protenix_embedding"])
        add_block("protenix_sequence_embedding", pd.DataFrame(prot_X), int(prot_meta["protenix_cache_hit"].sum()))

    if wants("protenix_sequence_cif_desc"):
        X_seq_cif, seq_cif_meta = sequence_cif_matrix(seqs, caches["sequence_cif"])
        add_block("protenix_sequence_cif_desc", X_seq_cif, int(seq_cif_meta["protenix_sequence_cif_ok"].sum()))

    if wants("protenix_customccd_embedding"):
        custom_emb_X, custom_emb_meta = customccd_embedding_matrix(helms, caches["customccd_embedding"])
        add_block("protenix_customccd_embedding", pd.DataFrame(custom_emb_X), int(custom_emb_meta["protenix_customccd_cache_hit"].sum()))

    if wants("protenix_customccd_cif_desc"):
        X_custom_cif, custom_cif_meta = customccd_cif_matrix(helms, caches["customccd_cif"])
        add_block("protenix_customccd_cif_desc", X_custom_cif, int(custom_cif_meta["protenix_customccd_cif_ok"].sum()))

    oligoai_path = Path(getattr(args, "oligoai_embedding_file", ""))
    if wants("oligoai_embedding") and oligoai_path.is_file():
        oligoai = pd.read_csv(oligoai_path)
        if dataset_key is not None and "dataset" in oligoai.columns:
            oligoai = oligoai[oligoai["dataset"].astype(str).eq(str(dataset_key))].copy()
        emb_cols = [col for col in oligoai.columns if col.startswith("oligoai_emb_")]
        if emb_cols and "x" in oligoai.columns:
            oligoai = oligoai.drop_duplicates("x", keep="first").set_index("x")
            aligned = oligoai.reindex(helms)[emb_cols]
            hits = int(aligned.notna().all(axis=1).sum())
            aligned = aligned.fillna(0.0).reset_index(drop=True)
            add_block("oligoai_embedding", aligned, hits)

    asoptimizer_path = Path(getattr(args, "asoptimizer_embedding_file", ""))
    if wants("asoptimizer_embedding") and asoptimizer_path.is_file():
        asoptimizer = pd.read_csv(asoptimizer_path)
        if dataset_key is not None and "dataset" in asoptimizer.columns:
            asoptimizer = asoptimizer[asoptimizer["dataset"].astype(str).eq(str(dataset_key))].copy()
        emb_cols = [col for col in asoptimizer.columns if col.startswith("asopt_emb_")]
        if emb_cols and "x" in asoptimizer.columns:
            asoptimizer = asoptimizer.drop_duplicates("x", keep="first").set_index("x")
            aligned = asoptimizer.reindex(helms)[emb_cols]
            hits = int(aligned.notna().all(axis=1).sum())
            add_block("asoptimizer_embedding", aligned.fillna(0.0).reset_index(drop=True), hits)

    if wants("admet_ai_descriptor") and dataset_key is not None:
        admet_path = Path(getattr(args, "admet_ai_cache_dir", "")) / f"{dataset_key}_admet_ai_features.csv"
        if admet_path.exists():
            admet = pd.read_csv(admet_path)
            numeric = admet.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            if len(numeric) != len(parsed):
                raise ValueError(
                    f"{admet_path} has {len(numeric)} rows but {dataset_key} has {len(parsed)}; "
                    "refuse positional alignment across different snapshots"
                )
            hits = int(numeric.notna().any(axis=1).sum())
            add_block("admet_ai_descriptor", numeric.fillna(0.0).reset_index(drop=True), hits)

    families = {
        name: _sanitize_feature_frame(_family_frame(X_base, block, name)).reset_index(drop=True)
        for name, block in raw_blocks.items()
    }
    return families, pd.DataFrame(coverage_rows)


def _run_family_grouped(
    families: dict[str, pd.DataFrame],
    y: np.ndarray,
    groups: pd.Series,
    seed: int,
    folds: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = groups.reset_index(drop=True)
    n_splits = min(folds, int(groups.nunique()))
    splitter = GroupKFold(n_splits=n_splits)
    preds = {name: np.zeros(len(y), dtype=float) for name in families}
    fold_ids = np.zeros(len(y), dtype=int)
    base_X = next(iter(families.values()))

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(base_X, y, groups), start=1):
        fold_ids[test_idx] = fold_idx
        for name, X in families.items():
            model = _model(seed + fold_idx, n_estimators, max_depth, learning_rate)
            model.fit(X.iloc[train_idx], y[train_idx])
            preds[name][test_idx] = model.predict(X.iloc[test_idx])

    pred_df = pd.DataFrame({"y": y, "fold": fold_ids, "group": groups})
    for name, values in preds.items():
        pred_df[f"pred_{name}"] = values
    family_names = list(preds.keys())
    pred_matrix = np.vstack([preds[name] for name in family_names]).T
    pred_df["pred_ensemble_mean"] = pred_matrix.mean(axis=1)
    pred_df["pred_ensemble_median"] = np.median(pred_matrix, axis=1)
    rank_matrix = np.vstack([pd.Series(preds[name]).rank(pct=True).to_numpy() for name in family_names]).T
    pred_df["pred_ensemble_rank_mean"] = rank_matrix.mean(axis=1)

    rows = []
    for name in family_names:
        rows.append({"model": name, "n_features": int(families[name].shape[1]), **_metric(y, preds[name])})
    total_features = int(sum(X.shape[1] for X in families.values()))
    rows.append({"model": "ensemble_mean", "n_features": total_features, **_metric(y, pred_df["pred_ensemble_mean"].to_numpy())})
    rows.append({"model": "ensemble_median", "n_features": total_features, **_metric(y, pred_df["pred_ensemble_median"].to_numpy())})
    rows.append({"model": "ensemble_rank_mean", "n_features": total_features, **_metric(y, pred_df["pred_ensemble_rank_mean"].to_numpy())})
    metrics = pd.DataFrame(rows)
    metrics["n_splits"] = n_splits
    return metrics, pred_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/current_descriptor_families_grouped"))
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
    parser.add_argument("--oligoai-embedding-file", default=str(ROOT / "outputs/oligoai_transfer/oligogym_oligoai_embeddings_full_features_only.csv"))
    parser.add_argument(
        "--asoptimizer-embedding-file",
        default=str(
            ROOT
            / "outputs/external_model_audit/asoptimizer_toxicity_full/"
            "asoptimizer_toxicity_embeddings_64d_merged.csv"
        ),
    )
    parser.add_argument(
        "--admet-ai-cache-dir",
        default=str(ROOT / "outputs/admet_ai_descriptor_grouped/admet_feature_cache"),
    )
    parser.add_argument("--include-similarity-validation", action="store_true", help="Also evaluate sequence-similarity cluster grouped CV.")
    parser.add_argument("--similarity-cluster-file", default=str(ROOT / "outputs/transfer_learning_audit/oligogym_sequence_similarity_clusters.csv"))
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--families", nargs="*", default=None, help="Optional descriptor families to build in addition to core_baseline.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_families = set(args.families or [])

    def selected(name: str) -> bool:
        return not selected_families or name in selected_families

    caches = {}
    if selected("unimol3d_embedding"):
        caches["unimol"] = load_unimol_cache(Path(args.unimol_cache_dir))
    if selected("protenix_sequence_embedding"):
        caches["protenix_embedding"] = discover_protenix_embedding_cache(ROOT, extra_dirs=[Path(args.protenix_cache_dir)])
    if selected("protenix_sequence_cif_desc"):
        caches["sequence_cif"] = load_sequence_cif_descriptor_cache(
            Path(args.protenix_sequence_cif_archive_dir),
            extra_npz_dir=Path(args.protenix_sequence_cif_extra_npz_dir),
        )
    if selected("protenix_customccd_embedding"):
        caches["customccd_embedding"] = load_customccd_embedding_cache(Path(args.customccd_protenix_cache_dir))
    if selected("protenix_customccd_cif_desc"):
        caches["customccd_cif"] = load_customccd_cif_descriptors(Path(args.customccd_protenix_cache_dir))

    all_metrics: list[pd.DataFrame] = []
    all_coverage: list[pd.DataFrame] = []
    for dataset_key in args.datasets:
        ds = get_dataset(dataset_key)
        df = load_oligogym(ds.oligogym_key)
        X_base, parsed = featurize_dataset(df)
        y = parsed["y"].to_numpy()
        print(f"{dataset_key}: building descriptor families", flush=True)
        families, coverage = _build_families(parsed, X_base, args, caches, dataset_key=dataset_key)
        coverage.insert(0, "dataset", dataset_key)
        all_coverage.append(coverage)
        modes = ["sequence"]
        if args.include_similarity_validation:
            modes.append("similarity")
        if dataset_key in TARGET_ELIGIBLE:
            modes.append("target")
        for mode in modes:
            groups = _groups(
                parsed,
                mode,
                dataset_key=dataset_key,
                similarity_cluster_file=Path(args.similarity_cluster_file),
                similarity_threshold=args.similarity_threshold,
            )
            metrics, preds = _run_family_grouped(
                families,
                y,
                groups,
                args.seed,
                args.folds,
                args.n_estimators,
                args.max_depth,
                args.learning_rate,
            )
            metrics.insert(0, "dataset", dataset_key)
            metrics.insert(1, "validation", mode)
            metrics.insert(2, "rows", len(parsed))
            metrics.insert(3, "groups", int(groups.nunique()))
            all_metrics.append(metrics)
            preds.insert(0, "dataset", dataset_key)
            preds.insert(1, "validation", mode)
            preds.to_csv(out_dir / f"{dataset_key}_{mode}_family_predictions.csv", index=False)
            pd.concat(all_metrics, ignore_index=True).to_csv(out_dir / "current_descriptor_family_metrics_partial.csv", index=False)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    coverage_df = pd.concat(all_coverage, ignore_index=True)
    metrics_df.to_csv(out_dir / "current_descriptor_family_metrics.csv", index=False)
    coverage_df.to_csv(out_dir / "current_descriptor_family_coverage.csv", index=False)
    best = metrics_df.loc[metrics_df.groupby(["dataset", "validation"])["spearman"].idxmax()].sort_values(["dataset", "validation"])
    best.to_csv(out_dir / "current_descriptor_family_best.csv", index=False)
    lines = [
        "# Current Descriptor Families Grouped CV",
        "",
        "Each family is `baseline + one descriptor block`; ensemble rows are fixed unweighted combinations of family predictions.",
        "",
        "## Best",
        "",
        best.round(4).to_markdown(index=False),
        "",
        "## All Metrics",
        "",
        metrics_df.round(4).to_markdown(index=False),
        "",
        "## Coverage",
        "",
        coverage_df.to_markdown(index=False),
        "",
    ]
    (out_dir / "CURRENT_DESCRIPTOR_FAMILIES_GROUPED.md").write_text("\n".join(lines), encoding="utf-8")
    print(best.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
