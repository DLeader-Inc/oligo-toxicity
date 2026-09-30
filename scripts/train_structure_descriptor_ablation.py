#!/usr/bin/env python
"""Train endpoint ablations with explicit 2D/3D structure descriptors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import TOXICITY_DATASETS, load_oligogym
from src.features.featurizer import featurize_dataset
from src.features.protenix_embedding_features import discover_feature_cache, matrix_for_sequences as protenix_matrix
from src.features.protenix_customccd_features import (
    descriptor_matrix_for_helms as customccd_descriptor_matrix,
    load_cif_descriptor_cache as load_customccd_cif_descriptors,
    load_compact_embedding_cache as load_customccd_embedding_cache,
    matrix_for_helms as customccd_matrix,
)
from src.features.protenix_sequence_cif_features import (
    descriptor_matrix_for_sequences as sequence_cif_descriptor_matrix,
    load_sequence_cif_descriptor_cache,
)
from src.features.secondary_structure_features import dataframe_for_helms as secondary2d_for_helms
from src.features.structure_descriptor_features import (
    dataframe_for_helms as structure3d_for_helms,
    numeric_descriptor_frame,
)
from src.features.unimol_3d_embedding_features import load_feature_cache as load_unimol_cache
from src.features.unimol_3d_embedding_features import helm_key
from src.features.unimol_3d_embedding_features import matrix_for_helms as unimol_matrix
from src.models.xgboost_baseline import aggregate_metrics, train_xgboost_cv


def _numeric_secondary(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["ss_sequence", "ss_dotbracket"], errors="ignore").astype(float)


def _sanitize_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Make feature matrices safe for XGBoost/PCA."""
    out = X.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].clip(lower=-1e12, upper=1e12)
    return out.astype(float)


def _run_xgb(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    seeds: tuple[int, ...],
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> dict:
    X = _sanitize_feature_frame(X)
    metrics = aggregate_metrics(
        train_xgboost_cv(
            X.reset_index(drop=True),
            y,
            seeds=seeds,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )
    )
    return {"model": name, "n_features": int(X.shape[1]), **metrics}


def _run_xgb_with_fold_pca(
    name: str,
    X_base: pd.DataFrame,
    X_desc: pd.DataFrame,
    y: np.ndarray,
    seeds: tuple[int, ...],
    pca_dim: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> dict:
    """Fit scaler/PCA inside each CV fold to avoid descriptor leakage."""
    seed_metrics = []
    X_base_np = _sanitize_feature_frame(X_base).reset_index(drop=True).to_numpy(dtype=float)
    X_desc_np = _sanitize_feature_frame(X_desc).reset_index(drop=True).to_numpy(dtype=float)
    X_base_np = np.nan_to_num(X_base_np, nan=0.0, posinf=1e12, neginf=-1e12)
    X_desc_np = np.nan_to_num(X_desc_np, nan=0.0, posinf=1e12, neginf=-1e12)
    max_dim = max(1, min(pca_dim, X_desc_np.shape[1], max(len(y) - 2, 1)))

    for seed in seeds:
        preds = np.zeros(len(y), dtype=float)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X_base_np):
            scaler = StandardScaler()
            train_desc = scaler.fit_transform(X_desc_np[train_idx])
            test_desc = scaler.transform(X_desc_np[test_idx])
            n_dim = min(max_dim, train_desc.shape[0] - 1, train_desc.shape[1])
            if n_dim <= 0:
                train_pca = np.zeros((len(train_idx), 1), dtype=float)
                test_pca = np.zeros((len(test_idx), 1), dtype=float)
            else:
                pca = PCA(n_components=n_dim, random_state=seed)
                train_pca = pca.fit_transform(train_desc)
                test_pca = pca.transform(test_desc)
            X_train = np.hstack([X_base_np[train_idx], train_pca])
            X_test = np.hstack([X_base_np[test_idx], test_pca])
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=seed,
                verbosity=0,
                n_jobs=-1,
            )
            model.fit(X_train, y[train_idx])
            preds[test_idx] = model.predict(X_test)
        sp, _ = stats.spearmanr(y, preds)
        pe, _ = stats.pearsonr(y, preds)
        seed_metrics.append(
            {
                "spearman": float(sp),
                "pearson": float(pe),
                "r2": float(r2_score(y, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            }
        )

    return {
        "model": name,
        "n_features": int(X_base.shape[1] + max_dim),
        "pca_dim": int(max_dim),
        "spearman_mean": round(float(np.mean([m["spearman"] for m in seed_metrics])), 4),
        "spearman_std": round(float(np.std([m["spearman"] for m in seed_metrics])), 4),
        "pearson_mean": round(float(np.mean([m["pearson"] for m in seed_metrics])), 4),
        "r2_mean": round(float(np.mean([m["r2"] for m in seed_metrics])), 4),
        "rmse_mean": round(float(np.mean([m["rmse"] for m in seed_metrics])), 4),
        "n_seeds": int(len(seeds)),
    }


def _load_or_compute_3d_cache(
    helms: pd.Series,
    cache_path: Path,
    conformer_dir: Path,
    backend: str,
    timeout_sec: int,
    seed: int,
    compute_limit: int | None,
) -> pd.DataFrame:
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["helm"])
    done = set(cache.get("helm", pd.Series(dtype=str)).astype(str))
    pending = [h for h in pd.Series(helms).astype(str).drop_duplicates() if h not in done]
    if compute_limit is not None:
        pending = pending[:compute_limit]

    rows = []
    for idx, helm in enumerate(pending, start=1):
        print(f"  3D descriptor {idx}/{len(pending)} backend={backend}", flush=True)
        sdf_path = conformer_dir / f"conformer_{helm_key(helm)}.sdf"
        row = {"helm": helm}
        row.update(
            structure3d_for_helms(
                [helm],
                backend=backend,
                seed=seed,
                timeout_sec=timeout_sec,
                save_sdf_paths=[sdf_path],
            )
            .iloc[0]
            .to_dict()
        )
        row["conformer_sdf"] = str(sdf_path) if sdf_path.exists() else ""
        rows.append(row)
        if rows and (len(rows) % 10 == 0):
            cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            cache.to_csv(cache_path, index=False)
            rows = []
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
    cache = cache.drop_duplicates("helm", keep="last")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(cache_path, index=False)
    return cache


def _align_3d(parsed: pd.DataFrame, cache: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.DataFrame({"helm": parsed["x"].astype(str)}).merge(cache, on="helm", how="left")
    status_col = next((col for col in merged.columns if col.endswith("_status")), None)
    meta = pd.DataFrame(
        {
            "descriptor3d_cache_hit": merged[status_col].notna().astype(int) if status_col else 0,
            "descriptor3d_ok": (merged[status_col] == "ok").astype(int) if status_col else 0,
        }
    )
    X = numeric_descriptor_frame(merged.drop(columns=["helm"], errors="ignore"))
    return X, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/structure_descriptor_ablation"))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-pos", type=int, default=30)
    parser.add_argument("--no-pair-probs", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--descriptor3d-cache", default=str(ROOT / "outputs/structure_descriptor_cache/structure3d_descriptors.csv"))
    parser.add_argument("--descriptor3d-conformer-dir", default=None, help="Directory for saved SDF conformers. Defaults to <descriptor cache parent>/conformers.")
    parser.add_argument("--descriptor3d-backend", choices=["openbabel", "rdkit"], default="openbabel")
    parser.add_argument("--descriptor3d-timeout-sec", type=int, default=30)
    parser.add_argument("--descriptor3d-compute-limit", type=int, default=None)
    parser.add_argument("--unimol-cache-dir", default=str(ROOT / "outputs/unimol3d_full_cache_openbabel_cuda"))
    parser.add_argument("--protenix-cache-dir", default=str(ROOT / "outputs/protenix_full_cache"))
    parser.add_argument("--customccd-protenix-cache-dir", default=str(ROOT / "outputs/protenix_customccd_full_cache"))
    parser.add_argument("--protenix-sequence-cif-archive-dir", default=str(ROOT / "outputs/protenix_structure_archive"))
    parser.add_argument("--protenix-sequence-cif-extra-npz-dir", default=str(ROOT / "outputs/protenix_plain_sequence_structure_cache"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds)
    selected = set(args.datasets) if args.datasets else None
    descriptor_cache_path = Path(args.descriptor3d_cache)
    conformer_dir = Path(args.descriptor3d_conformer_dir) if args.descriptor3d_conformer_dir else descriptor_cache_path.parent / "conformers"

    all_helms = []
    selected_datasets = []
    for ds in TOXICITY_DATASETS:
        if selected is not None and ds.model_key not in selected:
            continue
        df = load_oligogym(ds.oligogym_key)
        if args.limit:
            df = df.head(args.limit).copy()
        selected_datasets.append((ds, df))
        all_helms.extend(df["x"].astype(str).to_list())

    descriptor3d_cache = pd.DataFrame(columns=["helm"])
    if not args.skip_3d:
        descriptor3d_cache = _load_or_compute_3d_cache(
            pd.Series(all_helms),
            descriptor_cache_path,
            conformer_dir,
            backend=args.descriptor3d_backend,
            timeout_sec=args.descriptor3d_timeout_sec,
            seed=seeds[0],
            compute_limit=args.descriptor3d_compute_limit,
        )

    unimol_cache = load_unimol_cache(Path(args.unimol_cache_dir))
    protenix_cache = discover_feature_cache(ROOT, extra_dirs=[Path(args.protenix_cache_dir)])
    customccd_cache_dir = Path(args.customccd_protenix_cache_dir)
    customccd_cache = load_customccd_embedding_cache(customccd_cache_dir)
    customccd_cif_descriptor_cache = load_customccd_cif_descriptors(customccd_cache_dir)
    sequence_cif_descriptor_cache = load_sequence_cif_descriptor_cache(
        Path(args.protenix_sequence_cif_archive_dir),
        extra_npz_dir=Path(args.protenix_sequence_cif_extra_npz_dir),
    )

    metric_rows: list[dict] = []
    coverage_rows: list[dict] = []
    example_rows: list[dict] = []

    for ds, df in selected_datasets:
        print(f"\n=== {ds.model_key} ===", flush=True)
        X_base, parsed = featurize_dataset(df)
        y = parsed["y"].to_numpy()

        ss_full = secondary2d_for_helms(parsed["x"], max_pos=args.max_pos, with_pair_probs=not args.no_pair_probs)
        X_2d = _numeric_secondary(ss_full)
        X_base_2d = pd.concat([X_base.reset_index(drop=True), X_2d.reset_index(drop=True)], axis=1)

        metric_rows.append({"dataset": ds.model_key, **_run_xgb("baseline_xgboost", X_base, y, seeds, args.n_estimators, args.max_depth, args.learning_rate)})
        metric_rows.append({"dataset": ds.model_key, **_run_xgb("baseline_secondary2d_xgboost", X_base_2d, y, seeds, args.n_estimators, args.max_depth, args.learning_rate)})
        for dim in args.pca_dims:
            metric_rows.append(
                {
                    "dataset": ds.model_key,
                    **_run_xgb_with_fold_pca(
                        f"baseline_secondary2d_pca{dim}_xgboost",
                        X_base,
                        X_2d,
                        y,
                        seeds,
                        dim,
                        args.n_estimators,
                        args.max_depth,
                        args.learning_rate,
                    ),
                }
            )

        descriptor3d_ok = 0
        descriptor3d_hits = 0
        if not args.skip_3d:
            X_3d, meta3d = _align_3d(parsed, descriptor3d_cache)
            descriptor3d_hits = int(meta3d["descriptor3d_cache_hit"].sum())
            descriptor3d_ok = int(meta3d["descriptor3d_ok"].sum())
            if descriptor3d_ok > 0:
                X_base_3d = pd.concat([X_base.reset_index(drop=True), X_3d.reset_index(drop=True)], axis=1)
                X_base_2d_3d = pd.concat([X_base_2d.reset_index(drop=True), X_3d.reset_index(drop=True)], axis=1)
                metric_rows.append({"dataset": ds.model_key, **_run_xgb("baseline_structure3d_xgboost", X_base_3d, y, seeds, args.n_estimators, args.max_depth, args.learning_rate)})
                metric_rows.append({"dataset": ds.model_key, **_run_xgb("baseline_secondary2d_structure3d_xgboost", X_base_2d_3d, y, seeds, args.n_estimators, args.max_depth, args.learning_rate)})
                for dim in args.pca_dims:
                    metric_rows.append(
                        {
                            "dataset": ds.model_key,
                            **_run_xgb_with_fold_pca(
                                f"baseline_structure3d_pca{dim}_xgboost",
                                X_base,
                                X_3d,
                                y,
                                seeds,
                                dim,
                                args.n_estimators,
                                args.max_depth,
                                args.learning_rate,
                            ),
                        }
                    )

        uni_X, uni_meta = unimol_matrix(parsed["x"], unimol_cache)
        if int(uni_meta["unimol3d_cache_hit"].sum()) > 0:
            X_uni = pd.DataFrame(uni_X, index=X_base.index)
            metric_rows.append({"dataset": ds.model_key, **_run_xgb("baseline_unimol3d_xgboost", pd.concat([X_base.reset_index(drop=True), X_uni.reset_index(drop=True)], axis=1), y, seeds, args.n_estimators, args.max_depth, args.learning_rate)})
            for dim in args.pca_dims:
                metric_rows.append(
                    {
                        "dataset": ds.model_key,
                        **_run_xgb_with_fold_pca(
                            f"baseline_unimol3d_pca{dim}_xgboost",
                            X_base,
                            X_uni,
                            y,
                            seeds,
                            dim,
                            args.n_estimators,
                            args.max_depth,
                            args.learning_rate,
                        ),
                    }
                )

        prot_X, prot_meta = protenix_matrix(parsed["bases"], protenix_cache)
        if int(prot_meta["protenix_cache_hit"].sum()) > 0:
            X_prot = pd.DataFrame(prot_X, index=X_base.index)
            metric_rows.append({"dataset": ds.model_key, **_run_xgb("baseline_protenix_xgboost", pd.concat([X_base.reset_index(drop=True), X_prot.reset_index(drop=True)], axis=1), y, seeds, args.n_estimators, args.max_depth, args.learning_rate)})
            for dim in args.pca_dims:
                metric_rows.append(
                    {
                        "dataset": ds.model_key,
                        **_run_xgb_with_fold_pca(
                            f"baseline_protenix_pca{dim}_xgboost",
                            X_base,
                            X_prot,
                            y,
                            seeds,
                            dim,
                            args.n_estimators,
                            args.max_depth,
                            args.learning_rate,
                        ),
                    }
                )

        X_seq_cif_desc, seq_cif_meta = sequence_cif_descriptor_matrix(parsed["bases"], sequence_cif_descriptor_cache)
        if int(seq_cif_meta["protenix_sequence_cif_ok"].sum()) > 0:
            metric_rows.append({"dataset": ds.model_key, **_run_xgb("baseline_protenix_sequence_cif3d_desc_xgboost", pd.concat([X_base.reset_index(drop=True), X_seq_cif_desc.reset_index(drop=True)], axis=1), y, seeds, args.n_estimators, args.max_depth, args.learning_rate)})
            for dim in args.pca_dims:
                metric_rows.append(
                    {
                        "dataset": ds.model_key,
                        **_run_xgb_with_fold_pca(
                            f"baseline_protenix_sequence_cif3d_desc_pca{dim}_xgboost",
                            X_base,
                            X_seq_cif_desc,
                            y,
                            seeds,
                            dim,
                            args.n_estimators,
                            args.max_depth,
                            args.learning_rate,
                        ),
                    }
                )

        custom_X, custom_meta = customccd_matrix(parsed["x"], customccd_cache)
        if int(custom_meta["protenix_customccd_cache_hit"].sum()) > 0:
            X_custom = pd.DataFrame(custom_X, index=X_base.index)
            metric_rows.append({"dataset": ds.model_key, **_run_xgb("baseline_protenix_customccd128_xgboost", pd.concat([X_base.reset_index(drop=True), X_custom.reset_index(drop=True)], axis=1), y, seeds, args.n_estimators, args.max_depth, args.learning_rate)})
            for dim in args.pca_dims:
                metric_rows.append(
                    {
                        "dataset": ds.model_key,
                        **_run_xgb_with_fold_pca(
                            f"baseline_protenix_customccd128_pca{dim}_xgboost",
                            X_base,
                            X_custom,
                            y,
                            seeds,
                            dim,
                            args.n_estimators,
                            args.max_depth,
                            args.learning_rate,
                        ),
                    }
                )

        X_custom_desc, custom_desc_meta = customccd_descriptor_matrix(parsed["x"], customccd_cif_descriptor_cache)
        if int(custom_desc_meta["protenix_customccd_cif_ok"].sum()) > 0:
            metric_rows.append({"dataset": ds.model_key, **_run_xgb("baseline_protenix_cif3d_desc_xgboost", pd.concat([X_base.reset_index(drop=True), X_custom_desc.reset_index(drop=True)], axis=1), y, seeds, args.n_estimators, args.max_depth, args.learning_rate)})
            for dim in args.pca_dims:
                metric_rows.append(
                    {
                        "dataset": ds.model_key,
                        **_run_xgb_with_fold_pca(
                            f"baseline_protenix_cif3d_desc_pca{dim}_xgboost",
                            X_base,
                            X_custom_desc,
                            y,
                            seeds,
                            dim,
                            args.n_estimators,
                            args.max_depth,
                            args.learning_rate,
                        ),
                    }
                )

        if (
            int(seq_cif_meta["protenix_sequence_cif_ok"].sum()) > 0
            and int(custom_desc_meta["protenix_customccd_cif_ok"].sum()) > 0
        ):
            X_seq_custom_desc = pd.concat(
                [X_seq_cif_desc.reset_index(drop=True), X_custom_desc.reset_index(drop=True)],
                axis=1,
            )
            metric_rows.append(
                {
                    "dataset": ds.model_key,
                    **_run_xgb(
                        "baseline_protenix_sequence_plus_customccd_cif3d_desc_xgboost",
                        pd.concat([X_base.reset_index(drop=True), X_seq_custom_desc], axis=1),
                        y,
                        seeds,
                        args.n_estimators,
                        args.max_depth,
                        args.learning_rate,
                    ),
                }
            )
            for dim in args.pca_dims:
                metric_rows.append(
                    {
                        "dataset": ds.model_key,
                        **_run_xgb_with_fold_pca(
                            f"baseline_protenix_sequence_plus_customccd_cif3d_desc_pca{dim}_xgboost",
                            X_base,
                            X_seq_custom_desc,
                            y,
                            seeds,
                            dim,
                            args.n_estimators,
                            args.max_depth,
                            args.learning_rate,
                        ),
                    }
                )

        coverage_rows.append(
            {
                "dataset": ds.model_key,
                "rows": int(len(parsed)),
                "secondary2d_rows": int(len(X_2d)),
                "descriptor3d_cache_hits": descriptor3d_hits,
                "descriptor3d_ok": descriptor3d_ok,
                "unimol3d_cache_hits": int(uni_meta["unimol3d_cache_hit"].sum()),
                "protenix_cache_hits": int(prot_meta["protenix_cache_hit"].sum()),
                "protenix_sequence_cif_hits": int(seq_cif_meta["protenix_sequence_cif_hit"].sum()),
                "protenix_sequence_cif_ok": int(seq_cif_meta["protenix_sequence_cif_ok"].sum()),
                "protenix_customccd128_cache_hits": int(custom_meta["protenix_customccd_cache_hit"].sum()),
                "protenix_customccd_cif_hits": int(custom_desc_meta["protenix_customccd_cif_hit"].sum()),
                "protenix_customccd_cif_ok": int(custom_desc_meta["protenix_customccd_cif_ok"].sum()),
                "mean_pair_fraction": float(X_2d["ss_pair_fraction"].mean()),
                "mean_contact_order": float(X_2d["ss_contact_order"].mean()),
            }
        )
        for i in range(min(3, len(parsed))):
            example_rows.append(
                {
                    "dataset": ds.model_key,
                    "helm": parsed["x"].iloc[i],
                    "sequence": ss_full["ss_sequence"].iloc[i],
                    "dotbracket": ss_full["ss_dotbracket"].iloc[i],
                    "y": float(parsed["y"].iloc[i]),
                    "ss_mfe": float(X_2d["ss_mfe"].iloc[i]),
                    "ss_pair_fraction": float(X_2d["ss_pair_fraction"].iloc[i]),
                    "ss_contact_order": float(X_2d["ss_contact_order"].iloc[i]),
                }
            )

        pd.DataFrame(metric_rows).to_csv(out_dir / "structure_descriptor_ablation_metrics_partial.csv", index=False)
        pd.DataFrame(coverage_rows).to_csv(out_dir / "structure_descriptor_ablation_coverage_partial.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    coverage = pd.DataFrame(coverage_rows)
    examples = pd.DataFrame(example_rows)
    metrics.to_csv(out_dir / "structure_descriptor_ablation_metrics.csv", index=False)
    coverage.to_csv(out_dir / "structure_descriptor_ablation_coverage.csv", index=False)
    examples.to_csv(out_dir / "structure_descriptor_examples.csv", index=False)

    report = [
        "# Structure Descriptor Ablation",
        "",
        "This compares baseline HELM tabular features with explicit 2D fold descriptors, explicit 3D conformer descriptors, cached UniMol3D embeddings, sequence-only Protenix embeddings, custom-CCD Protenix 128D embeddings, and descriptors computed directly from saved Protenix CIF structures.",
        "",
        "Fold-safe PCA is fitted inside each CV fold for the `*_pca*_xgboost` rows.",
        "",
        "## Coverage",
        "",
        coverage.round(4).to_markdown(index=False),
        "",
        "## Metrics",
        "",
        metrics.round(4).to_markdown(index=False),
        "",
        "## Examples",
        "",
        examples.head(15).round(4).to_markdown(index=False),
        "",
    ]
    (out_dir / "STRUCTURE_DESCRIPTOR_ABLATION.md").write_text("\n".join(report), encoding="utf-8")
    print(out_dir, flush=True)
    print(coverage.to_string(index=False), flush=True)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
