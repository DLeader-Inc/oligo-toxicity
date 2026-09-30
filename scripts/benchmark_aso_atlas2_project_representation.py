#!/usr/bin/env python3
"""Compare ASO Atlas 2 XGBoost inputs with the project's fixed two-branch model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarking.descriptor_family_tree_models import regression_metrics  # noqa: E402
from src.features.aso_explicit_chemistry_features import (  # noqa: E402
    SCHEMA_VERSION as EXPLICIT_CHEMISTRY_SCHEMA_VERSION,
    canonical_aso_sequence,
    dataframe_for_helms as explicit_chemistry_dataframe_for_helms,
)
from src.features.featurizer import parse_helm_components  # noqa: E402
from src.features.position_reencoding_features import dataframes_for_helms  # noqa: E402
from src.features.sequence_chemistry_interaction_features import (  # noqa: E402
    dataframe_for_helms as seqchem_dataframe_for_helms,
)
from src.features.universal_oligo_representation import (  # noqa: E402
    universal_context_names,
    universal_core_names,
    universal_representation_frames,
)
from scripts.audit_sequence_similarity_groups import (  # noqa: E402
    cluster_sequences,
    compute_pair_identities,
)


DATA_DIR = ROOT / "data" / "external" / "aso_atlas_2_v2_0_2"
DEFAULT_OUT = ROOT / "outputs" / "aso_atlas2_project_xgb_comparison_v1"
TASKS = ("mouse_hepatic", "rat_hepatic", "mouse_neuro", "rat_neuro")
SOURCE_SPEARMAN = {
    "mouse_hepatic": (0.415, 0.110),
    "rat_hepatic": (0.179, 0.107),
    "mouse_neuro": (0.680, 0.023),
    "rat_neuro": (0.557, 0.101),
}
PROJECT_PARAMS = {
    "n_estimators": 600,
    "max_depth": 4,
    "learning_rate": 0.02,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "reg_alpha": 0.1,
}
AUTHOR_PARAMS = {"n_estimators": 500, "max_depth": 10}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_helm(value: object) -> str:
    helm = re.sub(r"\{+", "{", str(value))
    return re.sub(r"\}+(?=\$\$\$\$)", "}", helm)


def mean_array(value: object) -> float:
    if value is None:
        return np.nan
    try:
        values = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.nan
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def official_valid_chemistry(value: object) -> bool:
    """Reproduce ASO Atlas 2's fixed-commit Helm.valid_chemistry filter."""

    if pd.isna(value) or value == "None":
        return False
    helm = str(value)
    for excluded in ("[lna]", "[LNA]", "[fR]", "[FR]", "[am]", "[AM]", "[?]"):
        if excluded in helm:
            return False
    return re.search(r"\[m\]\(", helm) is None


def add_earliest_patent_group(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[
        frame["HELM Annotation"].notna() & frame["USPTO ID"].notna()
    ].copy()
    frame = frame.sort_values("USPTO ID", kind="stable")
    first = (
        frame.drop_duplicates("HELM Annotation", keep="first")
        .set_index("HELM Annotation")["USPTO ID"]
    )
    frame["cv_group"] = frame["HELM Annotation"].map(first).map(
        lambda value: f"patent_{value}"
    )
    frame["x"] = frame["HELM Annotation"].map(normalize_helm)
    return frame.reset_index(drop=True)


def load_task(task: str) -> pd.DataFrame:
    if task.endswith("hepatic"):
        species = task.removesuffix("_hepatic")
        frame = pd.read_parquet(DATA_DIR / "hepatotoxicity.parquet")
        frame = frame[
            frame["species"].eq(species)
            & frame["HELM Annotation"].map(official_valid_chemistry)
            & frame["ALT"].notna()
        ].copy()
        frame["y"] = frame["ALT"].map(mean_array)
    else:
        species = "Mouse" if task == "mouse_neuro" else "Rat"
        frame = pd.read_parquet(DATA_DIR / "neurotoxicity.parquet")
        mask = (
            frame["species"].eq(species)
            & frame["HELM Annotation"].map(official_valid_chemistry)
            & frame["FOB_score"].notna()
        )
        if task == "mouse_neuro":
            mask &= frame["dosage_ug"].eq(700) & frame[
                "administration_method"
            ].eq("ICV")
        else:
            mask &= frame["dosage_ug"].eq(3000) & frame[
                "latency_time_hours"
            ].eq(3)
        frame = frame[mask].copy()
        frame["y"] = frame["FOB_score"].map(mean_array)
    frame = frame[np.isfinite(frame["y"])].copy()
    return add_earliest_patent_group(frame)


def chemistry_counts(helms: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for helm in helms:
        residues = parse_helm_components(helm)
        sugars = Counter(residue.raw_sugar.lower() for residue in residues)
        nonterminal = residues[:-1]
        n_ps = sum(residue.raw_linkage.lower() == "sp" for residue in nonterminal)
        n_po = sum(residue.raw_linkage.lower() != "sp" for residue in nonterminal)
        rows.append(
            [
                float(sugars["moe"]),
                float(sugars["cet"]),
                float(sugars["d"]),
                float(n_ps),
                float(n_po),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def experimental_conditions(frame: pd.DataFrame) -> np.ndarray:
    if "dosage_mg_per_kg" not in frame:
        return np.empty((len(frame), 0), dtype=np.float32)
    columns = ["dosage_mg_per_kg", "num_doses", "dosing_period_days"]
    values = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        values.append(series.fillna(series.median()).to_numpy(dtype=np.float32))
    return np.column_stack(values)


def author_onehot(helms: list[str]) -> np.ndarray:
    from oligogym.features import OneHotEncoder

    encoder = OneHotEncoder(encode_components=["base", "sugar", "phosphate"])
    matrix = encoder.fit_transform(np.asarray(helms, dtype=object))
    return np.asarray(matrix, dtype=np.float32).reshape(len(helms), -1)


def similarity90_groups(
    frame: pd.DataFrame, task: str, out_dir: Path
) -> tuple[np.ndarray, dict[str, object]]:
    sequences = frame["x"].map(canonical_aso_sequence).astype(str)
    unique_sequences = sorted(sequences.unique().tolist())
    pairs, best_identity = compute_pair_identities(
        unique_sequences, min_threshold=0.90
    )
    mapping = cluster_sequences(unique_sequences, pairs, threshold=0.90)
    groups = sequences.map(mapping).astype(str).to_numpy()
    cluster_sequence_sizes = pd.Series(mapping).value_counts()
    cluster_row_sizes = pd.Series(groups).value_counts()
    assignments = pd.DataFrame(
        {
            "row_index": np.arange(len(frame), dtype=int),
            "canonical_sequence": sequences,
            "similarity90_group": groups,
            "original_helm": frame["HELM Annotation"].astype(str).to_numpy(),
        }
    )
    assignments.to_csv(
        out_dir / f"{task}_similarity90_assignments.csv.gz",
        index=False,
        compression="gzip",
    )
    return groups, {
        "threshold": 0.90,
        "identity": "1 - edit_distance/max_length; Hamming for equal lengths",
        "unique_sequences": len(unique_sequences),
        "similarity90_groups": int(len(cluster_sequence_sizes)),
        "multi_sequence_groups": int((cluster_sequence_sizes > 1).sum()),
        "largest_group_sequences": int(cluster_sequence_sizes.max()),
        "largest_group_rows": int(cluster_row_sizes.max()),
        "largest_group_fraction": float(cluster_row_sizes.max() / len(frame)),
        "high_identity_pairs": len(pairs),
        "maximum_nonidentical_identity": float(best_identity),
    }


def project_features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    helms = frame["x"].astype(str).tolist()
    representations, _, row_audit = universal_representation_frames(
        frame.loc[:, ["x"]]
    )
    enhanced = representations["universal_ohe_enhanced"]
    raw = representations["universal_ohe_raw"].to_numpy(dtype=np.float32)
    core = enhanced.loc[:, list(universal_core_names())].to_numpy(dtype=np.float32)
    context = enhanced.loc[:, list(universal_context_names())].to_numpy(
        dtype=np.float32
    )
    seqchem = seqchem_dataframe_for_helms(helms).to_numpy(dtype=np.float32)
    position = dataframes_for_helms(helms)["position_interactions"].to_numpy(
        dtype=np.float32
    )
    explicit_frame = explicit_chemistry_dataframe_for_helms(helms)
    explicit = explicit_frame.to_numpy(dtype=np.float32)
    explicit_local = explicit_frame.loc[
        :, [name for name in explicit_frame if name.startswith("aso3_pos_")]
    ].to_numpy(dtype=np.float32)
    explicit_summary = explicit_frame.loc[
        :, [name for name in explicit_frame if not name.startswith("aso3_pos_")]
    ].to_numpy(dtype=np.float32)
    explicit_quantity_names = (
        "aso3_n_cEt",
        "aso3_frac_cEt",
        "aso3_n_5meC",
        "aso3_frac_5meC",
    )
    explicit_region_names = tuple(
        name
        for name in explicit_frame
        if any(f"aso3_{region}_" in name for region in ("wing5", "gap_first3", "gap_mid", "gap_last3", "wing3", "other"))
    )
    explicit_architecture_names = tuple(
        name
        for name in explicit_frame
        if name not in explicit_quantity_names
        and not name.startswith("aso3_pos_")
        and name not in explicit_region_names
    )
    explicit_quantity = explicit_frame.loc[:, explicit_quantity_names].to_numpy(
        dtype=np.float32
    )
    explicit_region = explicit_frame.loc[:, explicit_region_names].to_numpy(
        dtype=np.float32
    )
    explicit_architecture = explicit_frame.loc[
        :, explicit_architecture_names
    ].to_numpy(dtype=np.float32)
    dimensions = (raw.shape[1], core.shape[1], context.shape[1], seqchem.shape[1], position.shape[1])
    if any(value <= 0 for value in dimensions):
        raise RuntimeError(f"empty project feature block: {dimensions}")
    branch_a = np.column_stack([raw, position]).astype(np.float32)
    branch_b = np.column_stack([core, context, seqchem]).astype(np.float32)
    conditions = experimental_conditions(frame)
    return {
        "raw": raw,
        "branch_a": branch_a,
        "branch_b": branch_b,
        "explicit_cet_5mec": explicit,
        "explicit_cet_5mec_local": explicit_local,
        "explicit_cet_5mec_summary": explicit_summary,
        "explicit_cet_5mec_quantity": explicit_quantity,
        "explicit_cet_5mec_position": explicit_local,
        "explicit_cet_5mec_region": explicit_region,
        "explicit_cet_5mec_architecture": explicit_architecture,
        "branch_a_explicit": np.column_stack([branch_a, explicit]).astype(np.float32),
        "branch_b_explicit": np.column_stack([branch_b, explicit]).astype(np.float32),
        "branch_a_explicit_routed": np.column_stack(
            [branch_a, explicit_local]
        ).astype(np.float32),
        "branch_b_explicit_routed": np.column_stack(
            [branch_b, explicit_summary]
        ).astype(np.float32),
        "branch_a_conditions": np.column_stack([branch_a, conditions]).astype(
            np.float32
        ),
        "branch_b_conditions": np.column_stack([branch_b, conditions]).astype(
            np.float32
        ),
        "row_audit": row_audit,
    }


def input_audit(task: str, frame: pd.DataFrame, row_audit: pd.DataFrame) -> dict[str, object]:
    raw_cet = 0
    raw_5mec = 0
    rows_cet = 0
    rows_5mec = 0
    lengths: list[int] = []
    for helm in frame["x"].astype(str):
        residues = parse_helm_components(helm)
        lengths.append(len(residues))
        cet = sum(residue.raw_sugar.lower() == "cet" for residue in residues)
        mec = sum(residue.raw_base.lower() == "5mec" for residue in residues)
        raw_cet += cet
        raw_5mec += mec
        rows_cet += bool(cet)
        rows_5mec += bool(mec)
    metadata = {}
    for column in (
        "target_RNA",
        "species_strain",
        "dosage_mg_per_kg",
        "num_doses",
        "dosing_period_days",
        "dosage_ug",
        "administration_method",
        "latency_time_hours",
        "measurement_source",
    ):
        if column in frame:
            metadata[column] = {
                "nonmissing": int(frame[column].notna().sum()),
                "unique_nonmissing": int(frame[column].dropna().astype(str).nunique()),
            }
    return {
        "task": task,
        "rows": len(frame),
        "unique_helm": int(frame["HELM Annotation"].nunique()),
        "patent_groups": int(frame["cv_group"].nunique()),
        "sequence_length_min": min(lengths),
        "sequence_length_max": max(lengths),
        "rows_with_cet": int(rows_cet),
        "raw_cet_residues": int(raw_cet),
        "rows_with_5meC": int(rows_5mec),
        "raw_5meC_residues": int(raw_5mec),
        "project_unknown_sugar_count": int(row_audit["unknown_sugar_count"].sum()),
        "project_unknown_base_count": int(row_audit["unknown_base_count"].sum()),
        "metadata": metadata,
    }


def model(params: dict[str, object], seed: int, device: str) -> XGBRegressor:
    return XGBRegressor(
        **params,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        device=device,
        verbosity=0,
    )


def author_model() -> XGBRegressor:
    """Match the public OligoGym wrapper without adding project settings."""

    return XGBRegressor(**AUTHOR_PARAMS)


def fit_predict(
    matrix: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    params: dict[str, object],
    seed: int,
    device: str,
) -> np.ndarray:
    estimator = model(params, seed, device)
    estimator.fit(matrix[train], y[train])
    return np.asarray(estimator.predict(matrix[test]), dtype=float)


def fit_predict_author(
    matrix: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    estimator = author_model()
    estimator.fit(matrix[train], y[train])
    return np.asarray(estimator.predict(matrix[test]), dtype=float)


def run_task(
    task: str,
    out_dir: Path,
    *,
    device: str,
    probe: bool,
    log_hepatic_target: bool,
    explicit_cet_5mec: bool,
    split_mode: str,
    oligoai_blocks: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    started = time.time()
    frame = load_task(task)
    y = frame["y"].to_numpy(dtype=float)
    use_log_target = log_hepatic_target and task.endswith("hepatic")
    fit_y = np.log1p(y) if use_log_target else y
    similarity_audit = None
    if split_mode == "patent":
        groups = frame["cv_group"].astype(str).to_numpy()
    else:
        groups, similarity_audit = similarity90_groups(frame, task, out_dir)
    helms = frame["x"].astype(str).tolist()
    project = project_features(frame)
    oligoai = None
    if oligoai_blocks is not None:
        block = np.load(oligoai_blocks)
        alignment = pd.read_csv(oligoai_blocks.parent / "inference_input.csv.gz")
        lookup = {helm: index for index, helm in enumerate(alignment["normalized_helm"].astype(str))}
        missing = [helm for helm in helms if helm not in lookup]
        if missing:
            raise ValueError(f"{task}: {len(missing)} HELMs missing from OligoAI cache")
        oligoai = block["aso_embedding"][[lookup[helm] for helm in helms]].astype(np.float32)
    author = np.column_stack(
        [author_onehot(helms), chemistry_counts(helms), experimental_conditions(frame)]
    ).astype(np.float32)
    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(author, y, groups))
    if probe:
        splits = splits[:1]

    predictions: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits, start=1):
        routes: dict[str, np.ndarray] = {}
        routes["author_recipe_reproduction"] = fit_predict_author(
            author, fit_y, train, test
        )
        routes["author_features_project_xgb"] = fit_predict(
            author, fit_y, train, test, PROJECT_PARAMS, fold, device
        )
        routes["project_raw"] = fit_predict(
            project["raw"], fit_y, train, test, PROJECT_PARAMS, fold, device
        )
        conditions = experimental_conditions(frame)
        raw_conditions = np.column_stack([project["raw"], conditions]).astype(
            np.float32
        )
        routes["project_raw_plus_conditions"] = fit_predict(
            raw_conditions, fit_y, train, test, PROJECT_PARAMS, fold, device
        )
        if explicit_cet_5mec:
            # Chemistry-complete position One-Hot baseline: two indicators per
            # residue slot (cEt and 5meC), without regional/count descriptors.
            raw_explicit_position = np.column_stack(
                [project["raw"], project["explicit_cet_5mec_position"], conditions]
            ).astype(np.float32)
            routes["project_raw_explicit_position_plus_conditions"] = fit_predict(
                raw_explicit_position,
                fit_y,
                train,
                test,
                PROJECT_PARAMS,
                fold,
                device,
            )
            raw_explicit_all = np.column_stack(
                [project["raw"], project["explicit_cet_5mec"], conditions]
            ).astype(np.float32)
            routes["project_raw_explicit_all_plus_conditions"] = fit_predict(
                raw_explicit_all,
                fit_y,
                train,
                test,
                PROJECT_PARAMS,
                fold,
                device,
            )
        project_a = fit_predict(
            project["branch_a"], fit_y, train, test, PROJECT_PARAMS, fold, device
        )
        project_b = fit_predict(
            project["branch_b"], fit_y, train, test, PROJECT_PARAMS, fold, device
        )
        routes["project_fixed_two_branch"] = 0.5 * (project_a + project_b)
        if oligoai is not None:
            oligo_prediction = fit_predict(
                oligoai, fit_y, train, test, PROJECT_PARAMS, fold, device
            )
            routes["oligoai_aso64"] = oligo_prediction
            oligo_descriptor = np.column_stack(
                [project["branch_b"], oligoai, experimental_conditions(frame)]
            ).astype(np.float32)
            routes["project_descriptor_plus_oligoai_aso64"] = fit_predict(
                oligo_descriptor, fit_y, train, test, PROJECT_PARAMS, fold, device
            )
            routes["late_project_two_branch_plus_oligoai_equal"] = (
                project_a + project_b + oligo_prediction
            ) / 3.0
        if project["branch_a_conditions"].shape[1] > project["branch_a"].shape[1]:
            conditions_a = fit_predict(
                project["branch_a_conditions"],
                fit_y,
                train,
                test,
                PROJECT_PARAMS,
                fold,
                device,
            )
            conditions_b = fit_predict(
                project["branch_b_conditions"],
                fit_y,
                train,
                test,
                PROJECT_PARAMS,
                fold,
                device,
            )
            routes["project_two_branch_plus_conditions"] = 0.5 * (
                conditions_a + conditions_b
            )
        else:
            routes["project_two_branch_plus_conditions"] = routes[
                "project_fixed_two_branch"
            ].copy()

        if explicit_cet_5mec:
            explicit_a = np.column_stack(
                [project["branch_a_explicit"], conditions]
            ).astype(np.float32)
            explicit_b = np.column_stack(
                [project["branch_b_explicit"], conditions]
            ).astype(np.float32)
            prediction_a = fit_predict(
                explicit_a, fit_y, train, test, PROJECT_PARAMS, fold, device
            )
            prediction_b = fit_predict(
                explicit_b, fit_y, train, test, PROJECT_PARAMS, fold, device
            )
            routes["project_two_branch_explicit_cet_5mec"] = 0.5 * (
                prediction_a + prediction_b
            )
            routed_a = np.column_stack(
                [project["branch_a_explicit_routed"], conditions]
            ).astype(np.float32)
            routed_b = np.column_stack(
                [project["branch_b_explicit_routed"], conditions]
            ).astype(np.float32)
            routed_prediction_a = fit_predict(
                routed_a, fit_y, train, test, PROJECT_PARAMS, fold, device
            )
            routed_prediction_b = fit_predict(
                routed_b, fit_y, train, test, PROJECT_PARAMS, fold, device
            )
            routes["project_two_branch_explicit_cet_5mec_routed"] = 0.5 * (
                routed_prediction_a + routed_prediction_b
            )
            # Separate quantity from location and local-context encodings. These
            # routes use the same base predictor and differ only in the named
            # cEt/5meC block added to one branch.
            explicit_blocks = {
                "quantity": (None, project["explicit_cet_5mec_quantity"]),
                "position": (project["explicit_cet_5mec_position"], None),
                "region": (None, project["explicit_cet_5mec_region"]),
                "architecture": (None, project["explicit_cet_5mec_architecture"]),
            }
            for block_name, (local_block, summary_block) in explicit_blocks.items():
                block_a = project["branch_a"]
                block_b = project["branch_b"]
                if local_block is not None:
                    block_a = np.column_stack([block_a, local_block]).astype(np.float32)
                if summary_block is not None:
                    block_b = np.column_stack([block_b, summary_block]).astype(np.float32)
                block_a = np.column_stack([block_a, conditions]).astype(np.float32)
                block_b = np.column_stack([block_b, conditions]).astype(np.float32)
                block_prediction_a = fit_predict(
                    block_a, fit_y, train, test, PROJECT_PARAMS, fold, device
                )
                block_prediction_b = fit_predict(
                    block_b, fit_y, train, test, PROJECT_PARAMS, fold, device
                )
                routes[f"project_two_branch_explicit_{block_name}"] = 0.5 * (
                    block_prediction_a + block_prediction_b
                )

            # Leave-one-subblock-out routes test whether a block adds unique
            # predictive value once the other cEt/5meC encodings are present.
            for omitted in explicit_blocks:
                kept_local = [
                    block[0]
                    for name, block in explicit_blocks.items()
                    if name != omitted and block[0] is not None
                ]
                kept_summary = [
                    block[1]
                    for name, block in explicit_blocks.items()
                    if name != omitted and block[1] is not None
                ]
                block_a = np.column_stack(
                    [project["branch_a"], *kept_local, conditions]
                ).astype(np.float32)
                block_b = np.column_stack(
                    [project["branch_b"], *kept_summary, conditions]
                ).astype(np.float32)
                block_prediction_a = fit_predict(
                    block_a, fit_y, train, test, PROJECT_PARAMS, fold, device
                )
                block_prediction_b = fit_predict(
                    block_b, fit_y, train, test, PROJECT_PARAMS, fold, device
                )
                routes[f"project_two_branch_explicit_drop_{omitted}"] = 0.5 * (
                    block_prediction_a + block_prediction_b
                )

        if use_log_target:
            routes = {
                route: np.maximum(np.expm1(prediction), 0.0)
                for route, prediction in routes.items()
            }

        train_groups = set(groups[train])
        test_groups = set(groups[test])
        train_helms = set(frame.iloc[train]["HELM Annotation"])
        test_helms = set(frame.iloc[test]["HELM Annotation"])
        if train_groups & test_groups or train_helms & test_helms:
            raise RuntimeError(f"{task}/fold{fold}: grouped split overlap")
        for route, prediction in routes.items():
            metrics = regression_metrics(y[test], prediction)
            log_metrics = regression_metrics(
                np.log1p(y[test]), np.log1p(np.maximum(prediction, 0.0))
            )
            fold_rows.append(
                {
                    "task": task,
                    "split_mode": split_mode,
                    "fold": fold,
                    "route": route,
                    "test_rows": len(test),
                    "train_groups": len(train_groups),
                    "test_groups": len(test_groups),
                    "group_overlap": 0,
                    "helm_overlap": 0,
                    "target_transform": "log1p" if use_log_target else "identity",
                    "log1p_r2": log_metrics["r2"],
                    "log1p_rmse": log_metrics["rmse"],
                    **metrics,
                }
            )
            predictions.extend(
                {
                    "task": task,
                    "split_mode": split_mode,
                    "fold": fold,
                    "row_index": int(frame.index[row]),
                    "group": groups[row],
                    "route": route,
                    "y": y[row],
                    "prediction": prediction[position],
                }
                for position, row in enumerate(test)
            )

    audit = input_audit(task, frame, project["row_audit"])
    audit["author_feature_dimension"] = int(author.shape[1])
    audit["project_raw_dimension"] = int(project["raw"].shape[1])
    audit["project_branch_a_dimension"] = int(project["branch_a"].shape[1])
    audit["project_branch_b_dimension"] = int(project["branch_b"].shape[1])
    audit["explicit_cet_5mec_dimension"] = int(
        project["explicit_cet_5mec"].shape[1]
    )
    audit["explicit_cet_5mec_subblock_dimensions"] = {
        name: int(project[f"explicit_cet_5mec_{name}"].shape[1])
        for name in ("quantity", "position", "region", "architecture")
    }
    audit["chemistry_complete_raw_dimensions"] = {
        "legacy_raw": int(project["raw"].shape[1]),
        "position_expanded_raw": int(
            project["raw"].shape[1]
            + project["explicit_cet_5mec_position"].shape[1]
        ),
        "all_explicit_descriptor_raw": int(
            project["raw"].shape[1] + project["explicit_cet_5mec"].shape[1]
        ),
        "experimental_conditions": int(experimental_conditions(frame).shape[1]),
    }
    audit["split_mode"] = split_mode
    audit["validation_groups"] = int(pd.Series(groups).nunique())
    audit["similarity90_audit"] = similarity_audit
    audit["elapsed_seconds"] = round(time.time() - started, 3)
    return pd.DataFrame(fold_rows), pd.DataFrame(predictions), audit


def summarize(folds: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, split_mode, route), group in folds.groupby(
        ["task", "split_mode", "route"], sort=False
    ):
        predictions = oof[
            oof["task"].eq(task)
            & oof["split_mode"].eq(split_mode)
            & oof["route"].eq(route)
        ]
        pooled = regression_metrics(
            predictions["y"].to_numpy(), predictions["prediction"].to_numpy()
        )
        source_mean, source_sd = (
            SOURCE_SPEARMAN[task] if split_mode == "patent" else (np.nan, np.nan)
        )
        rows.append(
            {
                "task": task,
                "split_mode": split_mode,
                "route": route,
                "folds": group["fold"].nunique(),
                "rows": len(predictions),
                "fold_mean_spearman": group["spearman"].mean(),
                "fold_sd_spearman": group["spearman"].std(ddof=0),
                "fold_mean_r2": group["r2"].mean(),
                "fold_mean_rmse": group["rmse"].mean(),
                "fold_mean_rmse_over_y_std": group["rmse_over_y_std"].mean(),
                "fold_mean_log1p_r2": group["log1p_r2"].mean(),
                "fold_mean_log1p_rmse": group["log1p_rmse"].mean(),
                "pooled_spearman": pooled["spearman"],
                "pooled_r2": pooled["r2"],
                "pooled_rmse": pooled["rmse"],
                "pooled_rmse_over_y_std": pooled["rmse_over_y_std"],
                "source_reported_xgb_spearman": source_mean,
                "source_reported_xgb_spearman_sd": source_sd,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--log-hepatic-target", action="store_true")
    parser.add_argument("--explicit-cet-5mec", action="store_true")
    parser.add_argument(
        "--split-mode", choices=("patent", "similarity90"), default="patent"
    )
    parser.add_argument("--oligoai-blocks", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fold_frames = []
    oof_frames = []
    audits = []
    for task in args.tasks:
        folds, oof, audit = run_task(
            task,
            args.out_dir,
            device=args.device,
            probe=args.probe,
            log_hepatic_target=args.log_hepatic_target,
            explicit_cet_5mec=args.explicit_cet_5mec,
            split_mode=args.split_mode,
            oligoai_blocks=args.oligoai_blocks,
        )
        fold_frames.append(folds)
        oof_frames.append(oof)
        audits.append(audit)
        print(
            f"{task}: rows={audit['rows']} groups={audit['patent_groups']} "
            f"elapsed={audit['elapsed_seconds']}s",
            flush=True,
        )

    folds = pd.concat(fold_frames, ignore_index=True)
    oof = pd.concat(oof_frames, ignore_index=True)
    summary = summarize(folds, oof)
    suffix = "probe" if args.probe else "full"
    if args.log_hepatic_target:
        suffix += "_log_hepatic"
    if args.explicit_cet_5mec:
        suffix += "_explicit_cet_5mec"
    if args.split_mode == "similarity90":
        suffix += "_similarity90"
    folds.to_csv(args.out_dir / f"fold_metrics_{suffix}.csv", index=False)
    oof.to_csv(
        args.out_dir / f"oof_predictions_{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    summary.to_csv(args.out_dir / f"summary_{suffix}.csv", index=False)
    (args.out_dir / f"input_audit_{suffix}.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "aso_atlas2_project_xgb_comparison_v1",
        "status": "PROBE" if args.probe else "COMPLETE",
        "tasks": args.tasks,
        "device": args.device,
        "log_hepatic_target": args.log_hepatic_target,
        "explicit_cet_5mec": args.explicit_cet_5mec,
        "explicit_chemistry_schema_version": EXPLICIT_CHEMISTRY_SCHEMA_VERSION,
        "split_mode": args.split_mode,
        "validation": (
            "five-fold GroupKFold by earliest-patent assignment per exact HELM"
            if args.split_mode == "patent"
            else "five-fold GroupKFold by 0.90 canonical-sequence identity connected components"
        ),
        "author_params": AUTHOR_PARAMS,
        "author_execution": (
            "XGBRegressor with only n_estimators=500 and max_depth=10, matching "
            "the public OligoGym wrapper; no project GPU/tree_method overrides"
        ),
        "project_params": PROJECT_PARAMS,
        "input_files": {
            name: {
                "path": str(DATA_DIR / name),
                "sha256": sha256(DATA_DIR / name),
            }
            for name in ("hepatotoxicity.parquet", "neurotoxicity.parquet")
        },
        "claim_boundary": (
            "Retrospective same-data comparison. Source-reported ASO Atlas scores "
            "remain authoritative for the published recipe; this run adds project "
            "representations and absolute-fit metrics on the archived data version."
        ),
    }
    (args.out_dir / f"manifest_{suffix}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
