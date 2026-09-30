"""Audit helpers for the NeurIPS 2025 OligoGym benchmark.

The paper's Table 3 reports the best configuration within each model class,
selected by mean test-set Pearson correlation across five repeated
nucleobase holdouts. The public benchmark CSV contains one row per
configuration and holdout, so the published table can be reconstructed
without retraining the original 10,657-run grid.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

import pandas as pd


PAPER_DATASET_ORDER: Final[tuple[str, ...]] = (
    "OpenASO",
    "ASOptimizer",
    "TLR7",
    "TLR8",
    "Cytotox LNA",
    "Neurotox LNA",
    "Neurotox MOE",
    "siRNAmod",
    "Sherwood",
    "Ichihara",
    "Huesken",
    "Shmushkovich",
)

PAPER_MODEL_ORDER: Final[tuple[str, ...]] = (
    "Linear",
    "KNN",
    "RF",
    "XGB",
    "MLP",
    "CNN",
    "GRU",
    "Transformer",
)

DATASET_NAMES: Final[dict[str, str]] = {
    "openASO": "OpenASO",
    "asoptimizer_cleaned": "ASOptimizer",
    "immune_modulation_TLR7": "TLR7",
    "immune_modulation_TLR8": "TLR8",
    "cytotox_lna": "Cytotox LNA",
    "acute_neurotox_lna": "Neurotox LNA",
    "acute_neurotox_moe_cleaned": "Neurotox MOE",
    "siRNAmod": "siRNAmod",
    "sherwood": "Sherwood",
    "siRNA2": "Ichihara",
    "siRNA1": "Huesken",
    "siRNA3": "Shmushkovich",
}

MODEL_NAMES: Final[dict[str, str]] = {
    "LinearModel": "Linear",
    "NearestNeighborsModel": "KNN",
    "RandomForestModel": "RF",
    "XGBoostModel": "XGB",
    "MLP": "MLP",
    "CNN": "CNN",
    "GRU": "GRU",
    "Transformer": "Transformer",
}

TOXICITY_DATASETS: Final[tuple[str, ...]] = (
    "TLR7",
    "TLR8",
    "Cytotox LNA",
    "Neurotox LNA",
    "Neurotox MOE",
)

# Values transcribed from Table 3. They are used only as a rounded-value
# verification target; reconstructed unrounded metrics remain authoritative.
PAPER_TABLE3: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "OpenASO": dict(zip(PAPER_MODEL_ORDER, ((.29,.08),(.23,.06),(.29,.03),(.21,.02),(.23,.09),(.25,.08),(.19,.05),(.27,.11)))),
    "ASOptimizer": dict(zip(PAPER_MODEL_ORDER, ((.39,.04),(.45,.06),(.48,.03),(.40,.06),(.43,.08),(.48,.04),(.41,.03),(.44,.08)))),
    "TLR7": dict(zip(PAPER_MODEL_ORDER, ((.68,.07),(.56,.13),(.73,.10),(.66,.07),(.66,.11),(.62,.17),(.32,.13),(.72,.11)))),
    "TLR8": dict(zip(PAPER_MODEL_ORDER, ((.59,.08),(.41,.13),(.70,.06),(.58,.16),(.57,.10),(.25,.19),(.13,.13),(.26,.20)))),
    "Cytotox LNA": dict(zip(PAPER_MODEL_ORDER, ((.64,.16),(.64,.07),(.69,.09),(.64,.15),(.72,.11),(.70,.11),(.32,.23),(.72,.06)))),
    "Neurotox LNA": dict(zip(PAPER_MODEL_ORDER, ((.56,.09),(.49,.09),(.56,.08),(.50,.16),(.53,.08),(.57,.08),(.17,.22),(.53,.10)))),
    "Neurotox MOE": dict(zip(PAPER_MODEL_ORDER, ((.59,.24),(.49,.05),(.61,.11),(.58,.04),(.66,.07),(.67,.10),(.67,.12),(.70,.06)))),
    "siRNAmod": dict(zip(PAPER_MODEL_ORDER, ((.44,.11),(.50,.09),(.53,.09),(.42,.14),(.47,.07),(.49,.12),(.25,.09),(.49,.09)))),
    "Sherwood": dict(zip(PAPER_MODEL_ORDER, ((.70,.05),(.73,.03),(.82,.03),(.90,.03),(.74,.04),(.74,.04),(.12,.10),(.44,.09)))),
    "Ichihara": dict(zip(PAPER_MODEL_ORDER, ((.49,.09),(.38,.14),(.49,.06),(.34,.13),(.57,.12),(.48,.10),(.21,.10),(.35,.14)))),
    "Huesken": dict(zip(PAPER_MODEL_ORDER, ((.62,.01),(.44,.01),(.59,.01),(.51,.03),(.62,.01),(.63,.03),(.16,.04),(.51,.09)))),
    "Shmushkovich": dict(zip(PAPER_MODEL_ORDER, ((.20,.08),(.28,.16),(.42,.14),(.35,.07),(.35,.04),(.27,.10),(.12,.11),(.17,.20)))),
}

CONFIG_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "model",
    "featurizer",
    "featurizer_args",
    "model_args",
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_public_benchmarks(path: Path) -> pd.DataFrame:
    """Load and validate the public fold-level benchmark table."""

    frame = pd.read_csv(path)
    required = set(CONFIG_COLUMNS) | {
        "cross_validation",
        "test_fold",
        "test_pearson_correlation",
        "test_spearman_correlation",
        "test_r2_score",
        "test_root_mean_squared_error",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"benchmark CSV is missing columns: {sorted(missing)}")
    return frame


def reconstruct_table3(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the best PCC configuration per dataset and model class."""

    source = frame[frame["cross_validation"].eq("nucleobase")].copy()
    source = source[
        source["dataset"].isin(DATASET_NAMES)
        & source["model"].isin(MODEL_NAMES)
    ].copy()
    grouped = (
        source.groupby(list(CONFIG_COLUMNS), dropna=False)
        .agg(
            split_rows=("test_pearson_correlation", "size"),
            pcc_mean=("test_pearson_correlation", "mean"),
            pcc_std=("test_pearson_correlation", "std"),
            scc_mean=("test_spearman_correlation", "mean"),
            r2_mean=("test_r2_score", "mean"),
            rmse_mean=("test_root_mean_squared_error", "mean"),
        )
        .reset_index()
    )
    best_indices = grouped.groupby(["dataset", "model"])["pcc_mean"].idxmax()
    best = grouped.loc[best_indices].copy()
    best["paper_dataset"] = best["dataset"].map(DATASET_NAMES)
    best["paper_model"] = best["model"].map(MODEL_NAMES)
    dataset_order = {name: index for index, name in enumerate(PAPER_DATASET_ORDER)}
    model_order = {name: index for index, name in enumerate(PAPER_MODEL_ORDER)}
    best["dataset_order"] = best["paper_dataset"].map(dataset_order)
    best["model_order"] = best["paper_model"].map(model_order)
    best = best.sort_values(["dataset_order", "model_order"]).reset_index(drop=True)
    return best.drop(columns=["dataset_order", "model_order"])


def verify_table3(best: pd.DataFrame) -> pd.DataFrame:
    """Compare reconstructed values with the two-decimal published table."""

    records: list[dict[str, object]] = []
    indexed = best.set_index(["paper_dataset", "paper_model"])
    for dataset in PAPER_DATASET_ORDER:
        for model in PAPER_MODEL_ORDER:
            row = indexed.loc[(dataset, model)]
            paper_mean, paper_std = PAPER_TABLE3[dataset][model]
            reconstructed_mean = float(row["pcc_mean"])
            reconstructed_std = float(row["pcc_std"])
            mean_match = round(reconstructed_mean + 1e-12, 2) == paper_mean
            std_match = round(reconstructed_std + 1e-12, 2) == paper_std
            records.append(
                {
                    "paper_dataset": dataset,
                    "paper_model": model,
                    "paper_pcc_mean": paper_mean,
                    "paper_pcc_std": paper_std,
                    "reconstructed_pcc_mean": reconstructed_mean,
                    "reconstructed_pcc_std": reconstructed_std,
                    "mean_rounding_match": mean_match,
                    "std_rounding_match": std_match,
                    "cell_match": mean_match and std_match,
                }
            )
    return pd.DataFrame(records)


def table3_display(best: pd.DataFrame) -> pd.DataFrame:
    """Return the paper-shaped mean +/- standard deviation table."""

    tidy = best.copy()
    tidy["display"] = [
        f"{mean:.2f} +/- {std:.2f}"
        for mean, std in zip(tidy["pcc_mean"], tidy["pcc_std"], strict=True)
    ]
    display = tidy.pivot(index="paper_dataset", columns="paper_model", values="display")
    return display.reindex(index=PAPER_DATASET_ORDER, columns=PAPER_MODEL_ORDER)


def toxicity_best_models(best: pd.DataFrame) -> pd.DataFrame:
    """Return the best published model class for each toxicity endpoint."""

    toxicity = best[best["paper_dataset"].isin(TOXICITY_DATASETS)].copy()
    indices = toxicity.groupby("paper_dataset")["pcc_mean"].idxmax()
    selected = toxicity.loc[indices].copy()
    order = {name: index for index, name in enumerate(TOXICITY_DATASETS)}
    selected["order"] = selected["paper_dataset"].map(order)
    return selected.sort_values("order").drop(columns="order").reset_index(drop=True)

