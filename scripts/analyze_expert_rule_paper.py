#!/usr/bin/env python3
"""Primary source comparison and controlled component accounting for paper v3."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/expert_rule_scale_v1/expert_rule_scale/oof_predictions.csv.gz"
IDENTITY_ENDPOINT = ROOT / "outputs/expert_rule_scale_v1/analysis/policy_endpoint_metrics.csv"
CAPACITY_ENDPOINT = ROOT / "outputs/expert_rule_capacity_v1/analysis/policy_endpoint_metrics.csv"
CAPACITY_TESTS = ROOT / "outputs/expert_rule_capacity_v1/analysis/selector_vs_current_paired_tests.csv"
FACTORIAL = ROOT / "outputs/integrated_auto_system_v1/report/endpoint_stage_metrics.csv"
OLIGO_SOURCE = ROOT / "outputs/oligogym_neurips2025_same_split/same_split_predictions.csv"
ATLAS_SOURCE = ROOT / "outputs/aso_atlas2_four_stage_factorial_v1/full/oof_predictions.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
    y = frame.y.to_numpy(float)
    prediction = frame[column].to_numpy(float)
    y_std = float(np.std(y, ddof=0))
    rmse = float(np.sqrt(mean_squared_error(y, prediction)))
    rho = float(spearmanr(y, prediction).statistic)
    return {
        "spearman": rho if np.isfinite(rho) else -1.0,
        "raw_r2": float(r2_score(y, prediction)) if len(y) > 1 and np.ptp(y) > 0 else -1.0,
        "rmse": rmse, "y_std": y_std,
        "rmse_over_y_std": rmse / max(y_std, np.finfo(float).eps),
    }


def source_predictions() -> pd.DataFrame:
    oligo = pd.read_csv(OLIGO_SOURCE)
    oligo = oligo[
        oligo.route.eq("oligogym_published_overall") & oligo.split_mode.eq("nucleobase_holdout")
    ].rename(columns={"project_dataset": "dataset", "split_id": "outer_partition", "prediction": "pred_source"})
    atlas = pd.read_csv(ATLAS_SOURCE).rename(
        columns={"task": "dataset", "outer_fold": "outer_partition", "pred_author_ohe_author_xgb": "pred_source"}
    )
    columns = ["dataset", "outer_partition", "row_index", "y", "pred_source"]
    return pd.concat([oligo[columns], atlas[columns]], ignore_index=True)


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    null = [abs(float(np.mean(values * signs))) for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def bootstrap(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(sampled, [0.025, 0.975]))


def holm(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted_ranked = np.maximum.accumulate(np.minimum(1.0, ranked * (len(array) - np.arange(len(array)))))
    adjusted = np.empty_like(array)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/expert_rule_paper_v3_analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=200_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "outer_partition", "row_index"]
    final = pd.read_csv(FINAL).rename(columns={"pred_policy": "pred_final"})
    source = source_predictions()
    if final.duplicated(keys).any() or source.duplicated(keys).any():
        raise RuntimeError("source/final comparison keys are not one-to-one")
    final_keys = pd.MultiIndex.from_frame(final[keys])
    source_keys = pd.MultiIndex.from_frame(source[keys])
    same_keys = len(final_keys.difference(source_keys)) == 0 and len(source_keys.difference(final_keys)) == 0
    if len(final) != 14_450 or len(source) != 14_450 or not same_keys:
        missing_in_source = int((~final_keys.isin(source_keys)).sum())
        missing_in_final = int((~source_keys.isin(final_keys)).sum())
        raise RuntimeError(
            "source/final outer partitions are not identical: "
            f"missing_in_source={missing_in_source}, missing_in_final={missing_in_final}"
        )
    paired = final.merge(source, on=keys, suffixes=("_final", "_source"), validate="one_to_one")
    if len(paired) != 14_450 or not np.allclose(paired.y_final, paired.y_source, atol=1e-7, rtol=0):
        raise RuntimeError("source/final pairing did not reconcile to 14,450 rows")
    paired["absolute_label_difference"] = (paired.y_final - paired.y_source).abs()
    paired = paired.rename(columns={"y_final": "y"})
    pairing_audit = paired.groupby(["dataset", "outer_partition"], as_index=False).agg(
        paired_rows=("row_index", "size"),
        unique_rows=("row_index", "nunique"),
        maximum_absolute_label_difference=("absolute_label_difference", "max"),
    )
    pairing_audit["same_endpoint"] = True
    pairing_audit["same_outer_partition"] = True
    pairing_audit["same_test_rows"] = True
    pairing_audit["same_labels"] = True
    fold_rows = []
    for (dataset, fold), part in paired.groupby(["dataset", "outer_partition"], sort=True):
        final_metric = metrics(part, "pred_final")
        source_metric = metrics(part, "pred_source")
        row = {"dataset": dataset, "outer_partition": int(fold), "rows": len(part)}
        for metric in ("spearman", "raw_r2", "rmse", "rmse_over_y_std"):
            row[f"source_{metric}"] = source_metric[metric]
            row[f"final_{metric}"] = final_metric[metric]
            row[f"delta_{metric}"] = final_metric[metric] - source_metric[metric]
        fold_rows.append(row)
    fold = pd.DataFrame(fold_rows)
    endpoint = fold.groupby("dataset", as_index=False).agg(
        **{f"source_{metric}": (f"source_{metric}", "mean") for metric in ("spearman", "raw_r2", "rmse", "rmse_over_y_std")},
        **{f"final_{metric}": (f"final_{metric}", "mean") for metric in ("spearman", "raw_r2", "rmse", "rmse_over_y_std")},
        **{f"delta_{metric}": (f"delta_{metric}", "mean") for metric in ("spearman", "raw_r2", "rmse", "rmse_over_y_std")},
        outer_partitions=("outer_partition", "nunique"), paired_rows=("rows", "sum"),
    )
    system_rows = []
    for offset, metric in enumerate(("spearman", "raw_r2", "rmse_over_y_std")):
        delta = endpoint[f"delta_{metric}"].to_numpy(float)
        improvement = -delta if metric == "rmse_over_y_std" else delta
        low, high = bootstrap(improvement, 20260831 + offset, args.bootstrap_replicates)
        system_rows.append({
            "metric": metric,
            "improvement_direction": "final-source" if metric != "rmse_over_y_std" else "source-final",
            "source_macro_mean": float(endpoint[f"source_{metric}"].mean()),
            "final_macro_mean": float(endpoint[f"final_{metric}"].mean()),
            "mean_improvement": float(improvement.mean()), "median_improvement": float(np.median(improvement)),
            "endpoint_wins": int(np.count_nonzero(improvement > 0)), "endpoints": len(improvement),
            "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
            "exact_two_sided_sign_flip_p": exact_sign_flip(improvement) if metric in {"spearman", "raw_r2"} else np.nan,
        })
    system = pd.DataFrame(system_rows)
    primary = system.metric.isin(["spearman", "raw_r2"])
    system.loc[primary, "holm_two_primary_metrics_p"] = holm(system.loc[primary, "exact_two_sided_sign_flip_p"].tolist())

    factorial = pd.read_csv(FACTORIAL)
    wide = factorial.pivot(index="dataset", columns="stage", values=["spearman", "raw_r2"])
    component_rows = []
    for dataset in wide.index:
        for metric in ("spearman", "raw_r2"):
            component_rows.extend([
                {"dataset": dataset, "component": "model_recipe", "metric": metric,
                 "baseline": "source_native", "candidate": "project_xgb",
                 "baseline_value": wide.loc[dataset, (metric, "source_native")],
                 "candidate_value": wide.loc[dataset, (metric, "project_xgb")]},
                {"dataset": dataset, "component": "expert_descriptor_fusion", "metric": metric,
                 "baseline": "project_xgb", "candidate": "fixed_fusion",
                 "baseline_value": wide.loc[dataset, (metric, "project_xgb")],
                 "candidate_value": wide.loc[dataset, (metric, "fixed_fusion")]},
            ])
    component = pd.DataFrame(component_rows)
    identity = pd.read_csv(IDENTITY_ENDPOINT)
    identity = identity[identity.policy.isin(["fixed_expert_identity", "expert_rule_scale"])]
    iw = identity.pivot(index="dataset", columns="policy", values=["spearman", "raw_r2", "rmse_over_y_std"])
    for dataset in iw.index:
        for metric in ("spearman", "raw_r2", "rmse_over_y_std"):
            component.loc[len(component)] = {
                "dataset": dataset, "component": "rule_log1p", "metric": metric,
                "baseline": "fixed_expert_identity", "candidate": "expert_rule_scale",
                "baseline_value": iw.loc[dataset, (metric, "fixed_expert_identity")],
                "candidate_value": iw.loc[dataset, (metric, "expert_rule_scale")],
            }
    component["delta"] = component.candidate_value - component.baseline_value
    component["improvement"] = np.where(component.metric.eq("rmse_over_y_std"), -component.delta, component.delta)
    component_macro = component.groupby(["component", "metric", "baseline", "candidate"], as_index=False).agg(
        baseline_macro=("baseline_value", "mean"), candidate_macro=("candidate_value", "mean"),
        mean_improvement=("improvement", "mean"), median_improvement=("improvement", "median"),
        endpoint_wins=("improvement", lambda values: int(np.count_nonzero(np.asarray(values) > 0))),
        endpoints=("dataset", "nunique"),
    )
    capacity = pd.read_csv(CAPACITY_ENDPOINT)
    capacity_macro = capacity.groupby("policy", as_index=False).agg(
        spearman=("spearman", "mean"), raw_r2=("raw_r2", "mean"), rmse_over_y_std=("rmse_over_y_std", "mean")
    )
    capacity_tests = pd.read_csv(CAPACITY_TESTS)
    paired.to_csv(args.out_dir / "source_vs_final_paired_predictions.csv.gz", index=False, compression="gzip")
    fold.to_csv(args.out_dir / "source_vs_final_fold_metrics.csv", index=False)
    endpoint.to_csv(args.out_dir / "source_vs_final_endpoint_effects.csv", index=False)
    system.to_csv(args.out_dir / "source_vs_final_system_tests.csv", index=False)
    pairing_audit.to_csv(args.out_dir / "source_vs_final_pairing_audit.csv", index=False)
    component.to_csv(args.out_dir / "component_endpoint_effects.csv", index=False)
    component_macro.to_csv(args.out_dir / "component_macro_effects.csv", index=False)
    capacity_macro.to_csv(args.out_dir / "capacity_macro_metrics.csv", index=False)
    capacity_tests.to_csv(args.out_dir / "capacity_paired_tests.csv", index=False)
    manifest = {
        "schema_version": "expert_rule_paper_analysis_v1",
        "primary_system": "fixed complete expert A/B descriptors + current_balanced XGBoost + outer-train rule scale",
        "paired_rows": len(paired), "outer_partitions": 45,
        "source_comparator": "same-partition endpoint-specific source replay",
        "pairing_contract": (
            "same endpoint classification, same grouped outer-partition identifier, "
            "same row_index, and identical observed label for every paired test prediction"
        ),
        "component_boundary": (
            "model and descriptor effects use the controlled source-XGB factorial; log1p uses strict expert identity-vs-rule. "
            "They are local sequential effects under different target/model baselines and are not additive shares of the main source-to-final contrast."
        ),
        "source_hashes": {"oligogym": sha256(OLIGO_SOURCE), "atlas": sha256(ATLAS_SOURCE)},
        "final_oof_sha256": sha256(FINAL),
        "claim_boundary": "retrospective grouped generalization to training-unseen assigned groups; not external confirmation",
    }
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(system.to_string(index=False))
    print(component_macro.to_string(index=False))


if __name__ == "__main__":
    main()
