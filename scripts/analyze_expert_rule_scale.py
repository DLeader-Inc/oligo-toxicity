#!/usr/bin/env python3
"""Audit the post-review expert A/B + current_balanced scale policies."""

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
BASE = ROOT / "outputs/mechanism_informed_automl_nested_v1"
DEFAULT_ROOT = ROOT / "outputs/expert_rule_scale_v1"
POLICY_PATHS = {
    "fixed_expert_identity": BASE / "fixed_expert",
    "route_scale": BASE / "route_scale",
    "full_automl": BASE / "full_automl",
    "expert_rule_scale": DEFAULT_ROOT / "expert_rule_scale",
    "expert_nested_scale": DEFAULT_ROOT / "expert_nested_scale",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame.y.to_numpy(float)
    prediction = frame.pred_policy.to_numpy(float)
    y_std = float(np.std(y, ddof=0))
    rmse = float(np.sqrt(mean_squared_error(y, prediction)))
    rho = float(spearmanr(y, prediction).statistic)
    return {
        "spearman": rho if np.isfinite(rho) else -1.0,
        "raw_r2": float(r2_score(y, prediction)) if np.ptp(y) > 0 else -1.0,
        "rmse": rmse, "y_std": y_std,
        "rmse_over_y_std": rmse / max(y_std, np.finfo(float).eps),
    }


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    null = (
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    )
    return sum(value >= observed - 1e-15 for value in null) / (2 ** len(values))


def bootstrap(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


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
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=200_000)
    args = parser.parse_args()
    paths = dict(POLICY_PATHS)
    paths["expert_rule_scale"] = args.root / "expert_rule_scale"
    paths["expert_nested_scale"] = args.root / "expert_nested_scale"
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "outer_partition", "row_index"]
    reference = None
    frames = []
    hashes = {}
    selection_rows = []
    for policy, directory in paths.items():
        oof_path = directory / "oof_predictions.csv.gz"
        frame = pd.read_csv(oof_path).sort_values(keys).reset_index(drop=True)
        if len(frame) != 14_450 or frame[keys].drop_duplicates().shape[0] != 14_450:
            raise RuntimeError(f"{policy}: OOF keys do not reconcile to 14,450")
        current = frame[[*keys, "y"]]
        if reference is None:
            reference = current
        elif not reference[keys].equals(current[keys]) or not np.allclose(reference.y, current.y, atol=1e-7, rtol=0):
            raise RuntimeError(f"{policy}: outer rows or labels differ")
        frame["comparison_policy"] = policy
        frames.append(frame)
        hashes[policy] = sha256(oof_path)
        for selection_path in sorted((directory / "fold_checkpoints").glob("*__selection.json")):
            selection = json.loads(selection_path.read_text())
            selection_rows.append({
                "comparison_policy": policy, "dataset": selection["dataset"],
                "outer_partition": selection["outer_partition"],
                "route": selection["selected_route"], "capacity": selection["selected_candidate"],
                "transform": selection["selected_transform"],
                "scale_rule_kind": selection["scale_rule"]["kind"],
                "scale_rule_skewness": selection["scale_rule"].get("fisher_pearson_skewness"),
                "reused_compatible_outer_artifacts": bool(selection.get("reused_compatible_outer_artifacts_from")),
                "selection_uses_outer_test": selection["selection_uses_outer_test"],
            })
    all_predictions = pd.concat(frames, ignore_index=True)
    fold_rows = []
    for (policy, dataset, fold), part in all_predictions.groupby(
        ["comparison_policy", "dataset", "outer_partition"], sort=True
    ):
        fold_rows.append({
            "policy": policy, "dataset": dataset, "outer_partition": int(fold),
            "rows": len(part), **metrics(part),
        })
    fold_metrics = pd.DataFrame(fold_rows)
    endpoint = fold_metrics.groupby(["policy", "dataset"], as_index=False).agg(
        spearman=("spearman", "mean"), raw_r2=("raw_r2", "mean"), rmse=("rmse", "mean"),
        y_std=("y_std", "mean"), rmse_over_y_std=("rmse_over_y_std", "mean"),
        outer_partitions=("outer_partition", "nunique"), oof_rows=("rows", "sum"),
    )
    macro = endpoint.groupby("policy", as_index=False).agg(
        spearman=("spearman", "mean"), raw_r2=("raw_r2", "mean"),
        rmse=("rmse", "mean"), rmse_over_y_std=("rmse_over_y_std", "mean"),
    )
    comparisons = []
    rule = endpoint[endpoint.policy.eq("expert_rule_scale")].set_index("dataset")
    for comparator in ("fixed_expert_identity", "route_scale", "full_automl"):
        other = endpoint[endpoint.policy.eq(comparator)].set_index("dataset").loc[rule.index]
        for offset, metric in enumerate(("spearman", "raw_r2", "rmse_over_y_std")):
            raw = rule[metric].to_numpy(float) - other[metric].to_numpy(float)
            improvement = -raw if metric == "rmse_over_y_std" else raw
            low, high = bootstrap(improvement, 20260830 + len(comparisons) + offset, args.bootstrap_replicates)
            comparisons.append({
                "candidate": "expert_rule_scale", "comparator": comparator, "metric": metric,
                "improvement_direction": "rule-comparator" if metric != "rmse_over_y_std" else "comparator-rule",
                "mean_improvement": float(improvement.mean()),
                "median_improvement": float(np.median(improvement)),
                "endpoint_wins": int(np.count_nonzero(improvement > 0)), "endpoints": len(improvement),
                "endpoint_bootstrap_ci95_low": low, "endpoint_bootstrap_ci95_high": high,
                "exact_two_sided_sign_flip_p": exact_sign_flip(improvement) if metric in {"spearman", "raw_r2"} else np.nan,
            })
    comparison = pd.DataFrame(comparisons)
    primary = comparison.metric.isin(["spearman", "raw_r2"])
    comparison.loc[primary, "holm_across_three_comparators_two_metrics_p"] = holm(
        comparison.loc[primary, "exact_two_sided_sign_flip_p"].tolist()
    )
    selections = pd.DataFrame(selection_rows)
    rule_sel = selections[selections.comparison_policy.eq("expert_rule_scale")].sort_values(["dataset", "outer_partition"])
    nested_sel = selections[selections.comparison_policy.eq("expert_nested_scale")].sort_values(["dataset", "outer_partition"])
    scale_compare = rule_sel.merge(
        nested_sel, on=["dataset", "outer_partition"], suffixes=("_rule", "_nested"), validate="one_to_one"
    )
    scale_compare["same_transform"] = scale_compare.transform_rule.eq(scale_compare.transform_nested)
    fold_metrics.to_csv(out / "policy_fold_metrics.csv", index=False)
    endpoint.to_csv(out / "policy_endpoint_metrics.csv", index=False)
    macro.to_csv(out / "policy_macro_metrics.csv", index=False)
    comparison.to_csv(out / "expert_rule_scale_paired_system_tests.csv", index=False)
    selections.to_csv(out / "selection_audit.csv", index=False)
    scale_compare.to_csv(out / "rule_vs_nested_scale_decisions.csv", index=False)
    manifest = {
        "schema_version": "expert_rule_scale_post_audit_analysis_v1",
        "policies": list(paths), "outer_rows_per_policy": 14_450,
        "outer_endpoint_partitions_per_policy": 45,
        "inference_unit": "nine endpoints after averaging five paired outer-fold differences",
        "test": "two-sided exhaustive 2^9 endpoint sign-flip for Spearman and raw-scale R2",
        "bootstrap": {"unit": "endpoint", "replicates": args.bootstrap_replicates},
        "multiplicity": "Holm across three comparators x two primary metrics",
        "oof_sha256": hashes,
        "development_timing_boundary": (
            "expert_rule_scale was specified after reviewing earlier outer-policy results; this is a strict grouped "
            "retrospective development estimate, not independent confirmation"
        ),
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(macro.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
