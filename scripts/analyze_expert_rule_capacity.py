#!/usr/bin/env python3
"""Analyze expert-descriptor, rule-scale XGBoost capacity policies."""

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
CURRENT = ROOT / "outputs/expert_rule_scale_v1/expert_rule_scale"
DEFAULT_ROOT = ROOT / "outputs/expert_rule_capacity_v1"


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
        "rmse": rmse,
        "y_std": y_std,
        "rmse_over_y_std": rmse / max(y_std, np.finfo(float).eps),
    }


def sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    null = [abs(float(np.mean(values * signs))) for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


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
    paths = {
        "fixed_ultra": args.root / "fixed_ultra",
        "fixed_compact": args.root / "fixed_compact",
        "fixed_conservative": args.root / "fixed_conservative",
        "fixed_current_balanced": CURRENT,
        "fixed_deep": args.root / "fixed_deep",
        "capacity_inner_best": args.root / "capacity_inner_best",
        "capacity_guarded": args.root / "capacity_guarded",
    }
    output = args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "outer_partition", "row_index"]
    reference: pd.DataFrame | None = None
    prediction_frames = []
    selection_rows = []
    hashes = {}
    for policy, directory in paths.items():
        oof_path = directory / "oof_predictions.csv.gz"
        frame = pd.read_csv(oof_path).sort_values(keys).reset_index(drop=True)
        if len(frame) != 14_450 or frame[keys].drop_duplicates().shape[0] != 14_450:
            raise RuntimeError(f"{policy}: expected 14,450 unique OOF rows")
        current = frame[[*keys, "y"]]
        if reference is None:
            reference = current
        elif not reference[keys].equals(current[keys]) or not np.allclose(reference.y, current.y, atol=1e-7, rtol=0):
            raise RuntimeError(f"{policy}: outer rows or labels differ")
        frame["comparison_policy"] = policy
        prediction_frames.append(frame)
        hashes[policy] = sha256(oof_path)
        checkpoints = sorted((directory / "fold_checkpoints").glob("*__selection.json"))
        if len(checkpoints) != 45:
            raise RuntimeError(f"{policy}: expected 45 selection checkpoints, got {len(checkpoints)}")
        for path in checkpoints:
            payload = json.loads(path.read_text())
            selection_rows.append({
                "policy": policy,
                "dataset": payload["dataset"],
                "outer_partition": payload["outer_partition"],
                "capacity": payload["selected_candidate"],
                "transform": payload["selected_transform"],
                "route": payload["selected_route"],
                "outer_group_overlap": payload["outer_group_overlap"],
                "selection_uses_outer_test": payload["selection_uses_outer_test"],
                "reused_outer": bool(payload.get("reused_compatible_outer_artifacts_from")),
                "selection_reason": payload["selection_audit"]["selected_reason"],
            })
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_rows = []
    for (policy, dataset, fold), part in predictions.groupby(["comparison_policy", "dataset", "outer_partition"]):
        fold_rows.append({"policy": policy, "dataset": dataset, "outer_partition": fold, "rows": len(part), **metrics(part)})
    fold_metrics = pd.DataFrame(fold_rows)
    endpoint = fold_metrics.groupby(["policy", "dataset"], as_index=False).agg(
        spearman=("spearman", "mean"), raw_r2=("raw_r2", "mean"), rmse=("rmse", "mean"),
        rmse_over_y_std=("rmse_over_y_std", "mean"), outer_partitions=("outer_partition", "nunique"),
        oof_rows=("rows", "sum"),
    )
    macro = endpoint.groupby("policy", as_index=False).agg(
        spearman=("spearman", "mean"), raw_r2=("raw_r2", "mean"),
        rmse=("rmse", "mean"), rmse_over_y_std=("rmse_over_y_std", "mean"),
    )
    comparisons = []
    incumbent = endpoint[endpoint.policy.eq("fixed_current_balanced")].set_index("dataset")
    for candidate in (
        "fixed_ultra", "fixed_compact", "fixed_conservative", "fixed_deep",
        "capacity_inner_best", "capacity_guarded",
    ):
        current = endpoint[endpoint.policy.eq(candidate)].set_index("dataset").loc[incumbent.index]
        for metric in ("spearman", "raw_r2", "rmse_over_y_std"):
            delta = current[metric].to_numpy(float) - incumbent[metric].to_numpy(float)
            improvement = -delta if metric == "rmse_over_y_std" else delta
            low, high = bootstrap(improvement, 20260830 + len(comparisons), args.bootstrap_replicates)
            comparisons.append({
                "candidate": candidate, "comparator": "fixed_current_balanced", "metric": metric,
                "mean_improvement": float(improvement.mean()), "median_improvement": float(np.median(improvement)),
                "endpoint_wins": int(np.count_nonzero(improvement > 0)), "endpoint_ties": int(np.count_nonzero(improvement == 0)),
                "endpoints": len(improvement), "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                "exact_two_sided_sign_flip_p": sign_flip(improvement) if metric in {"spearman", "raw_r2"} else np.nan,
            })
    comparison = pd.DataFrame(comparisons)
    primary = comparison.metric.isin(["spearman", "raw_r2"])
    comparison.loc[primary, "holm_twelve_primary_tests_p"] = holm(comparison.loc[primary, "exact_two_sided_sign_flip_p"].tolist())
    selections = pd.DataFrame(selection_rows)
    capacity_frequency = selections.groupby(["policy", "capacity"], as_index=False).size().rename(columns={"size": "outer_partitions"})
    endpoint_pivot = endpoint.pivot(index="dataset", columns="policy", values=["spearman", "raw_r2", "rmse_over_y_std"])
    endpoint_pivot.columns = [f"{metric}__{policy}" for metric, policy in endpoint_pivot.columns]
    endpoint_pivot.reset_index().to_csv(output / "endpoint_wide_comparison.csv", index=False)
    fold_metrics.to_csv(output / "policy_fold_metrics.csv", index=False)
    endpoint.to_csv(output / "policy_endpoint_metrics.csv", index=False)
    macro.to_csv(output / "policy_macro_metrics.csv", index=False)
    comparison.to_csv(output / "selector_vs_current_paired_tests.csv", index=False)
    selections.to_csv(output / "selection_audit.csv", index=False)
    capacity_frequency.to_csv(output / "capacity_selection_frequency.csv", index=False)
    manifest = {
        "schema_version": "expert_rule_capacity_analysis_v1",
        "outer_rows_per_policy": 14_450,
        "outer_partitions_per_policy": 45,
        "policies": list(paths),
        "inference_unit": "nine endpoints after averaging five paired outer-fold differences",
        "statistics": "two-sided exact 2^9 sign-flip; endpoint bootstrap; Holm across six candidates x two primary metrics",
        "bootstrap_replicates": args.bootstrap_replicates,
        "oof_sha256": hashes,
        "development_timing_boundary": (
            "capacity policies were specified after reviewing earlier outer-policy results; estimates are strict grouped "
            "retrospective development results, not independent confirmation"
        ),
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(macro.to_string(index=False))
    print(comparison.to_string(index=False))
    print(capacity_frequency.to_string(index=False))


if __name__ == "__main__":
    main()
