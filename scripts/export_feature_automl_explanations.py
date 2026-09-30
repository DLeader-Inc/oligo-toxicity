#!/usr/bin/env python3
"""Export global held-out SHAP, shared descriptor concepts, and leakage-safe cases."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_aso_atlas2_project_representation import load_task  # noqa: E402
from scripts.benchmark_feature_automl_nested import ATLAS, EARLIER, _outer_partitions  # noqa: E402
from scripts.benchmark_registry_nine_endpoints import earlier_frame  # noqa: E402
from scripts.predict_feature_automl_system import _adopted_payload, _eligible_manifests  # noqa: E402
from src.explainability.automl_concepts import concept_for_feature, high_correlation_clusters  # noqa: E402
from src.features.automl_descriptor_routes import build_route_matrices  # noqa: E402
from src.features.automl_descriptor_routes import descriptor_family  # noqa: E402


DEFAULT_RUN = ROOT / "outputs/integrated_feature_automl_v2/formal"


def _case_helm(row: pd.Series, dataset: str) -> str:
    """Return the executable HELM used by the corresponding training route."""
    return str(row["x"] if dataset in ATLAS else row["HELM Annotation"])


def _update(stats: dict[tuple[str, str, str], list[float]], key: tuple[str, str, str], values: np.ndarray) -> None:
    row = stats.setdefault(key, [0.0, 0.0, 0.0, 0.0])
    row[0] += float(values.sum())
    row[1] += float(np.abs(values).sum())
    row[2] += float((values > 0).sum())
    row[3] += float(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-count", type=int, default=9)
    parser.add_argument("--global-only", action="store_true", help="Skip deployment-dependent case packet generation.")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Bounded resource probe: process only the first available endpoint/fold and skip case generation.",
    )
    args = parser.parse_args()
    out = args.output_dir or args.run_dir / "explanations"
    out.mkdir(parents=True, exist_ok=True)
    feature_stats: dict[tuple[str, str, str], list[float]] = {}
    concept_stats: dict[tuple[str, str, str], list[float]] = {}
    family_stats: dict[tuple[str, str, str], list[float]] = {}
    member_concept_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []
    member_prediction_rows: list[pd.DataFrame] = []
    cluster_context: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, list[str]]]] = {}
    datasets = (EARLIER + ATLAS)[:1] if args.probe else EARLIER + ATLAS
    for dataset in datasets:
        frame = earlier_frame(dataset) if dataset in EARLIER else load_task(dataset).reset_index(drop=True)
        matrices, schemas, _manifest = build_route_matrices(frame)
        cluster_context[dataset] = (frame, matrices, schemas)
        partitions = {int(row["partition"]): row for row in _outer_partitions(dataset, frame)}
        selection_paths = sorted((args.run_dir / "fold_checkpoints").glob(f"{dataset}__partition*__selection.json"))
        if args.probe:
            selection_paths = selection_paths[:1]
        for selection_path in selection_paths:
            selection = _adopted_payload(args.run_dir, json.loads(selection_path.read_text()))
            partition_number = int(selection["outer_partition"])
            test = np.asarray(partitions[partition_number]["test"], dtype=int)
            by_member: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for record in selection["models"]:
                by_member[int(record["member"])].append(record)
            for member, records in sorted(by_member.items()):
                combined: dict[str, np.ndarray] = {}
                expected = np.zeros(len(test), dtype=float)
                prediction = np.zeros(len(test), dtype=float)
                for record in records:
                    component = str(record["component"])
                    weight = float(record["weight"])
                    estimator = joblib.load(record["path"])
                    matrix = matrices[component][test]
                    values = estimator.get_booster().predict(xgb.DMatrix(matrix), pred_contribs=True)
                    prediction += weight * np.asarray(estimator.predict(matrix), dtype=float)
                    expected += weight * values[:, -1]
                    for index, name in enumerate(schemas[component]):
                        combined[name] = combined.get(name, np.zeros(len(test), dtype=float)) + weight * values[:, index]
                reconstructed = expected + sum(combined.values(), np.zeros(len(test), dtype=float))
                residual = np.abs(prediction - reconstructed)
                member_prediction_rows.append(pd.DataFrame({
                    "dataset": dataset, "outer_partition": partition_number, "member": member,
                    "row_index": test, "model_scale": selection["model_output_scale"],
                    "target_transform": selection["selected_transform"], "expected_value": expected,
                    "model_prediction": prediction, "signed_contribution_sum": reconstructed - expected,
                    "reconstructed_prediction": reconstructed,
                    "additivity_residual": prediction - reconstructed,
                }))
                reconciliation_rows.append({
                    "dataset": dataset, "outer_partition": partition_number, "member": member,
                    "model_scale": selection["model_output_scale"], "heldout_rows": len(test),
                    "max_additivity_residual": float(residual.max()),
                    "mean_additivity_residual": float(residual.mean()),
                })
                concepts: dict[str, np.ndarray] = {}
                families: dict[str, np.ndarray] = {}
                for name, values in combined.items():
                    _update(feature_stats, (dataset, selection["model_output_scale"], name), values)
                    family = descriptor_family(name)
                    families[family] = families.get(family, np.zeros(len(test), dtype=float)) + values
                    concept = concept_for_feature(name)
                    concepts[concept] = concepts.get(concept, np.zeros(len(test), dtype=float)) + values
                # Family dependence must be summed within each held-out member-row
                # before absolute mass is accumulated.  Averaging feature-cells
                # would divide large families by their feature count and would not
                # reconcile with the member prediction.
                for family, values in families.items():
                    _update(family_stats, (dataset, selection["model_output_scale"], family), values)
                for concept, values in concepts.items():
                    _update(concept_stats, (dataset, selection["model_output_scale"], concept), values)
                    member_concept_rows.append({
                        "dataset": dataset, "model_scale": selection["model_output_scale"],
                        "outer_partition": partition_number, "member": member, "concept": concept,
                        "member_mean_signed_contribution": float(values.mean()),
                        "member_mean_absolute_contribution": float(np.abs(values).mean()),
                    })

    def records(stats: dict[tuple[str, str, str], list[float]], label: str) -> pd.DataFrame:
        rows = []
        for (dataset, scale, name), (signed, absolute, positive, count) in stats.items():
            rows.append({
                "dataset": dataset, "model_scale": scale, label: name,
                "mean_signed_contribution": signed / count,
                "mean_absolute_contribution": absolute / count,
                "positive_fraction": positive / count, "heldout_member_rows": int(count),
            })
        frame = pd.DataFrame(rows)
        frame["rank"] = frame.groupby(["dataset", "model_scale"]).mean_absolute_contribution.rank(method="first", ascending=False).astype(int)
        return frame.sort_values(["dataset", "model_scale", "rank"])

    feature_frame = records(feature_stats, "feature")
    concept_frame = records(concept_stats, "concept")
    family_frame = records(family_stats, "family")
    for frame in (feature_frame, concept_frame, family_frame):
        denominators = frame.groupby(["dataset", "model_scale"]).mean_absolute_contribution.transform("sum")
        frame["normalized_absolute_mass"] = frame.mean_absolute_contribution / denominators.clip(lower=np.finfo(float).eps)
    consistency = pd.DataFrame(member_concept_rows)
    if len(consistency):
        consistency["sign"] = np.sign(consistency.member_mean_signed_contribution)
        sign_summary = consistency.groupby(["dataset", "model_scale", "concept"], as_index=False).agg(
            fold_seed_members=("member_mean_signed_contribution", "size"),
            positive_members=("sign", lambda values: int((values > 0).sum())),
            negative_members=("sign", lambda values: int((values < 0).sum())),
            zero_members=("sign", lambda values: int((values == 0).sum())),
        )
        sign_summary["sign_consistency"] = sign_summary[["positive_members", "negative_members"]].max(axis=1) / sign_summary.fold_seed_members
        concept_frame = concept_frame.merge(sign_summary, on=["dataset", "model_scale", "concept"], how="left")
    feature_frame.to_csv(out / "global_heldout_feature_ranking.csv", index=False)
    concept_frame.to_csv(out / "global_heldout_concept_ranking.csv", index=False)
    family_frame.to_csv(out / "global_heldout_family_ranking.csv", index=False)
    consistency.to_csv(out / "fold_seed_concept_signs.csv", index=False)
    pd.DataFrame(reconciliation_rows).to_csv(out / "treeshap_reconciliation.csv", index=False)
    pd.concat(member_prediction_rows, ignore_index=True).to_csv(
        out / "heldout_member_expected_value_reconciliation.csv.gz", index=False, compression="gzip"
    )

    cluster_rows = []
    top_by_dataset = feature_frame[feature_frame["rank"].le(60)].groupby("dataset").feature.apply(set).to_dict()
    for dataset, (frame, matrices, schemas) in cluster_context.items():
        partitions = {int(row["partition"]): row for row in _outer_partitions(dataset, frame)}
        selection_paths = sorted((args.run_dir / "fold_checkpoints").glob(f"{dataset}__partition*__selection.json"))
        if args.probe:
            selection_paths = selection_paths[:1]
        for selection_path in selection_paths:
            selection = _adopted_payload(args.run_dir, json.loads(selection_path.read_text()))
            partition_number = int(selection["outer_partition"])
            train = np.asarray(partitions[partition_number]["train"], dtype=int)
            for component in selection["selected_components"]:
                clusters = high_correlation_clusters(
                    matrices[component][train], schemas[component], threshold=0.90,
                    candidate_names=top_by_dataset.get(dataset, set()),
                )
                for cluster in clusters:
                    cluster_rows.append({
                        "dataset": dataset, "outer_partition": partition_number,
                        "component": component, "cluster_id": cluster["cluster_id"],
                        "representative": cluster["representative"], "concept": cluster["concept"],
                        "size": cluster["size"], "members": json.dumps(cluster["members"]),
                        "absolute_spearman_threshold": cluster["threshold"],
                        "fit_scope": "outer_train_only",
                    })
    pd.DataFrame(cluster_rows).to_csv(out / "training_only_shared_descriptor_clusters.csv", index=False)

    selection_frequency_rows = []
    for path in sorted((args.run_dir / "fold_checkpoints").glob("*__selection.json")):
        payload = json.loads(path.read_text())
        selection_frequency_rows.append({
            "dataset": payload["dataset"], "outer_partition": payload["outer_partition"],
            "route": payload["selected_route"], "candidate": payload["selected_candidate"],
            "transform": payload["selected_transform"], "policy": payload.get("policy", "legacy"),
            "selection_reason": payload.get("selection_audit", {}).get("selected_reason", "legacy"),
        })
    pd.DataFrame(selection_frequency_rows).to_csv(out / "outer_selection_records.csv", index=False)

    if args.probe or args.global_only:
        reconciliation = pd.DataFrame(reconciliation_rows)
        manifest = {
            "schema_version": "feature_automl_outer_heldout_explanation_v2",
            "probe": bool(args.probe), "global_only": bool(args.global_only),
            "endpoints": list(datasets),
            "folds": len(reconciliation),
            "global_feature_rows": len(feature_frame),
            "global_concept_rows": len(concept_frame),
            "shared_descriptor_clusters": len(cluster_rows),
            "global_family_rows": len(family_frame), "heldout_cases": 0,
            "max_treeshap_additivity_residual": float(reconciliation.max_additivity_residual.max()),
            "claim_boundary": "same-model held-out TreeSHAP dependence on the fitted model scale; not biological causality",
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return

    oof = pd.read_csv(args.run_dir / "oof_predictions.csv.gz")
    case_rows = []
    cases_dir = out / "cases"
    cases_dir.mkdir(exist_ok=True)
    for dataset in EARLIER + ATLAS:
        frame = cluster_context[dataset][0]
        candidates = oof[oof.dataset.eq(dataset)].copy()
        candidates["absolute_error"] = (candidates.y - candidates.pred_automl_v2).abs()
        candidates["median_distance"] = (candidates.absolute_error - candidates.absolute_error.median()).abs()
        selected = None
        for row in candidates.sort_values("median_distance").itertuples():
            # Atlas 2 retains a display-only ``HELM Annotation`` column with
            # doubled outer braces.  Training and feature construction use the
            # normalized executable HELM in ``x``; case eligibility and the
            # predictor must use that same representation.
            source_row = frame.iloc[int(row.row_index)]
            helm = _case_helm(source_row, dataset)
            sequence = __import__("scripts.benchmark_aso_atlas2_project_representation", fromlist=["canonical_aso_sequence"]).canonical_aso_sequence(helm)
            try:
                _mode, manifests, _audit = _eligible_manifests(args.run_dir, dataset, helm, sequence, int(row.row_index))
            except RuntimeError:
                continue
            member_count = sum(len({int(item["member"]) for item in payload["models"]}) for payload, _path in manifests)
            if member_count >= 3:
                selected = row
                break
        if selected is None:
            raise RuntimeError(f"no leakage-safe case with >=3 members for {dataset}")
        source = frame.iloc[int(selected.row_index)]
        case_dir = cases_dir / dataset
        command = [
            sys.executable, str(ROOT / "scripts/predict_feature_automl_system.py"),
            "--endpoint", dataset,
            "--helm", _case_helm(source, dataset),
            "--name", f"{dataset}_row_{int(selected.row_index)}",
            "--source-row-index", str(int(selected.row_index)),
            "--run-dir", str(args.run_dir), "--output-dir", str(case_dir),
        ]
        if dataset in {"mouse_hepatic", "rat_hepatic"}:
            command.extend([
                "--dosage-mg-per-kg", str(float(source.dosage_mg_per_kg)),
                "--num-doses", str(float(source.num_doses)),
                "--dosing-period-days", str(float(source.dosing_period_days)),
            ])
        subprocess.run(command, cwd=ROOT, check=True)
        packet = json.loads((case_dir / "evidence_packet.json").read_text())
        case_rows.append({
            "dataset": dataset, "row_index": int(selected.row_index),
            "observed_y": float(selected.y), "prediction": packet["prediction"]["value"],
            "member_count": packet["prediction"]["ensemble"]["member_count"],
            "prediction_mode": packet["integrity"]["prediction_mode"],
            "max_treeshap_residual": packet["integrity"]["maximum_treeshap_additivity_residual"],
        })
    pd.DataFrame(case_rows).to_csv(out / "heldout_cases.csv", index=False)
    manifest = {
        "schema_version": "feature_automl_explanation_export_v1",
        "global_feature_rows": len(feature_frame), "global_concept_rows": len(concept_frame),
        "shared_descriptor_clusters": len(cluster_rows), "heldout_cases": len(case_rows),
        "max_treeshap_additivity_residual": float(pd.DataFrame(reconciliation_rows).max_additivity_residual.max()),
        "claim_boundary": "Held-out TreeSHAP and similarity describe fitted-model dependence and analogue context, not causality.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
