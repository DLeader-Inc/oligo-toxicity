#!/usr/bin/env python3
"""Predict with the feature-selected AutoML ensemble and emit an evidence packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_aso_atlas2_project_representation import (  # noqa: E402
    canonical_aso_sequence,
    load_task,
)
from scripts.benchmark_feature_automl_nested import (  # noqa: E402
    ATLAS,
    EARLIER,
    inverse_target,
)
from scripts.benchmark_registry_nine_endpoints import earlier_frame  # noqa: E402
from scripts.build_evidence_agent_prompt import build_prompt  # noqa: E402
from src.data.datasets import get_dataset  # noqa: E402
from src.explainability.automl_concepts import aggregate_contributions  # noqa: E402
from src.explainability.descriptor_provenance import (  # noqa: E402
    CATALOG,
    dataset_source_scope,
    describe_feature_zh,
    source_ids_for_feature,
)
from src.explainability.prediction_confidence import sequence_identity  # noqa: E402
from src.features.automl_descriptor_routes import build_route_matrices  # noqa: E402
from src.features.featurizer import parse_helm  # noqa: E402


DEFAULT_RUN = ROOT / "outputs/integrated_feature_automl_v2/formal"
ALIASES = {
    "hela_caspase_3_7_cytotoxicity": "cytotoxicity",
    "lna_neurotoxicity": "neurotoxicity_lna",
    "moe_icv_fob_neurotoxicity": "neurotoxicity_moe",
    "tlr8_potentiation": "tlr7_immunotox",
    "tlr7_inhibition": "tlr8_immunotox",
}
ATLAS_CONTRACTS = {
    "mouse_hepatic": ("Mouse ALT", "higher recorded ALT response", "source packaged response"),
    "rat_hepatic": ("Rat ALT", "higher recorded ALT response", "source packaged response"),
    "mouse_neuro": ("Mouse FOB", "higher recorded FOB response", "source packaged response"),
    "rat_neuro": ("Rat FOB", "higher recorded FOB response", "source packaged response"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chemistry_identity(left: str, right: str) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return float(sum(a == b for a, b in zip(left, right, strict=False)) / denominator)


def _contract(dataset: str) -> dict[str, str]:
    if dataset in EARLIER:
        source = get_dataset(dataset)
        return {
            "canonical_endpoint": source.canonical_endpoint_id,
            "endpoint": source.endpoint,
            "response_direction": source.response_direction,
            "label_unit": source.label_unit,
            "citation": source.citation,
        }
    endpoint, direction, unit = ATLAS_CONTRACTS[dataset]
    return {
        "canonical_endpoint": dataset,
        "endpoint": endpoint,
        "response_direction": direction,
        "label_unit": unit,
        "citation": CATALOG["lit:aso_atlas2:2026"]["citation"],
    }


def _query_frame(args: argparse.Namespace, dataset: str) -> pd.DataFrame:
    row: dict[str, Any] = {"x": args.helm, "HELM Annotation": args.helm}
    # Only the hepatic Atlas routes were trained with these three assay
    # conditions. The neuro FOB routes have no row-level condition matrix and
    # must not be given invented hepatic context at prediction time.
    if dataset in {"mouse_hepatic", "rat_hepatic"}:
        missing = [
            name for name, value in (
                ("dosage_mg_per_kg", args.dosage_mg_per_kg),
                ("num_doses", args.num_doses),
                ("dosing_period_days", args.dosing_period_days),
            ) if value is None
        ]
        if missing:
            raise ValueError("Atlas prediction requires assay context: " + ", ".join(missing))
        row.update({
            "dosage_mg_per_kg": args.dosage_mg_per_kg,
            "num_doses": args.num_doses,
            "dosing_period_days": args.dosing_period_days,
        })
    return pd.DataFrame([row])


def _eligible_manifests(
    run_dir: Path, dataset: str, helm: str, sequence: str, row_index: int | None
) -> tuple[str, list[tuple[dict[str, Any], Path]], list[dict[str, Any]]]:
    helm_hash = hashlib.sha256(helm.encode()).hexdigest()
    sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
    deployment_path = run_dir / "deployment_models" / dataset / "manifest.json"
    deployment = _adopted_payload(run_dir, json.loads(deployment_path.read_text()))
    deployment_membership = json.loads(Path(deployment["membership_path"]).read_text())
    exact_membership = (
        (row_index is not None and row_index in deployment_membership["train_row_indices"])
        or helm_hash in deployment_membership["train_helm_sha256"]
        or sequence_hash in deployment_membership["train_sequence_sha256"]
    )
    audit: list[dict[str, Any]] = []
    if not exact_membership:
        return "full_data_seed_ensemble", [(deployment, deployment_path)], audit
    eligible: list[tuple[dict[str, Any], Path]] = []
    for path in sorted((run_dir / "fold_checkpoints").glob(f"{dataset}__partition*__selection.json")):
        payload = _adopted_payload(run_dir, json.loads(path.read_text()))
        membership = json.loads(Path(payload["membership_path"]).read_text())
        reasons = []
        if row_index is not None and row_index in membership["train_row_indices"]:
            reasons.append("source_row_was_in_training")
        if helm_hash in membership["train_helm_sha256"]:
            reasons.append("exact_helm_was_in_training")
        if sequence_hash in membership["train_sequence_sha256"]:
            reasons.append("canonical_sequence_was_in_training")
        audit.append({
            "outer_partition": payload["outer_partition"],
            "eligible": not reasons, "exclusion_reasons": reasons,
        })
        if not reasons:
            eligible.append((payload, path))
    if not eligible:
        raise RuntimeError("no query-held-out outer model remains after row/HELM/canonical-sequence exclusion")
    return "query_held_out_outer_seed_ensemble", eligible, audit


def _adopted_payload(run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen outer-performance safety decision without hiding either route."""

    path = run_dir / "adoption_manifest.json"
    if not path.exists() or json.loads(path.read_text()).get("primary_route") != "incumbent_ab":
        return payload
    adopted = dict(payload)
    adopted.update({
        "selected_route": payload["incumbent_route"],
        "selected_components": payload["incumbent_components"],
        "selected_weights": payload["incumbent_weights"],
        "selected_candidate": payload["incumbent_candidate"],
        "selected_transform": payload["incumbent_transform"],
        "model_output_scale": payload["incumbent_model_output_scale"],
        "models": payload["incumbent_models"],
        "adoption_override": "outer macro Pareto safety gate retained legacy A/B",
    })
    return adopted


def _predict_members(
    manifests: list[tuple[dict[str, Any], Path]],
    matrices: dict[str, np.ndarray],
    schemas: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    member_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    for manifest, manifest_path in manifests:
        by_member: dict[int, list[dict[str, Any]]] = {}
        for record in manifest["models"]:
            by_member.setdefault(int(record["member"]), []).append(record)
        transform = str(manifest["selected_transform"])
        model_scale = str(manifest["model_output_scale"])
        partition = manifest.get("outer_partition", "deployment")
        for member, records in sorted(by_member.items()):
            member_id = f"{partition}:seed_member_{member}"
            fused_prediction = 0.0
            fused_reconstruction = 0.0
            for record in records:
                component = str(record["component"])
                weight = float(record["weight"])
                estimator = joblib.load(record["path"])
                matrix = matrices[component]
                names = schemas[component]
                prediction = float(estimator.predict(matrix)[0])
                values = estimator.get_booster().predict(
                    xgb.DMatrix(matrix), pred_contribs=True
                )[0]
                fused_prediction += weight * prediction
                fused_reconstruction += weight * float(values.sum())
                for name, value, observed in zip(names, values[:-1], matrix[0], strict=True):
                    contribution_rows.append({
                        "member_id": member_id, "outer_partition": partition,
                        "component": component, "model_scale": model_scale,
                        "feature": name, "feature_value": float(observed),
                        "contribution": weight * float(value),
                    })
                contribution_rows.append({
                    "member_id": member_id, "outer_partition": partition,
                    "component": component, "model_scale": model_scale,
                    "feature": "__EXPECTED_VALUE__", "feature_value": np.nan,
                    "contribution": weight * float(values[-1]),
                })
            member_rows.append({
                "member_id": member_id, "outer_partition": partition,
                "manifest": str(manifest_path.resolve()),
                "selected_route": manifest["selected_route"],
                "selected_candidate": manifest["selected_candidate"],
                "selected_transform": transform, "model_scale": model_scale,
                "prediction_model_scale": fused_prediction,
                "prediction_raw_scale": float(inverse_target(np.array([fused_prediction]), transform)[0]),
                "treeshap_additivity_residual": fused_prediction - fused_reconstruction,
            })
    members = pd.DataFrame(member_rows)
    contributions = pd.DataFrame(contribution_rows)
    features = contributions[contributions.feature.ne("__EXPECTED_VALUE__")].copy()
    concepts = aggregate_contributions(features)
    return members, contributions, concepts


def _neighbors(
    dataset: str,
    helm: str,
    sequence: str,
    top_k: int = 5,
    exclude_row_index: int | None = None,
) -> list[dict[str, Any]]:
    frame = earlier_frame(dataset) if dataset in EARLIER else load_task(dataset).reset_index(drop=True)
    query_mod, _query_bases = parse_helm(helm)
    contract = _contract(dataset)
    scope = dataset_source_scope(dataset)
    rows = []
    for index, row in frame.iterrows():
        if exclude_row_index is not None and int(index) == exclude_row_index:
            continue
        # Atlas 2's display column has doubled outer braces; ``x`` is the
        # normalized HELM used during training and must also drive similarity.
        row_helm = str(row["x"] if dataset in ATLAS else row["HELM Annotation"])
        row_mod, _ = parse_helm(row_helm)
        row_sequence = canonical_aso_sequence(str(row.x))
        seq = sequence_identity(sequence, row_sequence)
        chem = chemistry_identity(query_mod, row_mod)
        rows.append({
            "rank": 0, "dataset": dataset, "source_row_index": int(index),
            "sequence": row_sequence, "helm": row_helm,
            "modification_pattern": row_mod, "target": str(row.get("target", row.get("targets", ""))),
            "measured_value": float(row.y), "sequence_identity": seq,
            "normalized_levenshtein_identity": seq,
            "chemistry_identity": chem, "combined_similarity": 0.5 * (seq + chem),
            "exact_sequence_match": seq == 1.0, "exact_input_match": row_helm == helm,
            "citation": contract["citation"], "source_scope": scope["source_scope"],
            "row_level_source_status": scope["row_level_source_status"],
            "source_urls": scope["source_urls"],
            "response_direction": contract["response_direction"], "label_unit": contract["label_unit"],
        })
    rows.sort(key=lambda row: (row["combined_similarity"], row["sequence_identity"], row["chemistry_identity"]), reverse=True)
    for rank, row in enumerate(rows[:top_k], start=1):
        row["rank"] = rank
    return rows[:top_k]


def _primary_route(run_dir: Path) -> str:
    path = run_dir / "adoption_manifest.json"
    if not path.exists():
        return "automl_v2"
    return str(json.loads(path.read_text()).get("primary_route", "automl_v2"))


def _interval(run_dir: Path, dataset: str, prediction: float, route: str) -> dict[str, Any]:
    oof = pd.read_csv(run_dir / "oof_predictions.csv.gz")
    rows = oof[oof.dataset.eq(dataset)]
    prediction_column = "pred_incumbent_ab" if route == "incumbent_ab" else "pred_automl_v2"
    residual = rows.y.astype(float) - rows[prediction_column].astype(float)
    lower, upper = residual.quantile([0.05, 0.95]).tolist()
    return {
        "status": "available", "method": "grouped_OOF_marginal_residual_quantiles",
        "evaluation_route": route,
        "nominal_coverage": 0.90, "low": float(prediction + lower),
        "high": float(prediction + upper),
        "empirical_coverage": float(((residual >= lower) & (residual <= upper)).mean()),
        "conditional_coverage_guarantee": False,
    }


def _render_zh(packet: dict[str, Any]) -> str:
    prediction = packet["prediction"]
    lines = [
        f"# {packet['entry']['name']}：AutoML 预测与证据约束解释",
        "",
        "## 预测",
        "",
        f"系统对 **{prediction['endpoint']}** 的集成预测为 `{prediction['value']:.6g}`，"
        f"模型成员范围为 `[{prediction['ensemble']['member_min']:.6g}, {prediction['ensemble']['member_max']:.6g}]`。",
        f"该范围是 {prediction['ensemble']['member_count']} 个模型成员的分布，不是置信区间。",
        f"90% OOF 经验残差区间为 `[{prediction['uncertainty_interval']['low']:.6g}, {prediction['uncertainty_interval']['high']:.6g}]`，不保证条件覆盖。",
        "",
        "## 模型归因",
        "",
        "以下内容描述同一预测模型所使用的统计依赖，不构成因果机制证明。",
        "",
        "| 排名 | 模型尺度 | 特征 | 观测值 | 平均带符号贡献 | 家族 |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in packet["attributions"][:10]:
        lines.append(f"| {row['rank']} | `{row['model_scale']}` | `{row['feature']}` | {row['feature_value']:.4g} | {row['contribution']:.4g} | {row['family']} |")
    if len(packet["integrity"]["attribution_model_scales"]) > 1:
        lines.extend(["", "留出成员包含不同训练尺度；各尺度归因分别可加，不能跨尺度相加为原尺度集成预测。"]) 
    lines.extend(["", "## 相似实测序列", "", "| 排名 | 序列 | HELM | 序列相似度 | 化学相似度 | 实测值 | 来源 |", "|---:|---|---|---:|---:|---:|---|"])
    for row in packet["neighbors"]:
        lines.append(f"| {row['rank']} | `{row['sequence']}` | `{row['helm']}` | {row['sequence_identity']:.3f} | {row['chemistry_identity']:.3f} | {row['measured_value']:.4g} | {row['citation']} |")
    lines.extend([
        "", "## 解释边界", "",
        "相似序列仅提供适用域和回顾性类似物证据，不参与模型预测，也不证明共同机制。",
        "文献说明 descriptor 的设计依据；具体贡献方向来自本次拟合模型，而非文献预先规定。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--helm", required=True)
    parser.add_argument("--name", default="query")
    parser.add_argument("--source-row-index", type=int)
    parser.add_argument("--dosage-mg-per-kg", type=float)
    parser.add_argument("--num-doses", type=float)
    parser.add_argument("--dosing-period-days", type=float)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = ALIASES.get(args.endpoint, args.endpoint)
    if dataset not in EARLIER + ATLAS:
        raise KeyError(f"unsupported endpoint: {args.endpoint}")
    query = _query_frame(args, dataset)
    matrices, schemas, _family_manifest = build_route_matrices(query)
    sequence = canonical_aso_sequence(args.helm)
    mode, manifests, exclusion_audit = _eligible_manifests(
        args.run_dir, dataset, args.helm, sequence, args.source_row_index
    )
    members, contributions, concepts = _predict_members(manifests, matrices, schemas)
    prediction = float(members.prediction_raw_scale.mean())
    feature_rows = contributions[contributions.feature.ne("__EXPECTED_VALUE__")].copy()
    feature_summary = feature_rows.groupby(["model_scale", "feature"], as_index=False).agg(
        contribution=("contribution", "mean"), feature_value=("feature_value", "mean"),
        mean_absolute=("contribution", lambda values: float(np.abs(values).mean())),
    ).sort_values("mean_absolute", ascending=False)
    attribution_records = []
    attribution_quality = []
    for model_scale, scale_rows in feature_summary.groupby("model_scale", sort=True):
        total_mass = float(scale_rows["mean_absolute"].sum())
        top_mass = float(scale_rows.head(15)["mean_absolute"].sum())
        attribution_quality.append({
            "model_scale": str(model_scale),
            "top_k": 15,
            "top_k_absolute_contribution_mass_coverage": (
                top_mass / total_mass if total_mass > 0 else 1.0
            ),
            "quality_kind": "mean_absolute_feature_contribution_mass",
        })
        for rank, row in enumerate(scale_rows.head(15).itertuples(), start=1):
            family = "model"
            try:
                from src.features.automl_descriptor_routes import descriptor_family
                family = descriptor_family(row.feature)
            except ValueError:
                pass
            attribution_records.append({
                "rank": rank, "feature": row.feature, "feature_value": float(row.feature_value),
                "contribution": float(row.contribution),
                "direction": "raises_prediction" if row.contribution > 0 else "lowers_prediction",
                "family": family, "model_scale": model_scale,
                "interpretation": describe_feature_zh(row.feature, float(row.feature_value)),
                "provenance_evidence_ids": source_ids_for_feature(row.feature, dataset),
                "causal_claim": False,
            })
    evidence_ids = list(dict.fromkeys(
        [evidence_id for row in attribution_records for evidence_id in row["provenance_evidence_ids"]]
        + (["lit:aso_atlas2:2026"] if dataset in ATLAS else [])
    ))
    evidence = [
        {
            "evidence_id": evidence_id, "category": CATALOG[evidence_id]["category"],
            "origin": "curated", "relation": "context",
            "statement": CATALOG[evidence_id]["statement"],
            "source_title": CATALOG[evidence_id]["source_title"],
            "citation": CATALOG[evidence_id]["citation"], "url": CATALOG[evidence_id]["url"],
            "locator": CATALOG[evidence_id]["locator"],
            "metadata": {"claim_scope": CATALOG[evidence_id]["claim_scope"], "review_status": "source_reviewed"},
        }
        for evidence_id in evidence_ids
    ]
    summary = pd.read_csv(args.run_dir / "summary.csv")
    primary_route = _primary_route(args.run_dir)
    metric_row = summary[(summary.dataset == dataset) & (summary.evaluation_route == primary_route)].iloc[0]
    partition_metrics = pd.read_csv(args.run_dir / "partition_metrics.csv")
    fold_metric_rows = partition_metrics[
        partition_metrics.dataset.eq(dataset)
        & partition_metrics.evaluation_route.eq(primary_route)
    ]
    contract = _contract(dataset)
    interval = _interval(args.run_dir, dataset, prediction, primary_route)
    training_membership = []
    for payload, _manifest_path in manifests:
        rows = int(payload.get("outer_train_rows", payload.get("training_rows", 0)))
        groups = int(payload.get("outer_train_groups", payload.get("training_groups", 0)))
        partition = payload.get("outer_partition", "deployment")
        model_records = payload.get("models", [])
        seed_members = {
            int(record["member"])
            for record in model_records
            if "member" in record
        }
        training_membership.append({
            "outer_partition": partition,
            "training_rows": rows,
            "training_groups": groups,
            "seed_member_count": len(seed_members),
            "branch_estimator_reference_count": len(model_records),
        })
    training_row_values = [row["training_rows"] for row in training_membership]
    training_group_values = [row["training_groups"] for row in training_membership]
    source_frame = (
        earlier_frame(dataset)
        if dataset in EARLIER
        else load_task(dataset).reset_index(drop=True)
    )
    dataset_columns = ["x", "y"] + (["cv_group"] if "cv_group" in source_frame else [])
    dataset_bytes = source_frame[dataset_columns].to_csv(index=True).encode("utf-8")
    feature_schema_bytes = json.dumps(
        schemas, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    model_artifacts = []
    selection_manifests = []
    for payload, manifest_path in manifests:
        selection_manifests.append({
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
        })
        for record in payload.get("models", []):
            model_path = Path(record["path"])
            model_artifacts.append({
                "member": int(record["member"]),
                "component": str(record["component"]),
                "path": str(model_path.resolve()),
                "sha256": str(record.get("sha256") or sha256(model_path)),
            })
    expected_values = (
        contributions[contributions.feature.eq("__EXPECTED_VALUE__")]
        .groupby(["member_id", "model_scale"], as_index=False)
        .agg(expected_value=("contribution", "sum"))
        .to_dict(orient="records")
    )
    packet = {
        "schema_version": "feature_automl_evidence_packet_v1",
        "implementation_version": "v11_integrity_extended",
        "entry": {"name": args.name, "helm": args.helm, "sequence": sequence, "target": "", "source_row_index": args.source_row_index},
        "prediction": {
            "endpoint": contract["canonical_endpoint"], "endpoint_label": contract["endpoint"],
            "value": prediction, "model_family": "guarded feature-route XGBoost seed ensemble",
            "descriptor_family": "+".join(sorted(set(members.selected_route))),
            "representation": "HELM-derived Registry/motif/chemical-modification routes",
            "training_mode": mode, "validation_protocol": "source-appropriate grouped outer validation",
            "training_rows": {
                "minimum": min(training_row_values), "maximum": max(training_row_values),
                "by_outer_partition": training_membership,
            },
            "training_groups": {
                "minimum": min(training_group_values), "maximum": max(training_group_values),
            },
            "total_outer_partition_count": 5,
            "eligible_outer_partition_count": len(training_membership),
            "metrics_population": (
                "route-level grouped OOF predictions under the declared outer grouping contract; "
                "not the stricter case-eligibility subset that additionally excludes exact row, "
                "exact HELM, and canonical sequence"
            ),
            "response_direction": contract["response_direction"],
            "label_unit": contract["label_unit"],
            "metrics": {
                "spearman": float(metric_row.spearman), "r2": float(metric_row.raw_r2),
                "raw_r2": float(metric_row.raw_r2), "rmse": float(metric_row.rmse),
                "y_std": float(metric_row.y_std), "rmse_over_y_std": float(metric_row.rmse_over_y_std),
                "sd_spearman": float(fold_metric_rows.spearman.std(ddof=0)),
                "sd_r2": float(fold_metric_rows.r2.std(ddof=0)),
                "sd_rmse_over_y_std": float(fold_metric_rows.rmse_over_y_std.std(ddof=0)),
            },
            "uncertainty_interval": interval,
            "ensemble": {
                "member_count": len(members), "mean": prediction,
                "median": float(members.prediction_raw_scale.median()),
                "sd": float(members.prediction_raw_scale.std(ddof=0)),
                "member_min": float(members.prediction_raw_scale.min()),
                "member_max": float(members.prediction_raw_scale.max()),
                "range_kind": "model_member_distribution_not_confidence_interval",
            },
        },
        "attributions": attribution_records,
        "attribution_quality": attribution_quality,
        "expected_values_by_member": expected_values,
        "concept_attributions": concepts.to_dict(orient="records"),
        "neighbors": _neighbors(dataset, args.helm, sequence, exclude_row_index=args.source_row_index),
        "evidence": evidence,
        "mechanisms": [{
            "hypothesis_id": "model_concept_pattern", "name": "descriptor-linked candidate pattern",
            "status": "unresolved", "claim": "Top concept attributions identify a fitted-model dependency that requires experimental falsification.",
            "confidence": 0.0, "confidence_kind": "not_scored_unresolved_hypothesis",
            "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
            "alternative_explanations": ["correlated descriptors", "assay context", "training-set composition"],
            "falsification_tests": ["matched sequence/chemistry substitutions under the same assay"],
            "literature_queries": [], "patent_queries": [], "counterevidence_queries": [],
            "evidence_review_status": "requires_independent_review",
        }],
        "secondary_models": [],
        "limitations": [
            "Retrospective grouped validation is not prospective validation.",
            "TreeSHAP is model-dependent and not causal evidence.",
            "Member range is not a calibrated uncertainty interval.",
            "Similarity is analogue/applicability evidence only.",
        ],
        "integrity": {
            "prediction_mode": mode, "member_exclusion_audit": exclusion_audit,
            "validation_disclosure": {
                "outer_mode": mode,
                "repeat": "not_applicable_multi_partition_ensemble",
                "seed": "five_seed_member_ensemble",
                "outer_fold": [
                    int(row["outer_partition"])
                    for row in exclusion_audit if row["eligible"]
                ],
                "held_out_membership": True,
                "membership_exclusion_rule": (
                    "source row, exact HELM, and canonical sequence excluded "
                    "from every model member used for this training-example case"
                    if mode == "query_held_out_outer_seed_ensemble"
                    else "new input evaluated by deployment seed ensemble"
                ),
            },
            "exact_sequence_in_training_fold": False,
            "exact_helm_in_training_fold": False,
            "adopted_evaluation_route": primary_route,
            "attribution_model_scales": sorted(members.model_scale.unique().tolist()),
            "cross_scale_attribution_additivity": len(members.model_scale.unique()) == 1,
            "attribution_additivity_scope": "per_member_on_actual_model_scale",
            "raw_scale_strict_attribution_additivity": bool(members.selected_transform.eq("identity").all()),
            "maximum_treeshap_additivity_residual": float(members.treeshap_additivity_residual.abs().max()),
            "outer_or_deployment_manifests": [str(path.resolve()) for _payload, path in manifests],
            "selection_manifests": selection_manifests,
            "model_artifacts": model_artifacts,
            "dataset_contract": {
                "columns": dataset_columns,
                "rows": int(len(source_frame)),
                "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "hash_kind": "indexed_training_frame_csv",
            },
            "feature_schema": {
                "component_feature_counts": {
                    component: len(names) for component, names in schemas.items()
                },
                "sha256": hashlib.sha256(feature_schema_bytes).hexdigest(),
                "hash_kind": "canonical_component_to_feature_names_json",
            },
            "selection_uses_outer_test": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.output_dir / "evidence_packet.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    members.to_csv(args.output_dir / "ensemble_member_predictions.csv", index=False)
    contributions.to_csv(args.output_dir / "same_model_contributions.csv.gz", index=False, compression="gzip")
    concepts.to_csv(args.output_dir / "concept_ranking.csv", index=False)
    (args.output_dir / "EXPLANATION_zh.md").write_text(_render_zh(packet), encoding="utf-8")
    (args.output_dir / "agent_task.md").write_text(build_prompt(packet), encoding="utf-8")
    manifest = {
        "schema_version": "feature_automl_prediction_bundle_v1",
        "packet": str(packet_path.resolve()), "packet_sha256": sha256(packet_path),
        "agent_contract": "EvidenceAgentRuntime may synthesize only from this immutable packet; retrieved records remain candidates until reviewed.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prediction": prediction, "member_count": len(members), "mode": mode, "output": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
