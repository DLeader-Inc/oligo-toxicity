#!/usr/bin/env python3
"""Select and explain the predeclared leakage-safe TLR8-potentiation AutoML case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_aso_atlas2_project_representation import canonical_aso_sequence  # noqa: E402
from scripts.benchmark_registry_nine_endpoints import earlier_frame  # noqa: E402
from scripts.predict_feature_automl_system import _contract, _neighbors  # noqa: E402
from src.explainability.automl_concepts import concept_for_feature  # noqa: E402
from src.features.automl_descriptor_routes import build_route_matrices, descriptor_family  # noqa: E402
from src.features.featurizer import parse_helm  # noqa: E402

DATASET = "tlr7_immunotox"  # canonical endpoint: TLR8 potentiation


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/mechanism_informed_automl_nested_v1/full_automl")
    args = parser.parse_args()
    out = args.run_dir / "tlr8_case"
    out.mkdir(parents=True, exist_ok=True)
    frame = earlier_frame(DATASET).reset_index(drop=True)
    matrices, schemas, _manifest = build_route_matrices(frame)
    oof = pd.read_csv(args.run_dir / "oof_predictions.csv.gz")
    oof = oof[oof.dataset.eq(DATASET)].copy()
    grouped = oof.groupby("row_index", as_index=False).agg(
        observed=("y", "first"), prediction=("pred_policy", "mean"),
        eligible_outer_partitions=("outer_partition", "nunique"),
    )
    grouped = grouped[grouped.eligible_outer_partitions.ge(2)].copy()
    if grouped.empty:
        raise RuntimeError("no TLR8 case has at least two eligible outer partitions")
    y_std = float(np.std(frame.y.to_numpy(float), ddof=0))
    grouped["standardized_absolute_residual"] = (grouped.observed - grouped.prediction).abs() / max(y_std, np.finfo(float).eps)
    median = float(grouped.standardized_absolute_residual.median())
    grouped["distance_to_endpoint_median_residual"] = (grouped.standardized_absolute_residual - median).abs()
    analogue_scores = []
    for row in grouped.itertuples():
        source = frame.iloc[int(row.row_index)]
        helm = str(source["HELM Annotation"])
        sequence = canonical_aso_sequence(str(source.x))
        analogue_scores.append(_neighbors(DATASET, helm, sequence, top_k=1, exclude_row_index=int(row.row_index))[0]["combined_similarity"])
    grouped["nearest_measured_analogue_similarity"] = analogue_scores
    chosen = grouped.sort_values(
        ["distance_to_endpoint_median_residual", "nearest_measured_analogue_similarity", "row_index"],
        ascending=[True, False, True],
    ).iloc[0]
    row_index = int(chosen.row_index)
    source = frame.iloc[row_index]
    helm = str(source["HELM Annotation"])
    sequence = canonical_aso_sequence(str(source.x))
    modification_pattern, _bases = parse_helm(helm)
    contributions = []
    member_rows = []
    eligible_partitions = [
        int(value)
        for value in sorted(oof[oof.row_index.eq(row_index)].outer_partition.astype(int).unique())
    ]
    for fold in eligible_partitions:
        selection = json.loads((args.run_dir / "fold_checkpoints" / f"{DATASET}__partition{fold}__selection.json").read_text())
        membership = json.loads(Path(selection["membership_path"]).read_text())
        reasons = []
        if row_index in membership["train_row_indices"]:
            reasons.append("row")
        if digest(helm) in membership["train_helm_sha256"]:
            reasons.append("helm")
        if digest(sequence) in membership["train_sequence_sha256"]:
            reasons.append("canonical_sequence")
        if reasons:
            raise RuntimeError(f"case leakage in fold {fold}: {reasons}")
        records_by_member: dict[int, list[dict[str, object]]] = {}
        for record in selection["models"]:
            records_by_member.setdefault(int(record["member"]), []).append(record)
        for member, records in sorted(records_by_member.items()):
            expected = 0.0
            prediction = 0.0
            combined: dict[str, float] = {}
            feature_values: dict[str, float] = {}
            for record in records:
                component = str(record["component"])
                weight = float(record["weight"])
                estimator = joblib.load(record["path"])
                matrix = matrices[component][[row_index]]
                values = estimator.get_booster().predict(xgb.DMatrix(matrix), pred_contribs=True)[0]
                prediction += weight * float(estimator.predict(matrix)[0])
                expected += weight * float(values[-1])
                for index, name in enumerate(schemas[component]):
                    combined[name] = combined.get(name, 0.0) + weight * float(values[index])
                    feature_values[name] = float(matrix[0, index])
            reconstructed = expected + sum(combined.values())
            member_rows.append({
                "dataset": DATASET, "canonical_endpoint": "tlr8_potentiation", "outer_partition": fold,
                "member": member, "model_scale": selection["model_output_scale"],
                "selected_route": selection["selected_route"], "selected_candidate": selection["selected_candidate"],
                "selected_transform": selection["selected_transform"], "expected_value": expected,
                "prediction_model_scale": prediction, "signed_contribution_sum": sum(combined.values()),
                "reconstructed_prediction": reconstructed, "additivity_residual": prediction - reconstructed,
                "row_unseen": True, "exact_helm_unseen": True, "canonical_sequence_unseen": True,
            })
            for name, value in combined.items():
                contributions.append({
                    "dataset": DATASET, "canonical_endpoint": "tlr8_potentiation", "row_index": row_index,
                    "outer_partition": fold, "member": member, "model_scale": selection["model_output_scale"],
                    "feature": name, "feature_value": feature_values[name],
                    "family": descriptor_family(name), "concept": concept_for_feature(name), "signed_contribution": value,
                })
    member_frame = pd.DataFrame(member_rows)
    contribution_frame = pd.DataFrame(contributions)
    member_frame.to_csv(out / "case_member_reconciliation.csv", index=False)
    contribution_frame.to_csv(out / "case_feature_contributions.csv.gz", index=False, compression="gzip")
    # Sum all features belonging to a semantic concept inside each fitted
    # member first.  A feature-cell mean incorrectly divides large concepts by
    # their feature count and its sign fraction is not fold/seed consistency.
    concept_member = contribution_frame.groupby(
        ["model_scale", "family", "concept", "outer_partition", "member"],
        as_index=False,
    ).signed_contribution.sum()
    concept = concept_member.groupby(["model_scale", "family", "concept"], as_index=False).agg(
        mean_signed_contribution=("signed_contribution", "mean"),
        mean_absolute_contribution=("signed_contribution", lambda values: float(np.abs(values).mean())),
        fold_seed_sign_consistency=("signed_contribution", lambda values: float(max((values > 0).mean(), (values < 0).mean()))),
        members=("signed_contribution", "size"),
        outer_partitions=("outer_partition", "nunique"),
    )
    feature_counts = contribution_frame.groupby(
        ["model_scale", "family", "concept"], as_index=False
    ).feature.nunique().rename(columns={"feature": "features"})
    concept = concept.merge(feature_counts, on=["model_scale", "family", "concept"]).sort_values(
        "mean_absolute_contribution", ascending=False
    )
    concept.to_csv(out / "case_concept_contributions.csv", index=False)
    neighbors = _neighbors(DATASET, helm, sequence, top_k=20, exclude_row_index=row_index)
    pd.DataFrame(neighbors).to_csv(out / "case_measured_analogues.csv", index=False)
    chemistry_varies = any(item["chemistry_identity"] < 1.0 for item in neighbors)
    if chemistry_varies:
        contrast = sorted(
            neighbors,
            key=lambda item: (item["sequence_identity"] * (1.0 - item["chemistry_identity"]), abs(item["measured_value"] - float(chosen.observed)), -item["source_row_index"]),
            reverse=True,
        )[0]
        contrast_kind = "sequence_and_chemistry_contrast"
    else:
        # Alharbi_2020_1 has one fixed chemistry architecture.  Select an
        # outcome-divergent sequence contrast without implying chemistry
        # variation that does not exist.
        contrast = sorted(
            neighbors,
            key=lambda item: (abs(item["measured_value"] - float(chosen.observed)), item["sequence_identity"], -item["source_row_index"]),
            reverse=True,
        )[0]
        contrast_kind = "same_chemistry_outcome_divergent_sequence_contrast"
    contract = _contract(DATASET)
    selection_row = {
        "dataset": DATASET, "canonical_endpoint": contract["canonical_endpoint"], "row_index": row_index,
        "selection_rule": "exact HELM and canonical sequence absent from every contributing outer-train member; >=2 eligible partitions; standardized absolute residual nearest endpoint median; ties by nearest measured analogue similarity then row index",
        "eligible_outer_partitions": len(eligible_partitions), "outer_partition_ids": json.dumps(eligible_partitions),
        "endpoint_residual_median": median, "standardized_absolute_residual": float(chosen.standardized_absolute_residual),
        "helm": helm, "sequence": sequence, "modification_pattern": modification_pattern,
        "observed": float(chosen.observed), "prediction": float(chosen.prediction),
        "response_direction": contract["response_direction"], "label_unit": contract["label_unit"],
        "source_citation": contract["citation"], "nearest_analogue_row_index": neighbors[0]["source_row_index"],
        "contrast_row_index": contrast["source_row_index"], "contrast_kind": contrast_kind,
    }
    pd.DataFrame([selection_row]).to_csv(out / "case_selection.csv", index=False)
    top_concepts = concept.head(10).to_dict("records")
    manifest = {
        "schema_version": "tlr8_potentiation_outer_heldout_automl_case_v1", "selection": selection_row,
        "nearest_measured_analogue": neighbors[0],
        "informative_contrast": {"contrast_kind": contrast_kind, **contrast},
        "observed_model_dependence": {
            "selected_routes": sorted(member_frame.selected_route.unique().tolist()),
            "top_concepts_by_mean_absolute_contribution": top_concepts,
            "interpretation": (
                "Ranked same-model dependence is reported directly from the selected complete expert "
                "descriptor members; biological interpretation requires the separate evidence ledger."
            ),
        },
        "maximum_treeshap_additivity_residual": float(member_frame.additivity_residual.abs().max()),
        "claim_boundaries": [
            "reporter association is not clinical immunotoxicity", "similarity does not establish a shared mechanism",
            "TreeSHAP is same-model dependence on the fitted scale, not causality",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection_row, indent=2))


if __name__ == "__main__":
    main()
