"""Predict using hash-checked native XGBoost files (no pickle or remote code)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.automl_descriptor_routes import build_route_matrices


def predict(bundle: Path, request: dict) -> dict:
    catalog = json.loads((bundle / "catalog.json").read_text())
    key = catalog["aliases"].get(request["endpoint"], request["endpoint"])
    if key not in catalog["endpoints"]:
        raise ValueError("Unsupported endpoint: " + str(request["endpoint"]))
    endpoint = catalog["endpoints"][key]
    helm = request["helm"]
    if not isinstance(helm, str) or not helm.startswith("RNA1{") or "$$$$" not in helm:
        raise ValueError("Provide a single-oligonucleotide HELM string in 5′→3′ order.")
    row = {"x": helm, "HELM Annotation": helm}
    if endpoint["requires_assay_context"]:
        context = request.get("assay_context", {})
        for field in ("dosage_mg_per_kg", "num_doses", "dosing_period_days"):
            if field not in context or not np.isfinite(float(context[field])):
                raise ValueError("ALT requires finite assay_context." + field)
            row[field] = float(context[field])
    matrices, schemas, _ = build_route_matrices(pd.DataFrame([row]))
    manifest = json.loads((bundle / endpoint["manifest"]).read_text())
    members = {}
    residuals, feature_rows = [], []
    for record in manifest["models"]:
        path = bundle / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Weight checksum mismatch: " + record["path"])
        component = record["component"]
        if schemas[component] != manifest["feature_schemas"][component]:
            raise ValueError("Feature order differs from the published model schema")
        model = xgb.Booster(params={"device": "cpu", "nthread": 1})
        model.load_model(path)
        model.set_param({"device": "cpu", "nthread": 1})
        matrix = xgb.DMatrix(matrices[component])
        value = float(model.predict(matrix)[0])
        contributions = model.predict(matrix, pred_contribs=True)[0]
        residual = value - float(contributions.sum())
        if not np.isclose(value, contributions.sum(), atol=2e-4, rtol=2e-5):
            raise ValueError("TreeSHAP reconstruction tolerance exceeded")
        residuals.append(abs(residual))
        member = members.setdefault(str(record["member"]), {"model_scale_prediction": 0.0, "expected_value": 0.0})
        member["model_scale_prediction"] += record["weight"] * value
        member["expected_value"] += record["weight"] * float(contributions[-1])
        for name, contribution in zip(schemas[component], contributions[:-1], strict=True):
            feature_rows.append({"member": record["member"], "feature": name,
                                 "contribution": record["weight"] * float(contribution)})
    if len(members) != manifest["seed_count"]:
        raise ValueError("Incomplete ensemble")
    values = []
    for member in members.values():
        value = member["model_scale_prediction"]
        member["prediction"] = float(np.expm1(value)) if manifest["selected_transform"] == "log1p" else value
        values.append(member["prediction"])
    # Inverse-transform each fused seed prediction, then average seeds.
    grouped = pd.DataFrame(feature_rows).groupby(["member", "feature"]).contribution.sum()
    ranked = grouped.groupby("feature").agg(["mean", lambda s: np.abs(s).mean()])
    ranked.columns = ["signed_mean_model_scale", "absolute_mean_model_scale"]
    ranked = ranked.sort_values("absolute_mean_model_scale", ascending=False).head(20)
    return {"endpoint": endpoint["canonical_endpoint"], "prediction": float(np.mean(values)),
            "prediction_value_scale": endpoint["prediction_value_scale"],
            "response_direction": endpoint["response_direction"], "license": endpoint["weight_license"],
            "model_scale": manifest["model_output_scale"], "members": members,
            "top_features": ranked.reset_index().to_dict("records"),
            "maximum_branch_treeshap_residual": max(residuals),
            "evidence_boundary": "Full-data refit inference, not held-out validation. TreeSHAP describes model dependence, not chemical causality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict(args.models, json.loads(args.input.read_text()))
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
