"""Export newly fitted local deployment ensembles into the public native format.

Only load joblib files that you generated yourself and whose hashes agree with
your training manifest. Never use this tool on an untrusted downloaded pickle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_registry_nine_endpoints import EARLIER, ATLAS, earlier_frame
from scripts.benchmark_aso_atlas2_project_representation import load_task
from src.features.automl_descriptor_routes import build_route_matrices


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "models/catalog.json")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for key in EARLIER + ATLAS:
        source = args.run_dir / "deployment_models" / key / "manifest.json"
        manifest = json.loads(source.read_text())
        assert manifest["policy"] == "expert_rule_scale" and manifest["seed_count"] == 5
        frame = earlier_frame(key) if key in EARLIER else load_task(key)
        assert len(frame) == manifest["training_rows"]
        matrices, schemas, _ = build_route_matrices(frame.iloc[:1])
        endpoint = catalog["endpoints"][key]
        for record in manifest["models"]:
            path = Path(record["path"])
            assert path.is_file() and digest(path) == record["sha256"]
            model = joblib.load(path)
            model.set_params(device="cpu", n_jobs=1)
            booster = model.get_booster()
            booster.set_param({"device": "cpu", "nthread": 1})
            relative = f"models/{key}/{path.stem}.ubj"
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(destination)
            restored = xgb.Booster()
            restored.load_model(destination)
            restored.set_param({"device": "cpu", "nthread": 1})
            matrix = matrices[record["component"]]
            np.testing.assert_array_equal(model.predict(matrix), restored.predict(xgb.DMatrix(matrix)))
            record["original_joblib_sha256"] = record["sha256"]
            record["path"], record["sha256"] = relative, digest(destination)
        for field in ("membership_path", "inner_scores_path"):
            manifest.pop(field, None)
        manifest["feature_schemas"] = {c: schemas[c] for c in manifest["selected_components"]}
        manifest["weight_license"] = endpoint["weight_license"]
        (args.output / endpoint["manifest"]).write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print("Exported nine newly fitted ensembles; preserve applicable licences and source notices when sharing.")


if __name__ == "__main__":
    main()
