"""Bounded regression checks against locally retained reference predictions."""
import argparse
import json
from pathlib import Path
from predict import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = []
    for case in json.loads(args.cases.read_text()):
        result = predict(args.models, case)
        difference = abs(result["prediction"] - case["expected_prediction"])
        if difference >= 1e-10:
            raise ValueError("Prediction differs from reference: " + case["endpoint"])
        checks.append({"endpoint": case["endpoint"], "prediction_difference": difference,
                       "treeshap_max_residual": result["maximum_branch_treeshap_residual"],
                       "scope": "serialization and software regression, not generalization"})
    assert len(checks) == 9
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"endpoints": len(checks), "max_prediction_difference": max(c["prediction_difference"] for c in checks)}))


if __name__ == "__main__":
    main()
