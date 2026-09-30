"""Recompute primary paired results and compare every published numerical cell."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "rerun/primary_statistics"
    subprocess.run([sys.executable, "scripts/analyze_expert_rule_paper.py", "--out-dir", str(out)], cwd=ROOT, check=True)
    records = []
    for name in ("source_vs_final_endpoint_effects.csv", "source_vs_final_pairing_audit.csv", "source_vs_final_system_tests.csv"):
        expected = pd.read_csv(ROOT / "results/supporting" / name)
        observed = pd.read_csv(out / name)
        if list(expected.columns) != list(observed.columns) or expected.shape != observed.shape:
            raise ValueError("Published table schema mismatch: " + name)
        for column in expected:
            if pd.api.types.is_numeric_dtype(expected[column]):
                np.testing.assert_allclose(observed[column], expected[column], rtol=1e-10, atol=1e-10, equal_nan=True, err_msg=name + ":" + column)
            else:
                pd.testing.assert_series_equal(observed[column], expected[column])
        records.append({"file": name, "rows": len(expected), "columns": len(expected.columns), "matched": True})
    result = {"verified_tables": records, "paired_rows": 14450, "partitions": 45,
              "scope": "Statistics recalculated from archived OOF predictions, not model retraining."}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
