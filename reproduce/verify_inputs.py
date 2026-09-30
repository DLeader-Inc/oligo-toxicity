"""Rebuild source frames and source-native splits, without fitting models."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_registry_nine_endpoints import EARLIER, ATLAS, earlier_frame
from scripts.benchmark_aso_atlas2_project_representation import load_task, canonical_aso_sequence
from scripts.benchmark_xgb_robust_nested import _outer_partitions


def main():
    oof = pd.read_csv(ROOT / "outputs/expert_rule_scale_v1/expert_rule_scale/oof_predictions.csv.gz")
    expected = dict(zip(EARLIER + ATLAS, [768, 1825, 2437, 192, 192, 2701, 714, 2696, 1779]))
    rows = []
    for key in EARLIER + ATLAS:
        frame = earlier_frame(key) if key in EARLIER else load_task(key)
        assert len(frame) == expected[key], (key, len(frame))
        for split in _outer_partitions(key, frame):
            test, train = split["test"], split["train"]
            saved = oof[oof.dataset.eq(key) & oof.outer_partition.eq(split["partition"])].sort_values("row_index")
            assert saved.row_index.tolist() == sorted(test.tolist())
            np.testing.assert_allclose(saved.y, frame.iloc[saved.row_index].y, rtol=0, atol=1e-7)
            group_overlap = set(split["groups"][train]) & set(split["groups"][test])
            helm_overlap = set(frame.iloc[train].x) & set(frame.iloc[test].x)
            sequence_overlap = set(frame.iloc[train].x.map(canonical_aso_sequence)) & set(frame.iloc[test].x.map(canonical_aso_sequence))
            assert not group_overlap and not helm_overlap
            rows.append({"dataset": key, "partition": int(split["partition"]), "training_rows": len(train),
                         "test_rows": len(test), "assigned_group_overlap": len(group_overlap),
                         "exact_helm_overlap": len(helm_overlap), "canonical_sequence_overlap": len(sequence_overlap)})
    assert len(rows) == 45 and sum(r["test_rows"] for r in rows) == 14450
    out = ROOT / "rerun/input_verification.json"
    out.parent.mkdir(exist_ok=True)
    result = {"endpoint_rows": expected, "partitions": rows, "total_test_records": 14450,
              "partitions_with_canonical_sequence_overlap": sum(r["canonical_sequence_overlap"] > 0 for r in rows)}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "partitions"}, indent=2))


if __name__ == "__main__":
    main()
