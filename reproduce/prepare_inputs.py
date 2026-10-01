"""Rebuild the audited clustering prerequisite if absent; never use old predictions."""
import hashlib
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "7ba380c59a9013406534a27801755e713dcc959747a690b94277cb112f91a9ff"
DESTINATION = ROOT / "outputs/transfer_learning_audit/oligogym_sequence_similarity_clusters_source_complete_v2.csv"


def ensure_clusters(destination=DESTINATION, builder=None):
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != EXPECTED:
            raise ValueError("Clustering prerequisite differs from frozen protocol; refusing overwrite")
        return "verified_existing"
    if builder is None:
        sys.path.insert(0, str(ROOT))
        import pandas as pd
        from scripts.audit_sequence_similarity_groups import audit_dataset
        from src.data.datasets import TOXICITY_DATASETS
        rows = []
        for ds in TOXICITY_DATASETS:
            _, clusters = audit_dataset(ds.model_key, ds.oligogym_key)
            rows.extend(clusters)
        stream = io.StringIO()
        pd.DataFrame(rows).to_csv(stream, index=False)
        payload = stream.getvalue().encode()
    else:
        payload = builder()
    if hashlib.sha256(payload).hexdigest() != EXPECTED:
        raise ValueError("Regenerated clusters fail frozen SHA-256 verification")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids clobbering a concurrent preparation.
    with destination.open("xb") as f:
        f.write(payload)
    return "regenerated_and_verified"


if __name__ == "__main__":
    print(ensure_clusters())
