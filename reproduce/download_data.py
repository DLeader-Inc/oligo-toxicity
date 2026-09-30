"""Download original inputs from fixed revisions, verify hashes, never relicense them."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = json.loads((ROOT / "reproduce/source_downloads.json").read_text())
    for row in sources:
        subdir = "oligogym" if row["file"].endswith(".csv.gz") else "aso_atlas_2_v2_0_2"
        target = ROOT / "data/external" / subdir / row["file"]
        expected = row["downloaded_sha256"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Existing input differs from pinned source; refusing overwrite: {target}")
            print(f"Verified cached {row['file']}")
            continue
        with urllib.request.urlopen(row["url"], timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Downloaded source hash mismatch: {row['file']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"Downloaded and verified {row['file']}")


if __name__ == "__main__":
    main()
