"""Download the published immutable weight release without authentication."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    release = json.loads((ROOT / "reproduce/model_release.json").read_text())
    for relative, expected in release["sha256"].items():
        target = ROOT / "models" / relative
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError("Refusing to replace a differing local file: " + relative)
            continue
        url = f"https://huggingface.co/{release['repo']}/resolve/{release['commit']}/{relative}"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Downloaded weight release hash mismatch: " + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"Verified {len(release['sha256'])} files at immutable revision {release['commit']}")


if __name__ == "__main__":
    main()
