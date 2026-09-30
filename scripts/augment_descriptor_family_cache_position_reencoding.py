#!/usr/bin/env python3
"""Add re-encoded position families to validated descriptor-matrix caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_current_descriptor_families_grouped import (  # noqa: E402
    DEFAULT_DATASETS,
    _prefixed,
)
from scripts.train_structure_descriptor_ablation import _sanitize_feature_frame  # noqa: E402
from src.benchmarking.descriptor_family_cache import load_family_cache, save_family_cache  # noqa: E402
from src.data.datasets import get_dataset, load_oligogym  # noqa: E402
from src.features.featurizer import featurize_dataset  # noqa: E402
from src.features.position_reencoding_features import dataframes_for_helms  # noqa: E402


NEW_FAMILIES = (
    "position_token_ohe",
    "position_relative",
    "position_interactions",
    "position_token_relative",
    "position_reencoded_all",
)


def augment_families(
    existing: dict[str, pd.DataFrame],
    helms: pd.Series,
    *,
    max_pos: int,
) -> dict[str, pd.DataFrame]:
    """Return a copy of ``existing`` with five deterministic position blocks."""

    if "core_baseline" not in existing:
        raise ValueError("source cache has no core_baseline family")
    baseline = existing["core_baseline"].reset_index(drop=True)
    if len(baseline) != len(helms):
        raise ValueError("source cache and HELM rows differ")
    blocks = dataframes_for_helms(helms.astype(str), max_pos=max_pos)
    blocks["position_token_relative"] = pd.concat(
        [blocks["position_token_ohe"], blocks["position_relative"]],
        axis=1,
    )
    blocks["position_reencoded_all"] = pd.concat(
        [
            blocks["position_token_ohe"],
            blocks["position_relative"],
            blocks["position_interactions"],
        ],
        axis=1,
    )

    augmented = {name: frame.reset_index(drop=True).copy() for name, frame in existing.items()}
    for name in NEW_FAMILIES:
        added = _prefixed(blocks[name], name)
        augmented[name] = _sanitize_feature_frame(
            pd.concat([baseline, added], axis=1)
        ).reset_index(drop=True)
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/descriptor_family_matrix_cache")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/descriptor_family_matrix_cache_position_reencoded_v1",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-pos", type=int, default=30)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for dataset in args.datasets:
        _, parsed = featurize_dataset(load_oligogym(get_dataset(dataset).oligogym_key))
        source_path = args.source_dir / f"{dataset}.pkl"
        existing = load_family_cache(source_path, parsed["x"])
        if existing is None:
            raise FileNotFoundError(source_path)
        augmented = augment_families(existing, parsed["x"], max_pos=args.max_pos)
        output_path = args.out_dir / f"{dataset}.pkl"
        save_family_cache(output_path, parsed["x"], augmented)
        for family in NEW_FAMILIES:
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "rows": len(augmented[family]),
                    "total_columns": augmented[family].shape[1],
                    "added_columns": augmented[family].shape[1] - augmented["core_baseline"].shape[1],
                    "source_cache": str(source_path),
                    "output_cache": str(output_path),
                }
            )
        print(f"{dataset}: {len(existing)} -> {len(augmented)} families", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.out_dir / "position_reencoding_cache_manifest.csv", index=False)
    (args.out_dir / "position_reencoding_cache_manifest.json").write_text(
        json.dumps(
            {
                "max_pos": args.max_pos,
                "new_families": list(NEW_FAMILIES),
                "datasets": args.datasets,
                "source_dir": str(args.source_dir),
                "rows": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
