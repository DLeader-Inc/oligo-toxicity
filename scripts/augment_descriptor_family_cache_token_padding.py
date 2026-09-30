#!/usr/bin/env python3
"""Add left- and center-padded token families to a validated family cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
from src.features.position_reencoding_features import token_ohe_dataframe_for_helms  # noqa: E402


NEW_PADDING_FAMILIES = {
    "position_token_ohe_leftpad": "left",
    "position_token_ohe_centerpad": "center",
}


def augment_token_padding_families(
    existing: dict[str, pd.DataFrame],
    helms: pd.Series,
    *,
    max_pos: int,
) -> dict[str, pd.DataFrame]:
    """Preserve every source matrix and append two alignment alternatives."""

    if "core_baseline" not in existing:
        raise ValueError("source cache has no core_baseline family")
    baseline = existing["core_baseline"].reset_index(drop=True)
    if len(baseline) != len(helms):
        raise ValueError("source cache and HELM rows differ")

    augmented = {
        name: frame.reset_index(drop=True).copy() for name, frame in existing.items()
    }
    for family, alignment in NEW_PADDING_FAMILIES.items():
        token_block = token_ohe_dataframe_for_helms(
            helms.astype(str), max_pos=max_pos, alignment=alignment
        )
        added = _prefixed(token_block, family)
        augmented[family] = _sanitize_feature_frame(
            pd.concat([baseline, added], axis=1)
        ).reset_index(drop=True)
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "outputs/descriptor_family_matrix_cache_position_reencoded_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/descriptor_family_matrix_cache_position_reencoded_padding_v2",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-pos", type=int, default=30)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    identity_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        _, parsed = featurize_dataset(
            load_oligogym(get_dataset(dataset).oligogym_key)
        )
        source_path = args.source_dir / f"{dataset}.pkl"
        existing = load_family_cache(source_path, parsed["x"])
        if existing is None:
            raise FileNotFoundError(source_path)
        augmented = augment_token_padding_families(
            existing, parsed["x"], max_pos=args.max_pos
        )
        for family, source_frame in existing.items():
            identical = source_frame.equals(augmented[family])
            identity_rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "rows": len(source_frame),
                    "columns": source_frame.shape[1],
                    "identical": identical,
                }
            )
            if not identical:
                raise AssertionError(f"source matrix changed: {dataset}/{family}")

        baseline_columns = set(augmented["core_baseline"].columns)
        for family, alignment in NEW_PADDING_FAMILIES.items():
            frame = augmented[family]
            added_columns = [column for column in frame if column not in baseline_columns]
            added_values = frame[added_columns].to_numpy(dtype=float)
            nonfinite = int((~np.isfinite(added_values)).sum())
            if nonfinite:
                raise AssertionError(f"non-finite values: {dataset}/{family}")
            feature_rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "alignment": alignment,
                    "rows": len(frame),
                    "total_columns": frame.shape[1],
                    "added_columns": len(added_columns),
                    "nonconstant_added_columns": int(
                        (frame[added_columns].nunique(dropna=False) > 1).sum()
                    ),
                    "nonfinite_values": nonfinite,
                }
            )

        output_path = args.out_dir / f"{dataset}.pkl"
        save_family_cache(output_path, parsed["x"], augmented)
        print(f"{dataset}: {len(existing)} -> {len(augmented)} families", flush=True)

    identity = pd.DataFrame(identity_rows)
    features = pd.DataFrame(feature_rows)
    identity.to_csv(args.out_dir / "source_family_identity_audit.csv", index=False)
    features.to_csv(args.out_dir / "token_padding_feature_audit.csv", index=False)
    (args.out_dir / "token_padding_cache_manifest.json").write_text(
        json.dumps(
            {
                "max_pos": args.max_pos,
                "new_families": list(NEW_PADDING_FAMILIES),
                "alignments": NEW_PADDING_FAMILIES,
                "datasets": args.datasets,
                "source_dir": str(args.source_dir.resolve()),
                "all_source_matrices_exactly_identical": bool(
                    identity["identical"].astype(bool).all()
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
