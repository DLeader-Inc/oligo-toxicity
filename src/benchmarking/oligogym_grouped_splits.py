"""Leakage-safe grouping utilities for OligoGym toxicity comparisons.

The official OligoGym ``nucleobase`` protocol clusters normalized k-mer-count
vectors.  This module adds stricter molecular-sequence identity groups, target
groups, and a joint split that simultaneously prevents target and sequence
cluster overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.grouping import canonical_sequence_groups, normalize_base_sequence


DEFAULT_SIMILARITY_CLUSTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "transfer_learning_audit"
    / "oligogym_sequence_similarity_clusters_source_complete_v2.csv"
)

STRICT_GROUP_MODES = {
    "sequence_identity90",
    "target",
    "joint_target_sequence_identity90",
}

REFERENCE_OUTER_MODE = {
    "sequence_identity90": "similarity90",
    "joint_target_sequence_identity90": "joint_target_similarity90",
}


@dataclass(frozen=True)
class GroupingFeasibility:
    """Structural support available for one grouped validation mode."""

    split_mode: str
    n_rows: int
    n_groups: int
    largest_group_rows: int
    largest_group_fraction: float
    status: str
    reason: str


@dataclass(frozen=True)
class SplitDefinition:
    """Dependency-light grouped split record used by strict validation."""

    split_mode: str
    split_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    group_labels: np.ndarray


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if root_left > root_right:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left


def exact_sequence_groups(frame: pd.DataFrame, *, dataset_key: str) -> np.ndarray:
    """Return HELM-derived exact molecular base-sequence group labels."""

    return canonical_sequence_groups(frame, dataset_key=dataset_key).astype(str).to_numpy()


def target_groups(frame: pd.DataFrame) -> np.ndarray:
    """Return explicit biological-target labels without deriving new labels."""

    if "targets" not in frame.columns:
        raise KeyError("target grouping requires a 'targets' column")
    targets = frame["targets"].fillna("__missing_target__").astype(str).to_numpy()
    if np.any(targets == ""):
        targets = np.where(targets == "", "__missing_target__", targets)
    return targets


def load_similarity_groups(
    frame: pd.DataFrame,
    *,
    dataset_key: str,
    threshold: float = 0.90,
    cluster_path: Path = DEFAULT_SIMILARITY_CLUSTER_PATH,
) -> np.ndarray:
    """Map rows to precomputed sequence-identity connected components.

    The frozen source artifact uses Hamming mismatch identity for equal-length
    sequences and Levenshtein edit identity for unequal-length sequences, each
    normalized by maximum sequence length. It unions every pair at or above
    ``threshold``. Mapping uses the audited canonical sequence: exact
    HELM/source matches share one sequence, while a simple source terminal
    extension is retained and its absent chemistry is warned separately.
    """

    if not cluster_path.exists():
        raise FileNotFoundError(
            f"similarity cluster artifact not found: {cluster_path}; "
            "run scripts/audit_sequence_similarity_groups.py"
        )
    clusters = pd.read_csv(cluster_path)
    endpoint = clusters[
        clusters["dataset"].eq(dataset_key)
        & np.isclose(clusters["threshold"].astype(float), float(threshold))
    ].copy()
    if endpoint.empty:
        raise ValueError(
            f"no similarity clusters for dataset={dataset_key!r}, threshold={threshold}"
        )
    endpoint["canonical_sequence"] = endpoint["sequence"].map(normalize_base_sequence)
    duplicate_assignments = endpoint.groupby("canonical_sequence")["cluster_id"].nunique()
    if (duplicate_assignments > 1).any():
        raise ValueError("one canonical sequence maps to multiple similarity clusters")
    mapping = endpoint.drop_duplicates("canonical_sequence").set_index(
        "canonical_sequence"
    )["cluster_id"]
    sequences = exact_sequence_groups(frame, dataset_key=dataset_key)
    missing = sorted(set(sequences) - set(mapping.index))
    if missing:
        raise ValueError(
            f"similarity artifact misses {len(missing)} canonical sequences for {dataset_key}"
        )
    return pd.Series(sequences).map(mapping).astype(str).to_numpy()


def joint_target_sequence_component_groups(
    targets: np.ndarray,
    sequence_clusters: np.ndarray,
) -> np.ndarray:
    """Build bipartite connected components for simultaneous holdout.

    Grouping by the row-wise tuple ``(target, sequence_cluster)`` is unsafe:
    either target or sequence cluster can still cross folds.  Connected
    components of the target-sequence bipartite graph guarantee that neither
    dimension crosses a component boundary.
    """

    targets = np.asarray(targets, dtype=str)
    sequence_clusters = np.asarray(sequence_clusters, dtype=str)
    if targets.shape != sequence_clusters.shape:
        raise ValueError("target and sequence-cluster arrays must have equal shape")
    target_nodes = [f"target::{value}" for value in sorted(set(targets.tolist()))]
    sequence_nodes = [
        f"sequence::{value}" for value in sorted(set(sequence_clusters.tolist()))
    ]
    union_find = _UnionFind(target_nodes + sequence_nodes)
    for target, sequence_cluster in zip(targets, sequence_clusters, strict=True):
        union_find.union(f"target::{target}", f"sequence::{sequence_cluster}")

    roots = [union_find.find(f"target::{target}") for target in targets]
    root_to_component = {
        root: f"joint{index:04d}"
        for index, root in enumerate(sorted(set(roots)), start=1)
    }
    return np.asarray([root_to_component[root] for root in roots], dtype=str)


def strict_group_labels(
    frame: pd.DataFrame,
    *,
    dataset_key: str,
    mode: str,
    threshold: float = 0.90,
    cluster_path: Path = DEFAULT_SIMILARITY_CLUSTER_PATH,
) -> np.ndarray:
    """Return a canonical strict-validation group vector.

    The joint mode uses target/sequence-cluster bipartite connected components,
    not row-wise target-cluster tuples, so neither dimension can cross folds.
    """

    if mode not in STRICT_GROUP_MODES:
        raise ValueError(f"unsupported strict group mode: {mode!r}")
    if mode == "target":
        return target_groups(frame)
    sequence = load_similarity_groups(
        frame,
        dataset_key=dataset_key,
        threshold=threshold,
        cluster_path=cluster_path,
    )
    if mode == "sequence_identity90":
        return sequence
    return joint_target_sequence_component_groups(target_groups(frame), sequence)


def strict_group_dimensions(
    frame: pd.DataFrame,
    *,
    dataset_key: str,
    threshold: float = 0.90,
    cluster_path: Path = DEFAULT_SIMILARITY_CLUSTER_PATH,
) -> dict[str, np.ndarray]:
    """Return every dimension needed to audit strict outer-fold isolation."""

    sequence = load_similarity_groups(
        frame,
        dataset_key=dataset_key,
        threshold=threshold,
        cluster_path=cluster_path,
    )
    targets = target_groups(frame)
    return {
        "sequence_identity90": sequence,
        "target": targets,
        "joint_target_sequence_identity90": joint_target_sequence_component_groups(
            targets, sequence
        ),
    }


def load_reference_outer_splits(
    path: Path,
    *,
    dataset_key: str,
    validation: str,
    rows: int,
    y: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load immutable strict outer folds from a canonical prediction artifact."""

    if validation not in REFERENCE_OUTER_MODE:
        raise ValueError(
            "reference folds are only supported for strict identity/joint validation"
        )
    reference = pd.read_csv(path)
    required = {"dataset", "outer_mode", "outer_fold", "row_index"}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(f"reference fold file misses columns: {sorted(missing)}")
    selected = reference[
        reference["dataset"].astype(str).eq(dataset_key)
        & reference["outer_mode"].astype(str).eq(REFERENCE_OUTER_MODE[validation])
    ].copy()
    selected = selected.sort_values("row_index").reset_index(drop=True)
    expected_rows = np.arange(rows, dtype=int)
    if len(selected) != rows or not np.array_equal(
        selected["row_index"].to_numpy(dtype=int), expected_rows
    ):
        raise ValueError(
            f"reference folds do not cover canonical rows for {dataset_key}/{validation}"
        )
    if y is not None and "y" in selected and not np.allclose(
        selected["y"].to_numpy(dtype=float), np.asarray(y, dtype=float), atol=1e-6
    ):
        raise ValueError(f"reference labels differ for {dataset_key}/{validation}")
    fold_ids = selected["outer_fold"].to_numpy(dtype=int)
    splits = []
    for fold in sorted(np.unique(fold_ids).tolist()):
        test_idx = np.flatnonzero(fold_ids == fold)
        train_idx = np.flatnonzero(fold_ids != fold)
        splits.append((train_idx, test_idx))
    if len(splits) < 2 or sum(len(test) for _, test in splits) != rows:
        raise ValueError(f"invalid reference folds for {dataset_key}/{validation}")
    return splits


def grouping_feasibility(
    groups: np.ndarray,
    *,
    split_mode: str,
) -> GroupingFeasibility:
    """Classify whether a grouped benchmark is structurally interpretable."""

    group_counts = pd.Series(np.asarray(groups, dtype=str)).value_counts()
    n_rows = int(group_counts.sum())
    n_groups = int(len(group_counts))
    largest = int(group_counts.max()) if n_groups else 0
    largest_fraction = float(largest / n_rows) if n_rows else float("nan")
    if n_groups < 2:
        status = "impossible"
        reason = "fewer than two independent groups"
    elif largest_fraction >= 0.80:
        status = "severely_imbalanced"
        reason = "largest group contains at least 80% of rows"
    elif n_groups < 4:
        status = "limited_groups"
        reason = "fewer than four independent groups"
    else:
        status = "feasible"
        reason = "at least four groups without one group dominating 80% of rows"
    return GroupingFeasibility(
        split_mode=split_mode,
        n_rows=n_rows,
        n_groups=n_groups,
        largest_group_rows=largest,
        largest_group_fraction=largest_fraction,
        status=status,
        reason=reason,
    )


def make_groupkfold_splits(
    groups: np.ndarray,
    y: np.ndarray,
    *,
    split_mode: str,
    n_splits: int = 5,
) -> list[SplitDefinition]:
    """Create deterministic grouped OOF splits, or reject impossible modes."""

    groups = np.asarray(groups, dtype=str)
    y = np.asarray(y)
    if groups.shape[0] != y.shape[0]:
        raise ValueError("groups and y must contain the same number of rows")
    unique_groups = int(pd.Series(groups).nunique())
    folds = min(int(n_splits), unique_groups)
    if folds < 2:
        raise ValueError(f"{split_mode} is impossible with {unique_groups} group(s)")
    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros((len(groups), 1), dtype=np.float32)
    splits = []
    for split_id, (train_indices, test_indices) in enumerate(
        splitter.split(dummy, y, groups), start=1
    ):
        splits.append(
            SplitDefinition(
                split_mode=split_mode,
                split_id=split_id,
                train_indices=np.asarray(train_indices),
                test_indices=np.asarray(test_indices),
                group_labels=groups,
            )
        )
    return splits


def split_overlap_counts(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    dimensions: dict[str, np.ndarray],
) -> dict[str, int]:
    """Count train/test group overlap separately for every audited dimension."""

    records: dict[str, int] = {}
    for name, labels in dimensions.items():
        labels = np.asarray(labels, dtype=str)
        overlap = set(labels[train_indices]) & set(labels[test_indices])
        records[f"{name}_overlap"] = len(overlap)
    return records
