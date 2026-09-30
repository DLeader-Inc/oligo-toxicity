#!/usr/bin/env python3
"""Audit source-complete exact grouping versus similarity clustering.

The current toxicity benchmark uses exact base sequence groups. This script
checks whether highly similar but non-identical oligos would be split across
different exact groups, which is the leakage risk BLAST-style clustering is
meant to reduce.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import TOXICITY_DATASETS, load_oligogym
from src.data.grouping import build_sequence_contract
from src.features.featurizer import featurize_dataset


OUT_DIR = ROOT / "outputs" / "transfer_learning_audit"
SUMMARY_OUT = OUT_DIR / "oligogym_sequence_similarity_group_audit_source_complete_v2.csv"
CLUSTERS_OUT = OUT_DIR / "oligogym_sequence_similarity_clusters_source_complete_v2.csv"
REPORT_OUT = ROOT / "docs" / "reports" / "OLIGOGYM_SEQUENCE_SIMILARITY_GROUP_AUDIT_zh.md"


def report_date() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


@dataclass(frozen=True)
class UnionFind:
    parent: dict[str, str]

    @classmethod
    def from_items(cls, items: list[str]) -> "UnionFind":
        return cls({item: item for item in items})

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            self.parent[rb] = ra


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance for short oligo sequences."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def pair_identity(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    if len(a) == len(b):
        return 1.0 - sum(ca != cb for ca, cb in zip(a, b, strict=False)) / denom
    return 1.0 - edit_distance(a, b) / denom


def compute_pair_identities(seqs: list[str], min_threshold: float = 0.8) -> tuple[list[tuple[str, str, float]], float]:
    """Compute high-similarity pairs once and track global max identity.

    Length-based pruning avoids edit-distance work when even the best possible
    identity cannot reach the minimum threshold and cannot improve the current
    max identity.
    """
    pairs: list[tuple[str, str, float]] = []
    best = 0.0
    n = len(seqs)
    for i in range(n):
        len_i = len(seqs[i])
        for j in range(i + 1, n):
            len_j = len(seqs[j])
            max_len = max(len_i, len_j, 1)
            max_possible = min(len_i, len_j) / max_len
            if max_possible < min_threshold and max_possible <= best:
                continue
            ident = pair_identity(seqs[i], seqs[j])
            if ident > best:
                best = ident
            if ident >= min_threshold:
                pairs.append((seqs[i], seqs[j], ident))
    return pairs, best


def cluster_sequences(seqs: list[str], pair_identities: list[tuple[str, str, float]], threshold: float) -> dict[str, str]:
    uf = UnionFind.from_items(seqs)
    for a, b, ident in pair_identities:
        if ident >= threshold:
            uf.union(a, b)
    roots = {seq: uf.find(seq) for seq in seqs}
    # Stable compact IDs by root order.
    root_to_id = {root: f"sim{idx:04d}" for idx, root in enumerate(sorted(set(roots.values())), start=1)}
    return {seq: root_to_id[root] for seq, root in roots.items()}


def audit_dataset(model_key: str, oligogym_key: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    df = load_oligogym(oligogym_key)
    _, parsed = featurize_dataset(df)
    contract = build_sequence_contract(parsed, dataset_key=model_key)
    parsed = parsed.copy()
    parsed["canonical_sequence"] = contract[
        "canonical_sequence_group_id"
    ].to_numpy()
    seqs = sorted(parsed["canonical_sequence"].astype(str).unique())
    targets = parsed["targets"].astype(str) if "targets" in parsed.columns else pd.Series([""] * len(parsed))
    summary_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []

    exact_groups = len(seqs)
    pair_identities, max_identity = compute_pair_identities(seqs, min_threshold=0.8)

    for threshold in [0.8, 0.85, 0.9, 0.95]:
        cluster_map = cluster_sequences(seqs, pair_identities, threshold)
        cluster_series = parsed["canonical_sequence"].astype(str).map(cluster_map)
        cluster_row_sizes = cluster_series.value_counts()
        seq_cluster_sizes = pd.Series(cluster_map).value_counts()
        n_clusters = int(seq_cluster_sizes.size)
        n_multi = int((seq_cluster_sizes > 1).sum())
        largest_rows = int(cluster_row_sizes.max()) if len(cluster_row_sizes) else 0
        largest_seqs = int(seq_cluster_sizes.max()) if len(seq_cluster_sizes) else 0
        compression = exact_groups / max(n_clusters, 1)
        summary_rows.append(
            {
                "dataset": model_key,
                "threshold": threshold,
                "n_rows": len(parsed),
                "unique_exact_sequences": exact_groups,
                "similarity_clusters": n_clusters,
                "multi_sequence_clusters": n_multi,
                "largest_cluster_sequences": largest_seqs,
                "largest_cluster_rows": largest_rows,
                "cluster_compression_ratio": compression,
                "max_nonidentical_pair_identity": max_identity,
                "unique_targets": int(targets.nunique()),
                "interpretation": (
                    "similarity grouping is meaningfully coarser than exact grouping"
                    if compression >= 1.2
                    else "similarity grouping is close to exact grouping at this threshold"
                ),
            }
        )
        seq_to_nrows = parsed["canonical_sequence"].astype(str).value_counts().to_dict()
        seq_to_targets = parsed.groupby("canonical_sequence")["targets"].apply(lambda x: ";".join(sorted(set(map(str, x))))).to_dict() if "targets" in parsed.columns else {}
        for seq, cid in cluster_map.items():
            cluster_rows.append(
                {
                    "dataset": model_key,
                    "threshold": threshold,
                    "cluster_id": cid,
                    "sequence": seq,
                    "sequence_length": len(seq),
                    "n_rows_for_sequence": int(seq_to_nrows.get(seq, 0)),
                    "targets": seq_to_targets.get(seq, ""),
                }
            )

    return summary_rows, cluster_rows


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# OligoGym Sequence Similarity Group Audit",
        "",
        f"日期：{report_date()}",
        "",
        "## 目的",
        "",
        "本版 `sequence group` 使用经审计的完整序列：HELM 与 source FASTA 一致时使用共同序列；MOE 的 source FASTA 比 HELM 多一个已由专利核实的末端碱基时保留该碱基，但缺失 chemistry 另记为 UNKNOWN。这个审计进一步检查高度相似 oligo 是否会跨 fold。",
        "",
        "这里没有调用 BLAST，而是对短 oligo 使用 Levenshtein identity 做轻量 BLAST-like cluster audit。identity 定义为 `1 - edit_distance / max_length`。这不是替代正式 BLAST/CD-HIT，而是用于快速判断是否值得增加 sequence-similarity cluster validation。",
        "",
        "## 结论",
        "",
        "如果 0.9 或 0.95 threshold 下 cluster 数量接近 exact sequence 数量，说明 exact group 已经接近 similarity group；如果 cluster compression ratio 明显大于 1，则 exact group 可能仍允许相似序列跨 fold。",
        "",
        "| dataset | threshold | rows | exact seqs | clusters | multi-seq clusters | largest cluster seqs | largest cluster rows | compression | max non-identical identity | interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {dataset} | {threshold:.2f} | {n_rows} | {unique_exact_sequences} | {similarity_clusters} | {multi_sequence_clusters} | {largest_cluster_sequences} | {largest_cluster_rows} | {cluster_compression_ratio:.2f} | {max_nonidentical_pair_identity:.3f} | {interpretation} |".format(
                **row.to_dict()
            )
        )
    lines.extend(
        [
            "",
            "## 对当前模型评估的影响",
            "",
            "1. 主要 benchmark 使用 source-complete similarity90 connected components；旧 HELM-only split 只用于历史复现。",
            "2. Source-complete exact group 可用于辅助分析，但不能替代 similarity90 的主要 no-leak 检验。",
            "3. 不应把 OligoAI/OligoFormer/ASOptimizer source paper 的 split 分数与我们的 exact 或 similarity grouped toxicity CV 直接等价；source split 和我们的 holdout 定义不同。",
            "",
            "## 机器可读输出",
            "",
            f"- Summary: `{SUMMARY_OUT.relative_to(ROOT)}`",
            f"- Cluster assignments: `{CLUSTERS_OUT.relative_to(ROOT)}`",
        ]
    )
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    for ds in TOXICITY_DATASETS:
        s_rows, c_rows = audit_dataset(ds.model_key, ds.oligogym_key)
        summary_rows.extend(s_rows)
        cluster_rows.extend(c_rows)
    summary = pd.DataFrame(summary_rows)
    clusters = pd.DataFrame(cluster_rows)
    summary.to_csv(SUMMARY_OUT, index=False)
    clusters.to_csv(CLUSTERS_OUT, index=False)
    write_report(summary)
    print(f"wrote {SUMMARY_OUT} rows={len(summary)}")
    print(f"wrote {CLUSTERS_OUT} rows={len(clusters)}")
    print(f"wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
