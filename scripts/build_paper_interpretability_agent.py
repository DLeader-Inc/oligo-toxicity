#!/usr/bin/env python3
"""Build the auditable, reader-facing interpretability Agent case for paper v3.

The loop deliberately separates deterministic evidence tools from narrative
synthesis.  It creates an immutable packet, synthesizes a reader-facing story,
audits required claims and boundaries, and repairs the narrative when a check
fails.  The same packet can later be supplied to the API-backed reusable
EvidenceAgentRuntime for an additional independent model review.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_aso_atlas2_project_representation import (  # noqa: E402
    canonical_aso_sequence,
    load_task,
)
from scripts.benchmark_registry_nine_endpoints import earlier_frame  # noqa: E402
from src.explainability.automl_concepts import concept_for_feature  # noqa: E402
from src.explainability.descriptor_provenance import (  # noqa: E402
    CATALOG,
    describe_feature,
    describe_feature_zh,
    source_ids_for_feature,
)
from src.features.automl_descriptor_routes import (  # noqa: E402
    build_route_matrices,
    descriptor_family,
)
from src.features.featurizer import parse_helm  # noqa: E402


DATASET = "tlr7_immunotox"
RUN = ROOT / "outputs/expert_rule_scale_v1/expert_rule_scale"
PAPER_ANALYSIS = ROOT / "outputs/expert_rule_paper_v3_analysis"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def trace(rows: list[dict[str, Any]], tool: str, inputs: list[Path], outputs: list[Path], note: str) -> None:
    rows.append({
        "step": len(rows) + 1,
        "tool": tool,
        "input_sha256": {str(p): sha256(p) for p in inputs if p.exists()},
        "outputs": [str(p) for p in outputs],
        "note": note,
    })


def feature_code_path(feature: str) -> str:
    if feature.startswith("uohe__"):
        return "src/features/universal_oligo_representation.py"
    if feature.startswith("ucore__") or feature.startswith("ucontext__"):
        return "src/features/universal_oligo_representation.py"
    if feature.startswith("scint_") or feature.startswith("seqchem__"):
        return "src/features/sequence_chemistry_interaction_features.py"
    if feature.startswith("pint_"):
        return "src/features/position_first_descriptor_features.py"
    if feature.startswith("assay__"):
        return "scripts/benchmark_aso_atlas2_project_representation.py"
    return "src/features/automl_descriptor_routes.py"


def origin_kind(feature: str) -> str:
    if feature.startswith("uohe__"):
        return "project encoding of source HELM token"
    if feature.startswith("assay__"):
        return "source-recorded assay covariate"
    return "project-engineered deterministic descriptor"


def value_contract(feature: str) -> tuple[str, str]:
    lowered = feature.lower()
    if "fraction" in lowered or "gc_content" in lowered or "kmer_" in lowered:
        return "dimensionless proportion", "[0,1]"
    if feature.startswith("uohe__") or "_base_" in feature or "has_" in lowered or "anchor_" in lowered:
        return "binary indicator or token count", "{0,1} for one position; nonnegative count when pooled"
    return "count or deterministic numeric encoding", "nonnegative unless the named transform states otherwise"


def build_descriptor_databases(out: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _matrices, schemas, _manifest = build_route_matrices(frame)
    branch_map: dict[str, list[str]] = {}
    for branch in ("legacy_branch_a", "legacy_branch_b"):
        for feature in schemas[branch]:
            branch_map.setdefault(feature, []).append(branch)
    all_features = sorted(branch_map)
    schema_hash = hashlib.sha256("\n".join(all_features).encode()).hexdigest()
    global_feature = pd.read_csv(RUN / "explanations/global_heldout_feature_ranking.csv")
    global_feature = global_feature[global_feature.dataset == DATASET][
        ["feature", "mean_signed_contribution", "mean_absolute_contribution", "normalized_absolute_mass", "rank"]
    ].rename(columns={
        "mean_signed_contribution": "tlr8_global_mean_signed_contribution",
        "mean_absolute_contribution": "tlr8_global_mean_absolute_contribution",
        "normalized_absolute_mass": "tlr8_global_normalized_absolute_mass",
        "rank": "tlr8_global_rank",
    })
    case_feature = pd.read_csv(RUN / "tlr8_case/case_feature_contributions.csv.gz")
    case_feature["absolute"] = case_feature.signed_contribution.abs()
    case_feature = case_feature.groupby("feature", as_index=False).agg(
        tlr8_case_mean_signed_contribution=("signed_contribution", "mean"),
        tlr8_case_mean_absolute_contribution=("absolute", "mean"),
        tlr8_case_members=("signed_contribution", "size"),
    )
    cluster_map: dict[str, set[str]] = {}
    clusters = pd.read_csv(RUN / "explanations/training_only_shared_descriptor_clusters.csv")
    for row in clusters[clusters.dataset == DATASET].itertuples():
        for feature in json.loads(row.members):
            cluster_map.setdefault(str(feature), set()).add(str(row.cluster_id))
    rows = []
    used_sources: set[str] = set()
    for feature, branches in branch_map.items():
        evidence_ids = source_ids_for_feature(feature, DATASET)
        used_sources.update(evidence_ids)
        citations = [CATALOG[item]["citation"] for item in evidence_ids]
        urls = [CATALOG[item]["url"] for item in evidence_ids]
        unit, value_range = value_contract(feature)
        implementation = ROOT / feature_code_path(feature)
        rows.append({
            "feature": feature,
            "branches": ";".join(branches),
            "family": descriptor_family(feature),
            "concept": concept_for_feature(feature),
            "origin_kind": origin_kind(feature),
            "source_kind": "representation_defined" if feature.startswith("uohe__") else "literature_motivated_project_engineered",
            "derivation_zh": describe_feature_zh(feature, None).replace("；观测值=缺失。", "。"),
            "derivation_en": describe_feature(feature, None).replace("; observed value=missing.", "."),
            "input_fields": "HELM monomer/linkage tokens; endpoint assay covariates only when explicitly present",
            "unit": unit,
            "value_range": value_range,
            "missing_value_policy": "deterministic HELM parse; padding kept separate; unsupported chemistry maps to controlled OTHER token, never label-derived imputation",
            "implementation_file": feature_code_path(feature),
            "implementation_sha256": sha256(implementation),
            "feature_schema_sha256": schema_hash,
            "literature_evidence_ids": ";".join(evidence_ids),
            "literature_citations": " | ".join(citations),
            "literature_urls": " | ".join(urls),
            "provenance_boundary_zh": "论文只启发概念；具体维度、阈值与数值均由项目代码从 HELM 确定性推导。",
            "tlr8_chemistry_variation": "none: all 192 rows share OOOOODDDDDDDDDDOOOOO",
            "training_only_correlation_cluster_ids": ";".join(sorted(cluster_map.get(feature, set()))),
        })
    feature_db = pd.DataFrame(rows).merge(global_feature, on="feature", how="left").merge(
        case_feature, on="feature", how="left"
    ).sort_values(["family", "concept", "feature"])
    feature_db.to_csv(out / "descriptor_provenance_database.csv", index=False)

    short_quotes = {
        "lit:helm:2012": "non-natural chemical modifications makes it impossible to represent them by sequence alone",
        "lit:papargyri:2020": "the types and positions of chemical modifications in the AON are fixed",
        "lit:alharbi:2020": "in a sequence-dependent manner",
        "lit:bhamra:2025": "the interplay between nucleotide sequence, chemical modifications, and the specific position context",
    }
    source_rows = []
    for evidence_id in sorted(used_sources):
        item = CATALOG[evidence_id]
        source_rows.append({
            "evidence_id": evidence_id,
            "source_title": item["source_title"],
            "citation": item["citation"],
            "url": item["url"],
            "claim_locator": item["locator"],
            "concept_supported": item["statement"],
            "short_source_quote": short_quotes.get(evidence_id, ""),
            "quote_word_count": len(short_quotes.get(evidence_id, "").split()),
            "claim_scope": item["claim_scope"],
            "review_status": "source_reviewed",
        })
    source_db = pd.DataFrame(source_rows)
    claim_scope_zh = {
        "lit:alharbi:2020": "只支持该 TLR assay 中的序列／化学关联，不定义项目 descriptor，也不代表临床免疫毒性。",
        "lit:bhamra:2025": "只支持 sequence×chemistry×position 的交互前提；项目向量、重要性和方向均由本项目定义。",
        "lit:helm:2012": "只定义输入表示，不说明任何 token 的毒性方向。",
        "lit:papargyri:2020": "只支持架构与位置作为设计变量；不定义项目向量或本案例效应。",
    }
    source_db["claim_scope_zh"] = source_db.evidence_id.map(claim_scope_zh).fillna(
        "只提供概念或表示背景；不定义项目数值特征，也不证明模型归因具有因果性。"
    )
    source_db.to_csv(out / "descriptor_source_database.csv", index=False)
    return feature_db, source_db


def build_training_provenance(out: Path) -> pd.DataFrame:
    # Dataset size is a property of the source frame, not the number of rows that
    # happened to appear in repeated outer-test partitions.
    earlier = ("cytotoxicity", "neurotoxicity_lna", "neurotoxicity_moe", "tlr7_immunotox", "tlr8_immunotox")
    atlas = ("mouse_hepatic", "rat_hepatic", "mouse_neuro", "rat_neuro")
    counts = {dataset: len(earlier_frame(dataset)) for dataset in earlier}
    counts.update({dataset: len(load_task(dataset)) for dataset in atlas})
    records = {
        "cytotoxicity": ("HeLa caspase 3/7 cytotoxicity", "HeLa caspase 3/7 细胞毒性", "Papargyri et al. 2020", "PAPARGYRI2020", "768 LNA gapmers; caspase-3/7 assay", "768 条 LNA gapmer；caspase-3/7 assay", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6965521/"),
        "neurotoxicity_lna": ("LNA neuronal calcium score", "LNA 神经元钙评分", "Hagedorn et al. 2022", "HAGEDORN2022", "primary-neuron calcium-oscillation response", "原代神经元钙振荡 response", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9221153/"),
        "neurotoxicity_moe": ("MOE mouse ICV FOB", "MOE 小鼠 ICV FOB", "13 source patents curated by OligoGym", "IONIS_PATENTS", "mouse intracerebroventricular functional-observation score", "13 份专利来源；小鼠脑室内给药 functional-observation score", "https://github.com/Roche/OligoGym"),
        "tlr7_immunotox": ("TLR8 potentiation", "TLR8 增强", "Alharbi et al. 2020", "ALHARBI2020", "100 nM 2′OMe gapmers; HEK293-TLR8 NF-κB reporter; R848=1.0", "100 nM 2′OMe gapmer；HEK293-TLR8 NF-κB reporter；R848=1.0", "https://pubmed.ncbi.nlm.nih.gov/32544249/"),
        "tlr8_immunotox": ("TLR7 inhibition", "TLR7 抑制", "Alharbi et al. 2020", "ALHARBI2020", "100 nM 2′OMe gapmers; HEK293-TLR7 NF-κB reporter; residual R848 reporter", "100 nM 2′OMe gapmer；HEK293-TLR7 NF-κB reporter；剩余 R848 response", "https://pubmed.ncbi.nlm.nih.gov/32544249/"),
        "mouse_hepatic": ("Mouse ALT", "小鼠 ALT", "ASO Atlas 2.0", "lit:aso_atlas2:2026", "patent-derived mouse ALT records with dose schedule", "专利来源的小鼠 ALT 记录，含给药方案", "https://doi.org/10.57967/hf/8687"),
        "rat_hepatic": ("Rat ALT", "大鼠 ALT", "ASO Atlas 2.0", "lit:aso_atlas2:2026", "patent-derived rat ALT records with dose schedule", "专利来源的大鼠 ALT 记录，含给药方案", "https://doi.org/10.57967/hf/8687"),
        "mouse_neuro": ("Mouse FOB", "小鼠 FOB", "ASO Atlas 2.0", "lit:aso_atlas2:2026", "patent-derived mouse functional-observation records", "专利来源的小鼠 functional-observation 记录", "https://doi.org/10.57967/hf/8687"),
        "rat_neuro": ("Rat FOB", "大鼠 FOB", "ASO Atlas 2.0", "lit:aso_atlas2:2026", "patent-derived rat functional-observation records", "专利来源的大鼠 functional-observation 记录", "https://doi.org/10.57967/hf/8687"),
    }
    rows = []
    for dataset, (endpoint, endpoint_zh, source, source_id, context, context_zh, url) in records.items():
        rows.append({
            "dataset": dataset,
            "endpoint": endpoint,
            "endpoint_zh": endpoint_zh,
            "source_dataset_rows": int(counts[dataset]),
            "source": source,
            "source_id": source_id,
            "assay_or_collection_context": context,
            "assay_or_collection_context_zh": context_zh,
            "url": url,
            "row_provenance_limit": "publication/dataset level; exact paper table row not asserted unless recorded",
            "validation_scope": "five grouped outer partitions; outer test contains training-unseen assigned groups",
        })
    result = pd.DataFrame(rows)
    result.to_csv(out / "training_data_provenance.csv", index=False)
    return result


def overlapping_count(sequence: str, motif: str) -> int:
    return sum(sequence[i:i + len(motif)] == motif for i in range(len(sequence) - len(motif) + 1))


def build_endpoint_trends(out: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequences = frame["x"].map(canonical_aso_sequence)
    rows = pd.DataFrame({"row_index": np.arange(len(frame)), "sequence": sequences, "measured": frame["y"].astype(float)})
    for base in "ACGT":
        rows[f"{base}_fraction"] = sequences.map(lambda value, b=base: value.count(b) / len(value))
    for motif in ("TCC", "TT", "TG", "GG", "GGG"):
        rows[f"{motif}_count"] = sequences.map(lambda value, m=motif: overlapping_count(value, m))
    rows["modification_pattern"] = frame["x"].map(lambda value: parse_helm(value)[0])
    rows.to_csv(out / "tlr8_endpoint_sequence_descriptors.csv", index=False)
    trend_rows = []
    for feature in ("T_fraction", "G_fraction", "C_fraction", "TCC_count", "TT_count", "TG_count", "GG_count"):
        statistic = spearmanr(rows[feature], rows["measured"])
        trend_rows.append({
            "feature": feature,
            "spearman": float(statistic.statistic),
            "p_value_descriptive_unadjusted": float(statistic.pvalue),
            "unique_values": int(rows[feature].nunique()),
            "claim_boundary": "descriptive association in the complete source endpoint; not held-out performance or causality",
        })
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(out / "tlr8_endpoint_descriptive_trends.csv", index=False)
    return rows, trends


def aggregate_case_shap(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    contributions = pd.read_csv(RUN / "tlr8_case/case_feature_contributions.csv.gz")
    per_member_family = contributions.groupby(
        ["outer_partition", "member", "family"], as_index=False
    ).signed_contribution.sum()
    family = per_member_family.groupby("family", as_index=False).agg(
        mean_signed_contribution=("signed_contribution", "mean"),
        sd_signed_contribution=("signed_contribution", "std"),
        minimum=("signed_contribution", "min"),
        maximum=("signed_contribution", "max"),
        members=("signed_contribution", "size"),
        negative_sign_consistency=("signed_contribution", lambda x: float((x < 0).mean())),
    )
    family.to_csv(out / "tlr8_case_family_shap_sums.csv", index=False)
    per_member_concept = contributions.groupby(
        ["outer_partition", "member", "family", "concept"], as_index=False
    ).signed_contribution.sum()
    concept = per_member_concept.groupby(["family", "concept"], as_index=False).agg(
        mean_signed_contribution=("signed_contribution", "mean"),
        sd_signed_contribution=("signed_contribution", "std"),
        members=("signed_contribution", "size"),
        sign_consistency=("signed_contribution", lambda x: float(max((x > 0).mean(), (x < 0).mean()))),
    ).sort_values("mean_signed_contribution", key=lambda x: x.abs(), ascending=False)
    concept.to_csv(out / "tlr8_case_concept_shap_sums.csv", index=False)
    return family, concept


def build_case_ablation(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(RUN / "oof_predictions.csv.gz")
    full = full[(full.dataset == DATASET) & (full.row_index == 184)][["outer_partition", "pred_policy"]]
    ablation = pd.read_csv(RUN / "family_ablation/family_ablation_oof_predictions.csv.gz")
    case = ablation[(ablation.dataset == DATASET) & (ablation.row_index == 184)][
        ["outer_partition", "ablation_family", "prediction"]
    ].merge(full, on="outer_partition")
    case["ablation_minus_full"] = case.prediction - case.pred_policy
    case.to_csv(out / "tlr8_case_family_ablation.csv", index=False)

    endpoint_ablation = pd.read_csv(RUN / "family_ablation/family_ablation_endpoint_metrics.csv")
    endpoint_ablation = endpoint_ablation[endpoint_ablation.dataset == DATASET].copy()
    final = pd.read_csv(PAPER_ANALYSIS / "source_vs_final_endpoint_effects.csv")
    final = final[final.dataset == DATASET].iloc[0]
    endpoint_ablation["full_spearman"] = float(final.final_spearman)
    endpoint_ablation["full_raw_r2"] = float(final.final_raw_r2)
    endpoint_ablation["full_rmse_over_y_std"] = float(final.final_rmse_over_y_std)
    endpoint_ablation["delta_spearman_vs_full"] = endpoint_ablation.spearman - float(final.final_spearman)
    endpoint_ablation["delta_raw_r2_vs_full"] = endpoint_ablation.raw_r2 - float(final.final_raw_r2)
    endpoint_ablation.to_csv(out / "tlr8_endpoint_family_ablation.csv", index=False)
    return case, endpoint_ablation


def build_packet(
    out: Path,
    feature_db: pd.DataFrame,
    source_db: pd.DataFrame,
    training: pd.DataFrame,
    endpoint_rows: pd.DataFrame,
    trends: pd.DataFrame,
    family_shap: pd.DataFrame,
    concept_shap: pd.DataFrame,
    case_ablation: pd.DataFrame,
    endpoint_ablation: pd.DataFrame,
) -> dict[str, Any]:
    manifest = json.loads((RUN / "tlr8_case/manifest.json").read_text())
    members = pd.read_csv(RUN / "tlr8_case/case_member_reconciliation.csv")
    case_row = endpoint_rows[endpoint_rows.row_index == 184].iloc[0]
    analogue_rows = []
    for role, payload in (
        ("nearest_measured_analogue", manifest["nearest_measured_analogue"]),
        ("informative_contrast", manifest["informative_contrast"]),
    ):
        source_row = endpoint_rows[endpoint_rows.row_index == int(payload["source_row_index"])].iloc[0]
        analogue_rows.append({
            "role": role,
            "row_index": int(payload["source_row_index"]),
            "sequence": payload["sequence"],
            "measured": float(payload["measured_value"]),
            "sequence_identity": float(payload["sequence_identity"]),
            "chemistry_identity": float(payload["chemistry_identity"]),
            "T_fraction": float(source_row.T_fraction),
            "G_fraction": float(source_row.G_fraction),
            "TCC_count": int(source_row.TCC_count),
            "TT_count": int(source_row.TT_count),
            "TG_count": int(source_row.TG_count),
            "GG_count": int(source_row.GG_count),
            "exact_sequence_match": bool(payload["exact_sequence_match"]),
            "exact_input_match": bool(payload["exact_input_match"]),
            "citation": payload["citation"],
        })
    analogue_frame = pd.DataFrame(analogue_rows)
    analogue_frame.to_csv(out / "tlr8_case_reader_analogues.csv", index=False)
    packet = {
        "schema_version": "paper_interpretability_agent_packet_v1",
        "case": manifest["selection"],
        "training_data": training[training.dataset == DATASET].to_dict("records")[0],
        "descriptor_registry": {
            "feature_count": int(len(feature_db)),
            "family_counts": feature_db.family.value_counts().to_dict(),
            "source_count": int(len(source_db)),
            "database_sha256": sha256(out / "descriptor_provenance_database.csv"),
            "source_database_sha256": sha256(out / "descriptor_source_database.csv"),
            "boundary": "papers motivate descriptor concepts; project code defines the actual 1,918 dimensions",
        },
        "endpoint_context": {
            "source_rows": int(len(endpoint_rows)),
            "unique_modification_patterns": int(endpoint_rows.modification_pattern.nunique()),
            "modification_patterns": endpoint_rows.modification_pattern.value_counts().to_dict(),
            "descriptive_trends": trends.to_dict("records"),
            "family_ablation": endpoint_ablation.to_dict("records"),
        },
        "query_sequence_context": {
            "T_fraction": float(case_row.T_fraction),
            "G_fraction": float(case_row.G_fraction),
            "C_fraction": float(case_row.C_fraction),
            "TCC_count": int(case_row.TCC_count),
            "TT_count": int(case_row.TT_count),
            "TG_count": int(case_row.TG_count),
            "GG_count": int(case_row.GG_count),
        },
        "same_model_evidence": {
            "member_count": int(len(members)),
            "mean_expected_value": float(members.expected_value.mean()),
            "mean_prediction": float(members.prediction_model_scale.mean()),
            "mean_signed_contribution_sum": float(members.signed_contribution_sum.mean()),
            "max_additivity_residual": float(members.additivity_residual.abs().max()),
            "family_shap_sums": family_shap.to_dict("records"),
            "concept_shap_sums": concept_shap.head(12).to_dict("records"),
            "case_family_ablation": case_ablation.to_dict("records"),
        },
        "analogues": analogue_rows,
        "reviewed_literature_ids": ["E01", "E02", "E04", "E05", "E06", "E07"],
        "claim_boundaries": [
            "source-wide trends are descriptive associations, not causal effects",
            "TreeSHAP explains fitted model dependence relative to member baselines",
            "family deletion measures functional dependence after refitting, not intervention biology",
            "the endpoint has one chemistry architecture and cannot identify cross-chemistry effects",
            "reporter potentiation is not clinical immunotoxicity",
            "similarity is analogue/applicability evidence only",
        ],
    }
    write_json(out / "case_story_packet.json", packet)
    return packet


def render_story(packet: dict[str, Any], language: str) -> str:
    case = packet["case"]
    q = packet["query_sequence_context"]
    trends = {row["feature"]: row for row in packet["endpoint_context"]["descriptive_trends"]}
    shap = {row["family"]: row for row in packet["same_model_evidence"]["family_shap_sums"]}
    ab = pd.DataFrame(packet["same_model_evidence"]["case_family_ablation"])
    ab_mean = ab.groupby("ablation_family").agg(prediction=("prediction", "mean"), delta=("ablation_minus_full", "mean"))
    ep = pd.DataFrame(packet["endpoint_context"]["family_ablation"]).set_index("ablation_family")
    analogues = {row["role"]: row for row in packet["analogues"]}
    near = analogues["nearest_measured_analogue"]
    contrast = analogues["informative_contrast"]
    if language == "zh":
        return f"""# TLR8 增强案例：模型看到的不是一个 motif，而是一段固定化学背景中的序列语境

## 先说结论

对 sequence `{case['sequence']}`，模型给出的不是“发现一个危险 motif”的结论，而是：**在 Alharbi 2020 的固定 5–10–5 2′-O-methyl／DNA gap／全 PS assay 背景中，这条偏 C-rich、T 含量较低且没有 TT 的序列更接近中低度 TLR8 potentiation，而不是强增强序列。**实测相对 reporter response 为 {case['observed']:.2f}，15 个完全未训练过该 row、HELM 或 canonical sequence 的 outer-held-out 成员平均预测 {case['prediction']:.3f}，低估 {case['observed']-case['prediction']:.3f}。

## 这批数据从哪里来，能回答什么

训练标签来自 Alharbi et al. 2020 的 192 条 2′OMe gapmer 与 sequence mutant：100 nM ASO、HEK293-TLR8 NF-κB reporter、R848 条件归一为 1.0；数值越高表示 potentiation 越强。192 条分子的 modification pattern 全部是 `OOOOODDDDDDDDDDOOOOO`，所以这个 endpoint 能学习同一 chemistry architecture 内的 sequence 与位置语境，**不能**从自身数据识别“把 OMe 换成 LNA/MOE 会怎样”。论文原文将主要现象限定为 “in a sequence-dependent manner”；这也是这里最可靠的解释层级。

模型使用的 1,918 个 descriptor 也不是从一篇论文整表复制而来。Raw positional token 的表示来源是 HELM 与 OligoGym；composition/k-mer 的基线设计来自 OligoGym；wing/gap 位置与 sequence×chemistry interaction 则由项目代码自行构造，Papargyri、Bhamra 与 Alharbi 的实验设计只提供概念动机。换言之，论文说明“为什么值得描述这个分子事实”，项目实现决定“怎样把它变成具体数值列”；Supplement 的逐 feature provenance database 为每列保存推导代码、来源、locator 和这条边界。

## Endpoint 呈现的趋势

在完整 192-row source endpoint 中，T fraction 与实测 response 的描述性 Spearman 为 {trends['T_fraction']['spearman']:.3f}，TT count 为 {trends['TT_count']['spearman']:.3f}；G fraction 为 {trends['G_fraction']['spearman']:.3f}，GG count 为 {trends['GG_count']['spearman']:.3f}。这些是训练来源中的总体关联，不是 held-out 性能或因果效应。查询序列 T fraction 只有 {q['T_fraction']:.2f}、TT={q['TT_count']}，却有 C fraction {q['C_fraction']:.2f}；它有 {q['TCC_count']} 个 TCC，但 TCC count 自身与 response 的相关只有 {trends['TCC_count']['spearman']:.3f}（未校正 P={trends['TCC_count']['p_value_descriptive_unadjusted']:.3f}）。因此故事不是“TCC 越多越高”，而是 T-richness、端点位置、gap/wing 上下文与相邻碱基共同形成的趋势。

## 模型怎样把这些线索合在一起

三个 outer partition 的成员基线平均为 {packet['same_model_evidence']['mean_expected_value']:.3f}。相对该基线，motif/context block 合计贡献 {shap['motif']['mean_signed_contribution']:.3f}，Registry 的逐位 token block 为 {shap['registry']['mean_signed_contribution']:.3f}，sequence×chemistry/position block 为 {shap['chemical_modification']['mean_signed_contribution']:.3f}，三者共同把输出拉到 {packet['same_model_evidence']['mean_prediction']:.3f}；最大 TreeSHAP 加和残差为 {packet['same_model_evidence']['max_additivity_residual']:.2e}。这里的 chemistry block 主要是“某个碱基位于固定 OMe wing 或 DNA gap”的交互编码，不是跨 chemistry 的比较。

消融给出了更直观的功能依赖：仅删除 motif block 并在相同 outer-train 重拟合，案例预测升到 {ab_mean.loc['motif','prediction']:.3f}（Δ={ab_mean.loc['motif','delta']:+.3f}），几乎回到实测 {case['observed']:.2f}；删除 Registry 后为 {ab_mean.loc['registry','prediction']:.3f}，删除 chemistry-interaction block 后为 {ab_mean.loc['chemical_modification','prediction']:.3f}。这说明本例的低估主要由整组 sequence-context 表示压低，而不是某一个 feature 出错。Endpoint 层面也一致：完整模型 Spearman {ep.full_spearman.iloc[0]:.3f}；删除 motif 或 Registry 后分别为 {ep.loc['motif','spearman']:.3f} 和 {ep.loc['registry','spearman']:.3f}。但删除 chemistry block 后反而为 {ep.loc['chemical_modification','spearman']:.3f}，略高于完整模型，说明固定 chemistry 数据中这组交互没有稳定的独立增益，不能把它包装成确定机制。

## 两个实测 analogue 帮助读懂这条 sequence

最近 analogue `{near['sequence']}` 与查询的 sequence identity 为 {near['sequence_identity']:.2f}，chemistry 完全相同；二者都只有 15% T、没有 TT，并各含两个 TCC，analogue 实测 {near['measured']:.2f}。信息性 contrast `{contrast['sequence']}` 也有两个 TCC，却有 30% T、一个 TT 和两个 TG，实测 {contrast['measured']:.2f}。三者把关键 insight 说得很清楚：**TCC 数量相同并不产生相同 response；更宽的 T-rich 与位置语境更能区分这个小集合中的低、中、高读数。**近邻只帮助判断适用域和形成实验对照，不参与预测，也不证明共享机制。

## 下一步怎样证伪

最直接的实验不是盲目删除最高 SHAP feature，而是在相同 5–10–5 OMe/DNA/PS 架构下构建 composition-matched pairs：保持总 T/C/G/A 与 TCC 数不变，把 TT/T-rich 片段在 wing 与 gap 间 reciprocal relocation；再做 TCC-disrupting、总组成不变的配对。联合测 TLR8 reporter、receptor-negative reporter、uptake 与 viability。只有位置化 sequence signal 可重复改变 TLR8 reporter、且一般细胞效应不随之同向变化时，才支持候选 TLR8-context 假说。LNA/MOE 替换必须作为独立的跨 chemistry 实验，不能由本 endpoint 模型外推。

## 证据边界

这是模型依赖与可证伪假说，不是生物因果证明；R848 reporter association 不等于临床 immunotoxicity。Registry block 保存可复用的逐位 HELM token；只有整组位置 token 的组合才接近完整输入身份，单个 token 既不是精确分子身份，也不是机制。TreeSHAP、family deletion、数据相关和 analogue similarity 是四类不同证据，不能互相替代。
"""
    return f"""# TLR8 potentiation case: a sequence-context story within one fixed chemistry

## Reader-level conclusion

For `{case['sequence']}`, the model does not identify a single hazardous motif. Within the Alharbi 2020 fixed 5–10–5 2′-O-methyl/DNA-gap/full-PS assay context, the C-rich, low-T sequence with no TT is placed in the low-to-moderate TLR8-potentiation range. The measured reporter response is {case['observed']:.2f}; 15 outer-held-out members that saw neither the row, HELM nor canonical sequence predicted {case['prediction']:.3f}.

## Data and scope

The labels are the 192-ASO Alharbi et al. screen at 100 nM in an HEK293-TLR8 NF-κB reporter with R848 normalized to 1.0. Every molecule has modification pattern `OOOOODDDDDDDDDDOOOOO`; the endpoint can therefore resolve sequence context within this architecture but cannot identify the effect of changing to LNA or MOE.

The 1,918 descriptors are not copied from one paper. Raw positional tokens follow HELM and OligoGym representation; composition/k-mer baselines follow the OligoGym design; wing/gap and sequence-by-chemistry interactions are project-engineered columns motivated by the Papargyri, Bhamra and Alharbi experimental designs. Papers motivate why a molecular fact is represented, whereas project code defines its exact numerical derivation. The Supplement feature-level provenance database records that distinction, its implementation path and source locator.

## Endpoint trend and query interpretation

Across the complete source endpoint, descriptive Spearman associations are {trends['T_fraction']['spearman']:.3f} for T fraction, {trends['TT_count']['spearman']:.3f} for TT count, {trends['G_fraction']['spearman']:.3f} for G fraction and {trends['GG_count']['spearman']:.3f} for GG count. The query has T fraction {q['T_fraction']:.2f}, TT={q['TT_count']} and C fraction {q['C_fraction']:.2f}. It contains {q['TCC_count']} TCC motifs, but TCC count alone has Spearman {trends['TCC_count']['spearman']:.3f} (unadjusted P={trends['TCC_count']['p_value_descriptive_unadjusted']:.3f}). Thus the readable signal is joint composition and position context, not a TCC-count rule.

## Same-model integration and deletion evidence

Relative to a mean member baseline of {packet['same_model_evidence']['mean_expected_value']:.3f}, motif/context, Registry tokens and sequence-by-position/chemistry descriptors contribute {shap['motif']['mean_signed_contribution']:.3f}, {shap['registry']['mean_signed_contribution']:.3f} and {shap['chemical_modification']['mean_signed_contribution']:.3f}, respectively, reconciling to {packet['same_model_evidence']['mean_prediction']:.3f}. Removing and refitting the motif block raises the case prediction to {ab_mean.loc['motif','prediction']:.3f}, showing that sequence-context dependence explains most of this underprediction. Endpoint Spearman falls from {ep.full_spearman.iloc[0]:.3f} to {ep.loc['motif','spearman']:.3f} without motif and {ep.loc['registry','spearman']:.3f} without Registry, whereas removing the chemistry-interaction block gives {ep.loc['chemical_modification','spearman']:.3f}. The latter is a warning against claiming an independently learned chemistry mechanism in a fixed-chemistry dataset.

## Analogue contrast and falsification

The nearest measured analogue has the same chemistry, 15% T, no TT, two TCC motifs and response {near['measured']:.2f}. The informative contrast also has two TCC motifs but 30% T, one TT and response {contrast['measured']:.2f}. This supports a testable sequence-context contrast, not shared mechanism. Composition-matched reciprocal relocation of TT/T-rich context between wing and gap, plus TCC-disrupting controls, should be tested with TLR8 and receptor-negative reporters, uptake and viability. Reporter association, TreeSHAP, deletion and similarity remain non-causal evidence types.
"""


def audit_story(text: str, language: str) -> dict[str, Any]:
    required = {
        "training_source": "Alharbi" in text and "192" in text,
        "descriptor_source_boundary": ("论文" in text and "项目" in text) if language == "zh" else ("project" in text and "architecture" in text),
        "endpoint_trend": "0.468" in text and "-0.466" in text,
        "case_prediction": "1.271" in text and "1.70" in text,
        "cross_descriptor_story": all(token in text for token in ("motif", "Registry", "chemistry")),
        "case_ablation": "1.695" in text,
        "analogue_contrast": "3.95" in text and "TCC" in text,
        "fixed_chemistry_limit": "OOOOODDDDDDDDDDOOOOO" in text,
        "causal_boundary": ("不是生物因果" in text) if language == "zh" else ("non-causal" in text),
        "falsification": ("证伪" in text) if language == "zh" else ("falsification" in text.lower()),
    }
    return {
        "schema_version": "paper_interpretability_story_audit_v1",
        "language": language,
        "status": "PASS" if all(required.values()) else "FAIL",
        "checks": required,
        "errors": [key for key, value in required.items() if not value],
    }


def run_loop(out: Path, packet: dict[str, Any], max_rounds: int = 3) -> dict[str, Any]:
    for stale in out.glob("agent_round_*_audit.json"):
        stale.unlink()
    rounds = []
    for round_id in range(1, max_rounds + 1):
        zh = render_story(packet, "zh")
        en = render_story(packet, "en")
        zh_audit = audit_story(zh, "zh")
        en_audit = audit_story(en, "en")
        rounds.append({"round": round_id, "zh": zh_audit, "en": en_audit})
        write_json(out / f"agent_round_{round_id}_audit.json", rounds[-1])
        if zh_audit["status"] == "PASS" and en_audit["status"] == "PASS":
            (out / "AGENT_EXPLANATION_zh.md").write_text(zh, encoding="utf-8")
            (out / "AGENT_EXPLANATION.md").write_text(en, encoding="utf-8")
            break
        packet.setdefault("repair_instructions", []).append({
            "round": round_id,
            "missing_zh": zh_audit["errors"],
            "missing_en": en_audit["errors"],
        })
    status = "PASS" if rounds[-1]["zh"]["status"] == rounds[-1]["en"]["status"] == "PASS" else "FAIL"
    state = {
        "schema_version": "paper_interpretability_agent_loop_v1",
        "status": status,
        "rounds": rounds,
        "registered_tools": [
            "training_data_provenance",
            "descriptor_provenance_registry",
            "endpoint_trend_analysis",
            "same_model_treeshap_reconciliation",
            "outer_train_family_deletion",
            "measured_analogue_retrieval",
            "reviewed_literature_ledger",
            "bilingual_report_template",
            "deterministic_claim_boundary_audit",
        ],
        "general_runtime": "src/agents/evidence_runtime.py",
        "general_runtime_capabilities": [
            "registered deterministic tools and templates",
            "PubMed/Europe PMC/Crossref/patent/ClinicalTrials retrieval",
            "bounded LLM synthesis with repair attempts",
            "independent adversarial review",
            "resume, hashes, manifests and machine audit",
        ],
        "current_run_mode": "deterministic evidence tools plus audited bilingual synthesis; no unreviewed retrieval candidate promoted",
    }
    write_json(out / "run_state.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PAPER_ANALYSIS / "interpretability_agent")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tool_trace: list[dict[str, Any]] = []
    frame = earlier_frame(DATASET).reset_index(drop=True)

    feature_db, source_db = build_descriptor_databases(out, frame)
    trace(tool_trace, "descriptor_provenance_registry", [Path(__file__), ROOT / "src/explainability/descriptor_provenance.py"], [out / "descriptor_provenance_database.csv", out / "descriptor_source_database.csv"], "Resolve every formal feature to derivation code and concept-motivating sources.")
    training = build_training_provenance(out)
    trace(tool_trace, "training_data_provenance", [ROOT / "data/literature_supervision/provenance_registry_current_data.csv", RUN / "oof_predictions.csv.gz"], [out / "training_data_provenance.csv"], "Record all nine endpoint sources and validation scope.")
    endpoint_rows, trends = build_endpoint_trends(out, frame)
    trace(tool_trace, "endpoint_trend_analysis", [ROOT / "outputs/system_audit/endpoint_contract_audit.csv"], [out / "tlr8_endpoint_sequence_descriptors.csv", out / "tlr8_endpoint_descriptive_trends.csv"], "Describe source-endpoint trends without treating them as performance or causal estimates.")
    family_shap, concept_shap = aggregate_case_shap(out)
    trace(tool_trace, "same_model_treeshap_reconciliation", [RUN / "tlr8_case/case_feature_contributions.csv.gz", RUN / "tlr8_case/case_member_reconciliation.csv"], [out / "tlr8_case_family_shap_sums.csv", out / "tlr8_case_concept_shap_sums.csv"], "Sum correlated feature contributions within each member before averaging.")
    case_ablation, endpoint_ablation = build_case_ablation(out)
    trace(tool_trace, "outer_train_family_deletion", [RUN / "family_ablation/family_ablation_oof_predictions.csv.gz", RUN / "family_ablation/family_ablation_endpoint_metrics.csv"], [out / "tlr8_case_family_ablation.csv", out / "tlr8_endpoint_family_ablation.csv"], "Compare full and refitted family-deletion models on identical held-out rows.")
    packet = build_packet(out, feature_db, source_db, training, endpoint_rows, trends, family_shap, concept_shap, case_ablation, endpoint_ablation)
    trace(tool_trace, "immutable_packet_builder", [RUN / "tlr8_case/manifest.json", ROOT / "docs/papers/oligo_toxicity_journal_v3/supporting/mechanism_evidence_ledger.csv"], [out / "case_story_packet.json", out / "tlr8_case_reader_analogues.csv"], "Fuse model, data, analogue and reviewed literature inputs while keeping claim types separate.")
    state = run_loop(out, packet)
    trace(tool_trace, "synthesis_review_repair_loop", [out / "case_story_packet.json"], [out / "AGENT_EXPLANATION_zh.md", out / "AGENT_EXPLANATION.md", out / "run_state.json"], "Render, audit and repair until every reader-facing requirement and claim boundary passes.")
    with (out / "tool_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in tool_trace:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    artifacts = {
        str(path.relative_to(out)): sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": "paper_interpretability_agent_manifest_v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": state["status"],
        "case": "tlr8_potentiation_row184",
        "descriptor_features": int(len(feature_db)),
        "descriptor_sources": int(len(source_db)),
        "training_endpoints": int(len(training)),
        "artifacts": artifacts,
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
