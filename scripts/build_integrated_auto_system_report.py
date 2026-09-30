#!/usr/bin/env python3
"""Build bilingual reports, figures, audits, and bundle for the auto system."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/integrated_auto_system_v1/formal_v2"
OUTER_EXPLANATIONS = RUN / "outer_explanations"
STAGES = ROOT / "outputs/integrated_auto_system_v1/stages_v2"
OUT = ROOT / "outputs/integrated_auto_system_v1/report"
DOC = ROOT / "docs/reports/INTEGRATED_AUTO_PREDICTION_EXPLANATION_SYSTEM.md"
DOC_ZH = ROOT / "docs/reports/INTEGRATED_AUTO_PREDICTION_EXPLANATION_SYSTEM_zh.md"
FIG = ROOT / "docs/papers/oligo_toxicity_journal_v1/figures/integrated_auto_stage_gain.png"
FIG_ZH = ROOT / "docs/papers/oligo_toxicity_journal_v1/figures/integrated_auto_stage_gain_zh.png"
BUNDLE = ROOT / "INTEGRATED_AUTO_PREDICTION_EXPLANATION_SYSTEM_BUNDLE.zip"
OLIGO_SOURCE_OOF = ROOT / "outputs/oligogym_neurips2025_same_split/same_split_predictions.csv"
OLIGO_PROJECT_OOF = ROOT / "outputs/oligogym_source_features_project_xgb_v1/same_split_predictions.csv"
ATLAS_SOURCE_OOF = ROOT / "outputs/aso_atlas2_four_stage_factorial_v1/full/oof_predictions.csv.gz"

ORDER = (
    "cytotoxicity", "neurotoxicity_lna", "neurotoxicity_moe",
    "tlr7_immunotox", "tlr8_immunotox", "mouse_hepatic",
    "rat_hepatic", "mouse_neuro", "rat_neuro",
)
LABELS = {
    "cytotoxicity": "HeLa cytotoxicity",
    "neurotoxicity_lna": "LNA neuronal calcium",
    "neurotoxicity_moe": "MOE mouse ICV FOB",
    "tlr7_immunotox": "TLR8 potentiation",
    "tlr8_immunotox": "TLR7 inhibition",
    "mouse_hepatic": "Mouse ALT",
    "rat_hepatic": "Rat ALT",
    "mouse_neuro": "Mouse FOB",
    "rat_neuro": "Rat FOB",
}
LABELS_ZH = {
    "cytotoxicity": "HeLa 细胞毒性",
    "neurotoxicity_lna": "LNA 神经元钙",
    "neurotoxicity_moe": "MOE 小鼠 ICV FOB",
    "tlr7_immunotox": "TLR8 增强",
    "tlr8_immunotox": "TLR7 抑制",
    "mouse_hepatic": "小鼠 ALT",
    "rat_hepatic": "大鼠 ALT",
    "mouse_neuro": "小鼠 FOB",
    "rat_neuro": "大鼠 FOB",
}
STAGE_LABELS = {
    "source_native": "Protocol-aligned published-XGB/author-XGB composite rerun",
    "project_xgb": "Project XGBoost policy",
    "registry": "Registry One-Hot",
    "position": "Registry + position",
    "second_branch": "Chemistry-context branch",
    "fixed_fusion": "Fixed 50:50 fusion",
    "automatic_system": "Automatic scale + capacity",
}
STAGE_LABELS_ZH = {
    "source_native": "协议对齐的 published-XGB/作者-XGB 复合重跑",
    "project_xgb": "项目 XGBoost 策略",
    "registry": "Registry One-Hot",
    "position": "Registry + 位置",
    "second_branch": "化学背景第二分支",
    "fixed_fusion": "固定 50:50 融合",
    "automatic_system": "自动尺度 + 容量",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_metrics() -> pd.DataFrame:
    rerun = pd.read_csv(STAGES / "summary.csv")
    auto = pd.read_csv(RUN / "summary.csv")
    source_author = pd.read_csv(OLIGO_SOURCE_OOF)
    source_author = source_author[
        source_author.split_mode.eq("nucleobase_holdout")
        & source_author.route.eq("oligogym_published_xgb")
    ]
    source_project = pd.read_csv(OLIGO_PROJECT_OOF)
    source_project = source_project[
        source_project.split_mode.eq("nucleobase_holdout")
        & source_project.route.eq("oligogym_published_features_project_xgb")
    ]
    atlas_source = pd.read_csv(ATLAS_SOURCE_OOF)
    rows: list[dict[str, object]] = []
    for dataset in ORDER:
        if dataset in ORDER[:5]:
            for stage, source in (("source_native", source_author), ("project_xgb", source_project)):
                block = source[source.project_dataset.eq(dataset)]
                fold_metrics = [
                    (float(spearmanr(group.y, group.prediction).statistic), float(r2_score(group.y, group.prediction)))
                    for _, group in block.groupby("split_id")
                ]
                rows.append({"dataset": dataset, "stage": stage, "spearman": float(np.mean([x[0] for x in fold_metrics])), "raw_r2": float(np.mean([x[1] for x in fold_metrics])), "evidence_origin": "project-recomputed from source-protocol held-out predictions"})
        else:
            block = atlas_source[atlas_source.task.eq(dataset)]
            for stage, column in (("source_native", "pred_author_ohe_author_xgb"), ("project_xgb", "pred_author_ohe_project_xgb")):
                fold_metrics = [
                    (float(spearmanr(group.y, group[column]).statistic), float(r2_score(group.y, group[column])))
                    for _, group in block.groupby("outer_fold")
                ]
                rows.append({"dataset": dataset, "stage": stage, "spearman": float(np.mean([x[0] for x in fold_metrics])), "raw_r2": float(np.mean([x[1] for x in fold_metrics])), "evidence_origin": "project-recomputed from source-protocol held-out predictions"})
        stage_oof = pd.read_csv(STAGES / "oof_predictions.csv.gz")
        block = stage_oof[stage_oof.dataset.eq(dataset)]
        partition_column = "repeat" if dataset in ORDER[:5] else "outer_fold"
        for stage, prediction_column in (
            ("registry", "pred_onehot"),
            ("position", "pred_branch_a"),
            ("second_branch", "pred_branch_b"),
            ("fixed_fusion", "pred_descriptor"),
        ):
            fold_metrics = [
                (
                    float(spearmanr(group.y, group[prediction_column]).statistic),
                    float(r2_score(group.y, group[prediction_column])),
                )
                for _, group in block.groupby(partition_column)
            ]
            rows.append({
                "dataset": dataset,
                "stage": stage,
                "spearman": float(np.mean([value[0] for value in fold_metrics])),
                "raw_r2": float(np.mean([value[1] for value in fold_metrics])),
                "evidence_origin": "project-recomputed as five partition-level metrics from formal v2 stage OOF",
            })
        row = auto[auto.dataset.eq(dataset) & auto.route.eq("nested_one_se_seed_mean")].iloc[0]
        rows.append({"dataset": dataset, "stage": "automatic_system", "spearman": row.mean_spearman, "raw_r2": row.mean_r2, "evidence_origin": "project-computed strict nested auto run"})
    frame = pd.DataFrame(rows)
    frame["endpoint"] = frame.dataset.map(LABELS)
    frame["stage_label"] = frame.stage.map(STAGE_LABELS)
    return frame


def _scale_metrics() -> pd.DataFrame:
    oof = pd.read_csv(RUN / "oof_predictions.csv.gz")
    rows: list[dict[str, object]] = []
    for (dataset, partition), group in oof.groupby(["dataset", "outer_partition"]):
        y = group.y.to_numpy(dtype=float)
        pred = group.pred_nested_one_se_seed_mean.to_numpy(dtype=float)
        transform = str(group.selected_transform.iloc[0])
        if group.selected_transform.nunique() != 1:
            raise RuntimeError("a partition has inconsistent selected transforms")
        model_y = np.log1p(y) if transform == "log1p" else y
        model_pred = np.log1p(pred) if transform == "log1p" else pred
        rho = float(spearmanr(y, pred).statistic)
        rows.append({
            "dataset": dataset,
            "outer_partition": partition,
            "selected_transform": transform,
            "spearman": rho if np.isfinite(rho) else -1.0,
            "raw_scale_r2": float(r2_score(y, pred)),
            "model_scale_r2": float(r2_score(model_y, model_pred)),
            "model_scale": "log1p(y)" if transform == "log1p" else "y",
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "y_std": float(np.std(y, ddof=0)),
        })
    folds = pd.DataFrame(rows)
    folds["rmse_over_y_std"] = folds.rmse / folds.y_std.replace(0.0, np.nan)
    summary = folds.groupby("dataset", as_index=False).agg(
        partitions=("outer_partition", "nunique"),
        identity_partitions=("selected_transform", lambda x: int((x == "identity").sum())),
        log1p_partitions=("selected_transform", lambda x: int((x == "log1p").sum())),
        mean_spearman=("spearman", "mean"),
        mean_raw_scale_r2=("raw_scale_r2", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_y_std=("y_std", "mean"),
        mean_rmse_over_y_std=("rmse_over_y_std", "mean"),
    )
    identity_r2 = folds[folds.selected_transform.eq("identity")].groupby("dataset").model_scale_r2.mean()
    log1p_r2 = folds[folds.selected_transform.eq("log1p")].groupby("dataset").model_scale_r2.mean()
    summary["mean_identity_model_scale_r2"] = summary.dataset.map(identity_r2)
    summary["mean_log1p_model_scale_r2"] = summary.dataset.map(log1p_r2)
    summary["endpoint"] = summary.dataset.map(LABELS)
    folds.to_csv(OUT / "fold_scale_metrics.csv", index=False)
    return summary


def _deltas(stages: pd.DataFrame) -> pd.DataFrame:
    pivot = stages.pivot(index="dataset", columns="stage", values="spearman")
    result = pd.DataFrame({
        "dataset": pivot.index,
        "model_policy": pivot.project_xgb - pivot.source_native,
        "registry_representation": pivot.registry - pivot.project_xgb,
        "position": pivot.position - pivot.registry,
        "branch_b_vs_branch_a": pivot.second_branch - pivot.position,
        "fusion_over_branch_a": pivot.fixed_fusion - pivot.position,
        "automatic_scale_and_capacity": pivot.automatic_system - pivot.fixed_fusion,
        "total_system": pivot.automatic_system - pivot.source_native,
    }).reset_index(drop=True)
    result["endpoint"] = result.dataset.map(LABELS)
    return result


def _alignment_audit() -> pd.DataFrame:
    stage = pd.read_csv(STAGES / "oof_predictions.csv.gz")
    oligo = pd.read_csv(OLIGO_SOURCE_OOF)
    oligo = oligo[
        oligo.split_mode.eq("nucleobase_holdout")
        & oligo.route.eq("oligogym_published_xgb")
    ]
    atlas = pd.read_csv(ATLAS_SOURCE_OOF)
    automatic = pd.read_csv(RUN / "oof_predictions.csv.gz")
    rows: list[dict[str, object]] = []
    for dataset in ORDER[:5]:
        for partition in range(1, 6):
            left = set(oligo[oligo.project_dataset.eq(dataset) & oligo.split_id.eq(partition)].row_index.astype(int))
            right = set(stage[stage.dataset.eq(dataset) & stage.repeat.eq(partition)].row_index.astype(int))
            auto = set(automatic[automatic.dataset.eq(dataset) & automatic.outer_partition.eq(partition)].row_index.astype(int))
            rows.append({"dataset": dataset, "partition": partition, "source_stage_equal": left == right, "stage_auto_equal": right == auto, "all_three_equal": left == right == auto, "source_rows": len(left), "stage_rows": len(right), "auto_rows": len(auto)})
    for dataset in ORDER[5:]:
        for partition in range(1, 6):
            left = set(atlas[atlas.task.eq(dataset) & atlas.outer_fold.eq(partition)].row_index.astype(int))
            right = set(stage[stage.dataset.eq(dataset) & stage.outer_fold.eq(partition)].row_index.astype(int))
            auto = set(automatic[automatic.dataset.eq(dataset) & automatic.outer_partition.eq(partition)].row_index.astype(int))
            rows.append({"dataset": dataset, "partition": partition, "source_stage_equal": left == right, "stage_auto_equal": right == auto, "all_three_equal": left == right == auto, "source_rows": len(left), "stage_rows": len(right), "auto_rows": len(auto)})
    return pd.DataFrame(rows)


def _figure(macro: pd.DataFrame, path: Path, chinese: bool) -> None:
    labels = STAGE_LABELS_ZH if chinese else STAGE_LABELS
    if chinese:
        font_manager.fontManager.addfont(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
        )
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    values = [float(macro.loc[stage, "mean_spearman"]) for stage in STAGE_LABELS]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    x = np.arange(len(values))
    ax.plot(x, values, marker="o", lw=2.4, color="#24788c")
    ax.fill_between(x, values, min(values) - 0.03, alpha=0.1, color="#43a49b")
    ax.set_xticks(x, [labels[key] for key in STAGE_LABELS], rotation=18, ha="right")
    ax.set_ylabel("终点等权平均 Spearman" if chinese else "Endpoint-equal mean Spearman")
    ax.set_title("一体化系统阶段增益" if chinese else "Integrated-system stage gains", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    for i, value in enumerate(values):
        ax.text(i, value + 0.006, f"{value:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _report(stages: pd.DataFrame, macro: pd.DataFrame, scales: pd.DataFrame, deltas: pd.DataFrame, selections: pd.DataFrame, deployment: pd.DataFrame, chinese: bool) -> str:
    source = float(macro.loc["source_native", "mean_spearman"])
    final = float(macro.loc["automatic_system", "mean_spearman"])
    total = final - source
    fixed = float(macro.loc["fixed_fusion", "mean_spearman"])
    automatic_delta = final - fixed
    improved_over_fixed = int(
        (
            stages.pivot(index="dataset", columns="stage", values="spearman")["automatic_system"]
            > stages.pivot(index="dataset", columns="stage", values="spearman")["fixed_fusion"]
        ).sum()
    )
    log_endpoints = scales[scales.log1p_partitions.gt(0)]
    counts = selections.groupby(["selected_transform", "selected_candidate"]).size().reset_index(name="folds")
    if chinese:
        endpoint_table = stages.pivot(index="dataset", columns="stage", values="spearman").loc[list(ORDER)].reset_index()
        endpoint_table["终点"] = endpoint_table.dataset.map(LABELS_ZH)
        endpoint_table = endpoint_table[["终点", *STAGE_LABELS]].rename(columns=STAGE_LABELS_ZH)
        scale_table = scales.copy()
        scale_table["终点"] = scale_table.dataset.map(LABELS_ZH)
        scale_table = scale_table[["终点", "identity_partitions", "log1p_partitions", "mean_spearman", "mean_raw_scale_r2", "mean_rmse", "mean_y_std", "mean_rmse_over_y_std", "mean_identity_model_scale_r2", "mean_log1p_model_scale_r2"]].rename(columns={"identity_partitions":"identity 折", "log1p_partitions":"log1p 折", "mean_spearman":"Spearman", "mean_raw_scale_r2":"原尺度 R²", "mean_rmse":"RMSE", "mean_y_std":"折内 y 标准差", "mean_rmse_over_y_std":"RMSE/y 标准差", "mean_identity_model_scale_r2":"identity 模型尺度 R²", "mean_log1p_model_scale_r2":"log1p 模型尺度 R²"})
        return f"""# 九终点一体化自动预测与解释系统

## 决策摘要

**项目代码计算结果：**完整自动系统相对明确定义的“共同 XGBoost 复合重跑”阶梯起点，终点等权平均 Spearman 为 `{source:.4f}→{final:.4f}`（`{total:+.4f}`）。这不是 OligoGym 论文按终点选择的最佳模型表，也不是单一来源原生基线。相同 45 个分区、相同折均值估计目标下，固定融合与自动路线为 `{fixed:.4f}→{final:.4f}`（`{automatic_delta:+.4f}`），自动路线在 `{improved_over_fixed}/9` 个终点改善 Spearman，原尺度 R²则小幅改善。自动选择在所用分组契约下隔离外层测试标签，不能宣称带来明确的排序增益。总差异不能只归因于描述符。

## 系统阶梯

{endpoint_table.round(4).to_markdown(index=False)}

![一体化系统阶段增益](../papers/oligo_toxicity_journal_v1/figures/integrated_auto_stage_gain_zh.png)

阶段宏平均 Spearman 为：

{macro.reset_index().rename(columns={"stage":"阶段", "stage_label_zh":"阶段名称", "mean_spearman":"平均 Spearman"})[["阶段名称","平均 Spearman"]].round(4).to_markdown(index=False)}

## 自动选择与尺度审计

非负目标在每个 outer-train 内比较 `identity × 4` 与 `log1p × 4`；预测先逆变换到原尺度，再以 `0.5 × Spearman + 0.5 × clip(原尺度 R²,−1,1)` 评分。one-SE 范围内先选更小容量，同容量平局优先 identity。含负值目标只允许 identity。本轮所有来源目标非负，但代码和测试覆盖负值保护。

{scale_table.round(4).to_markdown(index=False)}

表中 identity 与 log1p 的模型尺度 R² 分列，混合尺度终点不做单一模型尺度平均。使用 log1p 的折中，TreeSHAP 严格位于 `log1p(y)` 模型尺度；`expm1` 仅用于原尺度展示，不被误写为严格可加的原尺度归因。原尺度负 R² 未删除，表示绝对校准未胜过均值尺度基准。

选择计数：

{counts.to_markdown(index=False)}

全训练部署模型重新执行同一分组内层选择。另为 45 个外层分区保存 90 个分支模型及具名特征模式，并从产生预测的同一留出模型直接导出九个案例的分支预测、偏置项/期望值和 TreeSHAP。全部留出行的最大加和残差为 `6.49e-05`，与保存预测的最大差为 `2.28e-13`。

## 组件贡献与解释边界

{deltas.round(4).to_markdown(index=False)}

这些差异是项目代码在固定协议下计算的描述性结果。来源记录事实包括标签、终点定义和来源划分；项目计算值包括预测、指标、选择与 TreeSHAP；“可能的序列/化学机制”仅是科学解释或假设。TreeSHAP、相似序列和文献来源分别表示同模型依赖、类似物/适用域背景和知识来源，均不构成因果证明。

OligoAI 与 ASOptimizer 不进入九终点主基线 或宏平均；相关工作仅按其原始任务和已审计结论讨论。对 Atlas 的限定结论是作者 XGBoost `max_depth=10, n_estimators=500` 在这些小型分组队列中有过容量风险，不泛化为所有 OligoGym 来源模型。

## 局限与验证

九终点来自不同测定、量纲与生物学契约；Spearman 可做描述性终点等权汇总，RMSE 不跨终点宏平均。外层提供的分组变量与 exact HELM 重叠均为零，但 MOE/Atlas 的 canonical 碱基序列仍有跨折重复，部分 Atlas 专利/来源字段也跨折；这是真实泄漏控制局限，不能宣称 sequence/source 全部留出。自动路线的 Spearman 与固定融合近乎持平、原尺度 R²略高，不能宣传为明确的预测性能提升。`expm1` 未作 smearing 校正，因此 log 路线存在重变换偏差风险。
"""
    endpoint_table = stages.pivot(index="dataset", columns="stage", values="spearman").loc[list(ORDER)].reset_index()
    endpoint_table["Endpoint"] = endpoint_table.dataset.map(LABELS)
    endpoint_table = endpoint_table[["Endpoint", *STAGE_LABELS]].rename(columns=STAGE_LABELS)
    scale_table = scales[["endpoint", "identity_partitions", "log1p_partitions", "mean_spearman", "mean_raw_scale_r2", "mean_rmse", "mean_y_std", "mean_rmse_over_y_std", "mean_identity_model_scale_r2", "mean_log1p_model_scale_r2"]].rename(columns={"endpoint":"Endpoint", "identity_partitions":"Identity folds", "log1p_partitions":"Log1p folds", "mean_spearman":"Spearman", "mean_raw_scale_r2":"Raw-scale R²", "mean_rmse":"RMSE", "mean_y_std":"Fold y SD", "mean_rmse_over_y_std":"RMSE/y SD", "mean_identity_model_scale_r2":"Identity model-scale R²", "mean_log1p_model_scale_r2":"Log1p model-scale R²"})
    return f"""# Integrated Automatic Prediction-and-Explanation System Across Nine Endpoints

## Decision summary

**Project-computed result:** endpoint-equal mean Spearman changed from `{source:.4f}` for the explicitly defined common-XGBoost composite rerun to `{final:.4f}` for the complete automatic system (`{total:+.4f}`). This is neither the OligoGym endpoint-selected best-model table nor a single source-native baseline. Under the same 45 partitions and fold-mean estimand, fixed fusion and automatic selection were `{fixed:.4f}→{final:.4f}` (`{automatic_delta:+.4f}`), with Spearman improvement in `{improved_over_fixed}/9` endpoints and a small raw-scale R² improvement. Automatic selection isolates selection from outer-test labels under the supplied grouping contract, without a clear ranking gain. The compound contrast is not descriptor-only.

## System staircase

{endpoint_table.round(4).to_markdown(index=False)}

![Integrated system stage gains](../papers/oligo_toxicity_journal_v1/figures/integrated_auto_stage_gain.png)

{macro.reset_index()[["stage_label", "mean_spearman"]].rename(columns={"stage_label":"Stage", "mean_spearman":"Mean Spearman"}).round(4).to_markdown(index=False)}

## Automatic selection and scale audit

Each nonnegative outer-training target compared `identity × 4` and `log1p × 4`. Inner predictions were inverse-transformed before scoring by `0.5 × Spearman + 0.5 × clip(raw-scale R²,−1,1)`. The one-SE tie break prefers lower capacity, then identity at equal capacity. Negative targets permit identity only.

{scale_table.round(4).to_markdown(index=False)}

Identity and log1p model-scale R² are shown in separate columns; mixed-scale endpoints receive no single model-scale average. TreeSHAP for log1p-selected folds remains strictly on the `log1p(y)` model scale. `expm1` provides original-scale display and is not presented as a strictly additive original-scale attribution. Negative raw-scale R² values remain visible as calibration limitations.

Selection counts:

{counts.to_markdown(index=False)}

Full-data deployment repeated grouped inner selection. In addition, 90 outer branch models and named schemas were saved for 45 partitions; nine cases directly export branch predictions, booster bias/expected values, and TreeSHAP from the same held-out models that generated their predictions. The maximum all-held-out-row additivity residual was `6.49e-05`, and the maximum difference from saved predictions was `2.28e-13`.

## Component audit and claim boundaries

{deltas.round(4).to_markdown(index=False)}

Source-recorded facts, project-computed values, and scientific interpretations are distinct. TreeSHAP describes fitted-model dependence; same-endpoint neighbours describe analogue/applicability context; literature establishes provenance. None proves causality. OligoAI and ASOptimizer remain related work rather than nine-endpoint baselines. The capacity-risk statement is limited to the Atlas author XGBoost (`max_depth=10, n_estimators=500`) on these small grouped cohorts.

## Limitations and verification

Endpoints retain distinct assays, units, and biological contracts. Supplied outer-group and exact-HELM overlap were zero, but canonical base sequences recur across some MOE/Atlas folds and some Atlas patent/source fields cross folds; sequence/source-wide holdout therefore cannot be claimed. The automatic route was essentially tied with fixed fusion in Spearman and slightly higher in raw-scale R²; it must not be advertised as a clear predictive improvement. `expm1` was used without smearing correction, so log-route display may have retransformation bias. Results are retrospective grouped validation, not prospective or causal evidence.
"""


def main() -> None:
    required = [RUN / "summary.csv", RUN / "oof_predictions.csv.gz", RUN / "selections.csv", RUN / "deployment_summary.csv", STAGES / "summary.csv", STAGES / "oof_predictions.csv.gz", OUTER_EXPLANATIONS / "all_outer_reconciliation.csv", OUTER_EXPLANATIONS / "split_overlap_audit.csv", OUTER_EXPLANATIONS / "heldout_cases.csv", OUTER_EXPLANATIONS / "heldout_case_top_contributions.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing formal inputs: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    stages = _stage_metrics()
    macro = stages.groupby("stage").agg(mean_spearman=("spearman", "mean"), mean_raw_r2=("raw_r2", "mean"))
    macro = macro.reindex(STAGE_LABELS)
    macro["stage_label"] = [STAGE_LABELS[key] for key in macro.index]
    macro["stage_label_zh"] = [STAGE_LABELS_ZH[key] for key in macro.index]
    scales = _scale_metrics()
    deltas = _deltas(stages)
    selections = pd.read_csv(RUN / "selections.csv")
    deployment = pd.read_csv(RUN / "deployment_summary.csv")
    alignment = _alignment_audit()
    inner = pd.read_csv(RUN / "inner_scores.csv.gz")
    outer_reconciliation = pd.read_csv(OUTER_EXPLANATIONS / "all_outer_reconciliation.csv")
    split_overlap = pd.read_csv(OUTER_EXPLANATIONS / "split_overlap_audit.csv")
    stages.to_csv(OUT / "endpoint_stage_metrics.csv", index=False)
    macro.reset_index().to_csv(OUT / "stage_macro_metrics.csv", index=False)
    scales.to_csv(OUT / "endpoint_scale_metrics.csv", index=False)
    deltas.to_csv(OUT / "component_spearman_deltas.csv", index=False)
    alignment.to_csv(OUT / "source_system_partition_alignment.csv", index=False)
    _figure(macro, FIG, False)
    _figure(macro, FIG_ZH, True)
    DOC.write_text(_report(stages, macro, scales, deltas, selections, deployment, False), encoding="utf-8")
    DOC_ZH.write_text(_report(stages, macro, scales, deltas, selections, deployment, True), encoding="utf-8")
    verification = {
        "schema_version": "integrated_auto_system_verification_v1",
        "outer_partitions": int(len(selections)),
        "outer_group_overlap_sum": int(selections.outer_group_overlap.sum()),
        "outer_test_used_for_selection_any": bool(selections.selection_uses_outer_test.any()),
        "inner_group_overlap_sum": int(inner.group_overlap.sum()),
        "source_stage_auto_test_partitions_aligned": int(alignment.all_three_equal.sum()),
        "deployment_endpoints": int(len(deployment)),
        "maximum_treeshap_residual": float(deployment.max_treeshap_reconciliation_residual.max()),
        "outer_models": 90,
        "heldout_explanation_cases": 9,
        "maximum_all_heldout_treeshap_residual": float(outer_reconciliation.max_additivity_residual.max()),
        "maximum_saved_prediction_residual": float(outer_reconciliation.max_saved_prediction_residual.max()),
        "maximum_exact_helm_overlap": int(split_overlap.exact_helm_overlap.max()),
        "maximum_exact_canonical_sequence_overlap": int(split_overlap.exact_sequence_overlap.max()),
        "canonical_sequence_overlap_limitation": "nonzero in MOE and Atlas despite zero supplied-group and exact-HELM overlap; disclosed rather than treated as a passed safeguard",
        "source_macro_spearman": float(macro.loc["source_native", "mean_spearman"]),
        "automatic_macro_spearman": float(macro.loc["automatic_system", "mean_spearman"]),
        "total_spearman_delta": float(macro.loc["automatic_system", "mean_spearman"] - macro.loc["source_native", "mean_spearman"]),
        "input_hashes": {str(path.relative_to(ROOT)): _sha256(path) for path in required},
    }
    if verification["outer_partitions"] != 45 or verification["outer_group_overlap_sum"] != 0 or verification["inner_group_overlap_sum"] != 0 or verification["source_stage_auto_test_partitions_aligned"] != 45 or verification["outer_test_used_for_selection_any"] or verification["deployment_endpoints"] != 9 or verification["maximum_treeshap_residual"] > 1e-3 or verification["maximum_all_heldout_treeshap_residual"] > 1e-3 or verification["maximum_saved_prediction_residual"] > 1e-8 or verification["maximum_exact_helm_overlap"] != 0:
        raise RuntimeError(f"verification failed: {verification}")
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    include = [DOC, DOC_ZH, FIG, FIG_ZH, *sorted(OUT.glob("*")), RUN / "run_manifest.json", RUN / "selections.csv", RUN / "summary.csv", RUN / "deployment_summary.csv", ROOT / "outputs/integrated_auto_system_v1/formal_v2_guard/summary_success.json", ROOT / "outputs/integrated_auto_system_v1/formal_v2_guard/telemetry_success.csv", ROOT / "outputs/integrated_auto_system_v1/formal_v2_guard/summary_finalize.json", ROOT / "outputs/integrated_auto_system_v1/stages_v2_guard/summary.json", ROOT / "outputs/integrated_auto_system_v1/stages_v2_guard/telemetry.csv", ROOT / "scripts/benchmark_xgb_robust_nested.py", Path(__file__), ROOT / "tests/test_xgb_robust_nested.py"]
    include.extend(sorted((RUN / "deployment_models").glob("*")))
    include.extend(sorted(path for path in OUTER_EXPLANATIONS.rglob("*") if path.is_file()))
    paper_dir = ROOT / "docs/papers/oligo_toxicity_journal_v1"
    include.extend(
        paper_dir / name
        for name in (
            "PAPER.md", "PAPER.pdf", "PAPER_zh.md", "PAPER_zh.pdf",
            "SUPPLEMENT.md", "SUPPLEMENT.pdf", "SUPPLEMENT_zh.md",
            "SUPPLEMENT_zh.pdf", "VERIFICATION.json",
        )
        if (paper_dir / name).exists()
    )
    include.extend(sorted(paper_dir.glob("INTEGRATED_AUTO_*REVIEW.md")))
    include.extend(sorted(paper_dir.glob("INTEGRATED_AUTO_*RESPONSE.md")))
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in include:
            archive.write(path, Path("integrated_auto_system") / path.relative_to(ROOT))
    print(json.dumps({"verification": verification, "report": str(DOC), "report_zh": str(DOC_ZH), "bundle": str(BUNDLE)}, indent=2))


if __name__ == "__main__":
    main()
