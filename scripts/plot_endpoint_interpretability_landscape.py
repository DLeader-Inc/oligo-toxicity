#!/usr/bin/env python3
"""Render bilingual endpoint-resolved measured/model interpretability figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ORDER = [
    "cytotoxicity", "neurotoxicity_lna", "neurotoxicity_moe", "tlr7_immunotox",
    "tlr8_immunotox", "mouse_hepatic", "rat_hepatic", "mouse_neuro", "rat_neuro",
]
LABEL_EN = {
    "cytotoxicity": "HeLa cytotoxicity", "neurotoxicity_lna": "LNA neuronal calcium",
    "neurotoxicity_moe": "MOE acute FOB", "tlr7_immunotox": "TLR8 potentiation",
    "tlr8_immunotox": "TLR7 inhibition", "mouse_hepatic": "Mouse ALT",
    "rat_hepatic": "Rat ALT", "mouse_neuro": "Mouse FOB", "rat_neuro": "Rat FOB",
}
LABEL_ZH = {
    "cytotoxicity": "HeLa 细胞毒性", "neurotoxicity_lna": "LNA 神经元钙评分",
    "neurotoxicity_moe": "MOE 急性 FOB", "tlr7_immunotox": "TLR8 增强",
    "tlr8_immunotox": "TLR7 抑制", "mouse_hepatic": "小鼠 ALT",
    "rat_hepatic": "大鼠 ALT", "mouse_neuro": "小鼠 FOB", "rat_neuro": "大鼠 FOB",
}
CASE_ORDER = ["cytotoxicity", "neurotoxicity_lna", "neurotoxicity_moe", "mouse_neuro"]


def style() -> None:
    cjk_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(cjk_path)
    cjk_family = font_manager.FontProperties(fname=cjk_path).get_name()
    plt.rcParams.update({
        "font.family": [cjk_family, "DejaVu Sans"], "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 220,
    })


def endpoint_landscape(data_dir: Path, out: Path, zh: bool) -> None:
    summary = pd.read_csv(data_dir / "dataset_direct_chemistry_summary.csv").set_index("dataset").loc[ORDER]
    deletion = pd.read_csv(data_dir / "model_family_deletion_by_endpoint.csv")
    deletion = deletion[deletion.ablation_family.eq("chemical_modification")].set_index("dataset").loc[ORDER]
    recovery = pd.read_csv(data_dir / "source_protocol_oof_pair_recovery_by_endpoint.csv").set_index("dataset")
    labels = [LABEL_ZH[x] if zh else LABEL_EN[x] for x in ORDER]
    y = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.3), gridspec_kw={"wspace": .58})

    axes[0].barh(y, summary.eligible_row_fraction * 100, color="#3b82f6")
    axes[0].set_yticks(y, labels); axes[0].invert_yaxis(); axes[0].set_xlim(0, 105)
    axes[0].set_xlabel("具有同序列化学对照的行（%）\nAtlas 另匹配已有条件" if zh else "Rows with same-sequence chemical comparator (%)\nAtlas additionally matches recorded context")
    for i, value in enumerate(summary.eligible_row_fraction * 100):
        axes[0].text(min(value + 1.2, 96), i, f"{value:.1f}", va="center", fontsize=8)

    loss = deletion.spearman_loss_when_removed
    colors = ["#ef4444" if value > 0 else "#94a3b8" for value in loss]
    axes[1].barh(y, loss, color=colors); axes[1].axvline(0, color="black", lw=.7)
    axes[1].set_yticks(y, labels); axes[1].invert_yaxis()
    axes[1].set_xlabel("删除显式化学信息后的 Spearman 损失" if zh else "Spearman loss after deleting explicit\nchemical-information block")

    agreement = [recovery.loc[x, "group_level_direction_agreement"] * 100 if x in recovery.index else np.nan for x in ORDER]
    axes[2].barh(y, np.nan_to_num(agreement), color=["#10b981" if np.isfinite(v) else "#e2e8f0" for v in agreement])
    axes[2].set_yticks(y, labels); axes[2].invert_yaxis(); axes[2].set_xlim(0, 105)
    axes[2].set_xlabel("同一 outer model 对实测差值\n方向的恢复率（%）" if zh else "Direction recovery for measured contrasts\nby the same outer model (%)")
    for i, value in enumerate(agreement):
        text = "无对照" if zh and not np.isfinite(value) else ("no contrast" if not np.isfinite(value) else f"{value:.1f}")
        axes[2].text(1 if not np.isfinite(value) else min(value + 1.2, 96), i, text, va="center", fontsize=8)

    title = "九个终点中，化学变化的可辨识性与模型依赖并不相同" if zh else "Chemical estimability and model dependence differ across endpoints"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=.99)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def short_axis(axis: str, zh: bool) -> str:
    text = axis.replace("sugar@", "位点 ").replace("linkage@", "连接位点 ") if zh else axis.replace("sugar@", "sugar ").replace("linkage@", "linkage ")
    return text


def matched_edit_cases(data_dir: Path, out: Path, zh: bool) -> None:
    frame = pd.read_csv(data_dir / "single_edit_context_dependence.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.3), gridspec_kw={"hspace": .54, "wspace": .43})
    for ax, dataset in zip(axes.flat, CASE_ORDER):
        block = frame[frame.dataset.eq(dataset)].sort_values("sequence_backgrounds", ascending=False).head(6).iloc[::-1]
        ypos = np.arange(len(block)); labels = [short_axis(x, zh) for x in block.single_edit_axis]
        obs_low = block.observed_median_delta - block.observed_q25
        obs_high = block.observed_q75 - block.observed_median_delta
        pred_low = block.predicted_median_delta - block.predicted_min_delta
        pred_high = block.predicted_max_delta - block.predicted_median_delta
        ax.errorbar(block.observed_median_delta, ypos + .11, xerr=np.vstack([obs_low, obs_high]), fmt="o", color="#2563eb", capsize=3, label="实测中位数与四分位距" if zh else "Measured median and IQR")
        ax.errorbar(block.predicted_median_delta, ypos - .11, xerr=np.vstack([pred_low, pred_high]), fmt="s", color="#f97316", capsize=3, label="模型中位数与跨背景范围" if zh else "Model median and context range")
        ax.axvline(0, color="black", lw=.7); ax.set_yticks(ypos, labels)
        ax.set_title(LABEL_ZH[dataset] if zh else LABEL_EN[dataset], fontweight="bold")
        ax.set_xlabel("高等级化学形式 − 低等级化学形式的响应差" if zh else "Response difference: higher-coded chemistry minus lower-coded chemistry")
        for i, n in enumerate(block.sequence_backgrounds):
            ax.text(.99, (i+.05)/max(len(block),1), f"n={int(n)}", transform=ax.transAxes, ha="right", fontsize=8, color="#475569")
    axes[0, 0].legend(loc="best", fontsize=8)
    title = "同一化学编辑在不同 sequence 背景中的实测差异与 outer-model 响应" if zh else "Measured contrasts and outer-model responses for the same edit across sequence contexts"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=.99)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def joint_descriptor_atlas(data_dir: Path, out: Path, zh: bool) -> None:
    frame = pd.read_csv(data_dir / "model_top_joint_descriptors.csv")
    fig, axes = plt.subplots(3, 3, figsize=(15.2, 11.2), gridspec_kw={"hspace": .62, "wspace": .68})
    for ax, dataset in zip(axes.flat, ORDER):
        block = frame[frame.dataset.eq(dataset)].sort_values("mean_absolute_contribution").tail(5)
        labels = block.plain_language_zh if zh else block.feature.str.replace("pint_", "", regex=False).str.replace("scint_", "", regex=False)
        ax.barh(np.arange(len(block)), block.normalized_absolute_mass * 100, color="#7c3aed")
        ax.set_yticks(np.arange(len(block)), labels, fontsize=7)
        ax.set_title(LABEL_ZH[dataset] if zh else LABEL_EN[dataset], fontsize=10, fontweight="bold")
        ax.set_xlabel("绝对 TreeSHAP 总量（%）" if zh else "Absolute TreeSHAP mass (%)", fontsize=8)
    title = "各终点最显著的模型内 sequence×chemistry/位置组合" if zh else "Leading within-model sequence-by-chemistry/position combinations by endpoint"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=.995)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True); style()
    endpoint_landscape(args.data_dir, args.out_dir / "endpoint_interpretability_landscape.png", False)
    endpoint_landscape(args.data_dir, args.out_dir / "endpoint_interpretability_landscape_zh.png", True)
    matched_edit_cases(args.data_dir, args.out_dir / "matched_edit_context_cases.png", False)
    matched_edit_cases(args.data_dir, args.out_dir / "matched_edit_context_cases_zh.png", True)
    joint_descriptor_atlas(args.data_dir, args.out_dir / "model_joint_descriptors_by_endpoint.png", False)
    joint_descriptor_atlas(args.data_dir, args.out_dir / "model_joint_descriptors_by_endpoint_zh.png", True)


if __name__ == "__main__":
    main()
