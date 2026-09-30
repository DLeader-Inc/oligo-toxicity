#!/usr/bin/env python3
"""Endpoint-resolved measured chemistry contrasts and held-out joint-descriptor evidence."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_aso_atlas2_project_representation import load_task  # noqa: E402
from scripts.benchmark_registry_nine_endpoints import ATLAS, EARLIER, earlier_frame  # noqa: E402
from src.features.featurizer import parse_helm_components  # noqa: E402

RUN = ROOT / "outputs/expert_rule_scale_v1/expert_rule_scale"
ENDPOINT_ORDER = EARLIER + ATLAS
DISPLAY = {
    "cytotoxicity": "HeLa caspase-3/7 cytotoxicity",
    "neurotoxicity_lna": "LNA neuronal calcium score",
    "neurotoxicity_moe": "MOE acute FOB neurotoxicity",
    "tlr7_immunotox": "TLR8 potentiation",
    "tlr8_immunotox": "TLR7 inhibition",
    "mouse_hepatic": "Mouse ALT",
    "rat_hepatic": "Rat ALT",
    "mouse_neuro": "Mouse FOB",
    "rat_neuro": "Rat FOB",
}
DIRECTION = {
    "cytotoxicity": "higher = stronger cytotoxic response",
    "neurotoxicity_lna": "lower calcium score = greater acute neurotoxicity",
    "neurotoxicity_moe": "higher = greater behavioral toxicity",
    "tlr7_immunotox": "higher = stronger TLR8 potentiation",
    "tlr8_immunotox": "lower residual reporter = stronger TLR7 inhibition",
    "mouse_hepatic": "higher = higher measured ALT",
    "rat_hepatic": "higher = higher measured ALT",
    "mouse_neuro": "higher = stronger FOB phenotype",
    "rat_neuro": "higher = stronger FOB phenotype",
}
CONTEXT = {
    "mouse_hepatic": ["dosage_mg_per_kg", "num_doses", "dosing_period_days", "adminstration_method", "species_strain"],
    "rat_hepatic": ["dosage_mg_per_kg", "num_doses", "dosing_period_days", "adminstration_method", "species_strain"],
    "mouse_neuro": ["dosage_ug", "latency_time_hours", "administration_method", "species_strain"],
    "rat_neuro": ["dosage_ug", "latency_time_hours", "administration_method", "species_strain"],
}
SUGAR_LABEL = {"D": "DNA", "R": "RNA", "O": "2′-O-methyl", "M": "MOE", "L": "LNA", "E": "cEt", "X": "other"}


def molecule_fields(helm: str) -> dict[str, object]:
    residues = parse_helm_components(str(helm))
    sequence = "".join(item.base for item in residues)
    sugar = "".join(item.modification for item in residues)
    linkage = tuple(item.linkage for item in residues[:-1])
    modified_base = tuple(item.raw_base.upper() not in {"A", "C", "G", "T", "U"} for item in residues)
    base_pattern = "".join("m" if flag else "-" for flag in modified_base)
    signature = f"{sugar}|{'-'.join(linkage)}|{base_pattern}"
    counts = {f"n_{code}": sugar.count(code) for code in SUGAR_LABEL}
    counts.update({"n_PS": linkage.count("PS"), "n_PO": linkage.count("PO"), "n_modified_base": sum(modified_base)})
    return {
        "sequence": sequence, "sugar_pattern": sugar, "linkage_pattern": "-".join(linkage),
        "modified_base_pattern": base_pattern, "chemistry_signature": signature, **counts,
    }


def frame_for(dataset: str) -> pd.DataFrame:
    frame = earlier_frame(dataset) if dataset in EARLIER else load_task(dataset).reset_index(drop=True)
    parsed = pd.DataFrame([molecule_fields(value) for value in frame.x.astype(str)])
    result = pd.concat([frame.reset_index(drop=True), parsed], axis=1)
    result["source_row_index"] = np.arange(len(result))
    keys = [key for key in CONTEXT.get(dataset, []) if key in result.columns]
    for key in keys:
        result[key] = result[key].fillna("MISSING").astype(str)
    result["context_key"] = result[keys].astype(str).agg("|".join, axis=1) if keys else "within_endpoint_context_unverified"
    result["match_key"] = result.sequence + "||" + result.context_key
    return result


def edit_details(left: pd.Series, right: pd.Series) -> tuple[list[str], int, int, int]:
    edits = []
    sugar_edits = 0
    for position, (a, b) in enumerate(zip(left.sugar_pattern, right.sugar_pattern, strict=False), start=1):
        if a != b:
            sugar_edits += 1
            edits.append(f"sugar@{position}:{SUGAR_LABEL.get(a,a)}→{SUGAR_LABEL.get(b,b)}")
    left_link, right_link = str(left.linkage_pattern).split("-"), str(right.linkage_pattern).split("-")
    linkage_edits = 0
    for position, (a, b) in enumerate(zip(left_link, right_link, strict=False), start=1):
        if a != b:
            linkage_edits += 1
            edits.append(f"linkage@{position}:{a}→{b}")
    base_edits = sum(a != b for a, b in zip(left.modified_base_pattern, right.modified_base_pattern, strict=False))
    if base_edits:
        edits.append(f"modified_base_positions:{base_edits}")
    length_penalty = abs(len(left.sugar_pattern) - len(right.sugar_pattern)) + abs(len(left_link) - len(right_link))
    return edits, sugar_edits, linkage_edits, base_edits + length_penalty


def measured_pairs(dataset: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    # Median-collapse exact replicate rows before forming chemistry contrasts.
    group_cols = ["match_key", "sequence", "context_key", "chemistry_signature", "sugar_pattern", "linkage_pattern", "modified_base_pattern"]
    collapsed = frame.groupby(group_cols, as_index=False).agg(
        y=("y", "median"), source_row_index=("source_row_index", "min"), replicate_rows=("y", "size")
    )
    pairs = []
    for match_key, block in collapsed.groupby("match_key", sort=False):
        if block.chemistry_signature.nunique() < 2:
            continue
        for (_, left), (_, right) in combinations(block.sort_values("chemistry_signature").iterrows(), 2):
            edits, sugar_edits, linkage_edits, other_edits = edit_details(left, right)
            total_edits = sugar_edits + linkage_edits + other_edits
            pairs.append({
                "dataset": dataset, "endpoint": DISPLAY[dataset], "response_direction": DIRECTION[dataset],
                "match_key": match_key, "sequence": left.sequence, "context_key": left.context_key,
                "left_row": int(left.source_row_index), "right_row": int(right.source_row_index),
                "left_sugar": left.sugar_pattern, "right_sugar": right.sugar_pattern,
                "left_linkage": left.linkage_pattern, "right_linkage": right.linkage_pattern,
                "left_y": float(left.y), "right_y": float(right.y), "delta_y_left_minus_right": float(left.y-right.y),
                "absolute_delta_y": float(abs(left.y-right.y)), "edit_count": total_edits,
                "sugar_edit_count": sugar_edits, "linkage_edit_count": linkage_edits,
                "modified_base_or_length_edit_count": other_edits, "edits": "; ".join(edits),
            })
    pairs = pd.DataFrame(pairs)
    if not pairs.empty:
        pairs = normalize_single_edit_axes(pairs)
    multi_keys = set(pairs.match_key) if not pairs.empty else set()
    eligible_rows = int(frame.match_key.isin(multi_keys).sum())
    summary = {
        "dataset": dataset, "endpoint": DISPLAY[dataset], "rows": len(frame),
        "unique_sequences": int(frame.sequence.nunique()),
        "chemistry_patterns": int(frame.chemistry_signature.nunique()),
        "context_matched_sequence_groups_with_chemistry_variation": len(multi_keys),
        "eligible_rows": eligible_rows, "eligible_row_fraction": eligible_rows / len(frame),
        "matched_chemistry_pairs": len(pairs),
        "single_edit_pairs": int((pairs.edit_count == 1).sum()) if not pairs.empty else 0,
        "pair_weighted_median_absolute_outcome_difference": float(pairs.absolute_delta_y.median()) if not pairs.empty else np.nan,
        "group_weighted_median_absolute_outcome_difference": (
            float(pairs.groupby("match_key").absolute_delta_y.median().median()) if not pairs.empty else np.nan
        ),
        "group_weighted_p90_absolute_outcome_difference": (
            float(pairs.groupby("match_key").absolute_delta_y.median().quantile(.9)) if not pairs.empty else np.nan
        ),
    }
    return pairs, summary


def normalize_single_edit_axes(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    axes, normalized, multipliers = [], [], []
    for row in pairs.itertuples():
        if row.edit_count != 1 or ";" in row.edits or "→" not in row.edits:
            axes.append(""); normalized.append(np.nan); multipliers.append(np.nan); continue
        prefix, change = row.edits.split(":", 1)
        left, right = change.split("→", 1)
        if prefix.startswith("sugar@"):
            priority = {"DNA": 0, "RNA": 1, "2′-O-methyl": 2, "MOE": 3, "LNA": 4, "cEt": 5, "other": 6}
        else:
            priority = {"PO": 0, "PS": 1, "OTHER": 2}
        low, high = sorted((left, right), key=lambda value: (priority.get(value, 99), value))
        # Stored delta is y(left)-y(right); normalize to y(high)-y(low).
        multiplier = -1.0 if (left, right) == (low, high) else 1.0
        axes.append(f"{prefix}:{high}−{low}")
        normalized.append(multiplier * float(row.delta_y_left_minus_right))
        multipliers.append(multiplier)
    pairs["single_edit_axis"] = axes
    pairs["normalized_delta_y_high_minus_low"] = normalized
    pairs["single_edit_orientation_multiplier"] = multipliers
    return pairs


def select_pair_examples(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    chosen = []
    specifications = [
        ("minimal_single_edit", pairs[pairs.edit_count.eq(1)], .50),
        ("typical_context_matched_contrast", pairs, .50),
        ("upper_decile_observed_contrast", pairs, .90),
    ]
    used: set[tuple[int, int]] = set()
    for role, candidates, quantile in specifications:
        candidates = candidates.copy()
        candidates["pair_id"] = list(zip(candidates.left_row, candidates.right_row))
        candidates = candidates[~candidates.pair_id.isin(used)]
        if candidates.empty:
            continue
        target = candidates.absolute_delta_y.quantile(quantile)
        candidates["distance"] = (candidates.absolute_delta_y-target).abs()
        row = candidates.sort_values(["distance", "edit_count", "left_row", "right_row"]).iloc[0].copy()
        row["selection_role"] = role
        used.add(row.pair_id)
        chosen.append(row.drop(labels=["pair_id", "distance"]))
    return pd.DataFrame(chosen)


def linear_trends(pairs: pd.DataFrame, frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    lookup = frame.set_index("source_row_index")
    features = [f"n_{code}" for code in ("L", "M", "O", "D", "E")] + ["n_PS", "n_PO", "n_modified_base"]
    rows = []
    for feature in features:
        slopes, isolated, group_values, isolated_group_values = [], [], {}, {}
        for pair in pairs.itertuples():
            left, right = lookup.loc[pair.left_row], lookup.loc[pair.right_row]
            delta_x = float(left[feature]-right[feature])
            if delta_x == 0:
                continue
            value = float(pair.delta_y_left_minus_right/delta_x)
            slopes.append(value)
            group_values.setdefault(pair.sequence, []).append(value)
            if pair.edit_count == abs(delta_x):
                isolated.append(value)
                isolated_group_values.setdefault(pair.sequence, []).append(value)
        if slopes:
            group_medians = np.asarray([np.median(values) for values in group_values.values()], dtype=float)
            isolated_group_medians = np.asarray([np.median(values) for values in isolated_group_values.values()], dtype=float)
            rng = np.random.default_rng(20260907 + len(rows))
            if len(group_medians) > 1:
                boots = np.asarray([np.median(rng.choice(group_medians, len(group_medians), replace=True)) for _ in range(2000)])
                ci_low, ci_high = np.quantile(boots, [.025, .975])
            else:
                ci_low = ci_high = group_medians[0]
            rows.append({
                "dataset": dataset, "endpoint": DISPLAY[dataset], "feature": feature,
                "pair_slopes": len(slopes), "median_delta_y_per_plus_one": float(np.median(slopes)),
                "q25": float(np.quantile(slopes,.25)), "q75": float(np.quantile(slopes,.75)),
                "independent_sequences": len(group_medians),
                "group_weighted_median_delta_y_per_plus_one": float(np.median(group_medians)),
                "group_bootstrap_ci95_low": float(ci_low), "group_bootstrap_ci95_high": float(ci_high),
                "isolated_pair_slopes": len(isolated),
                "isolated_median_delta_y_per_plus_one": float(np.median(isolated)) if isolated else np.nan,
                "isolated_independent_sequences": len(isolated_group_medians),
                "isolated_group_weighted_median": float(np.median(isolated_group_medians)) if len(isolated_group_medians) else np.nan,
                "claim_boundary": "descriptive within-sequence/context association; co-changing chemistry may confound non-isolated slopes",
            })
    return pd.DataFrame(rows)


def add_source_protocol_oof_recovery(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare measured pair deltas with predictions from a shared outer partition."""
    if pairs.empty:
        return pairs, pd.DataFrame(), pd.DataFrame()
    oof = pd.read_csv(RUN / "oof_predictions.csv.gz")[["dataset", "outer_partition", "row_index", "pred_policy"]]
    left = oof.rename(columns={"row_index": "left_row", "pred_policy": "left_prediction"})
    right = oof.rename(columns={"row_index": "right_row", "pred_policy": "right_prediction"})
    common = pairs.merge(left, on=["dataset", "left_row"], how="left").merge(
        right, on=["dataset", "right_row", "outer_partition"], how="inner"
    )
    common["predicted_delta_left_minus_right"] = common.left_prediction-common.right_prediction
    common["predicted_normalized_delta_high_minus_low"] = (
        common.predicted_delta_left_minus_right * common.single_edit_orientation_multiplier
    )
    pair_keys = ["dataset", "left_row", "right_row"]
    predicted = common.groupby(pair_keys, as_index=False).agg(
        common_outer_partitions=("outer_partition", "nunique"),
        predicted_delta_left_minus_right=("predicted_delta_left_minus_right", "mean"),
        predicted_normalized_delta_high_minus_low=("predicted_normalized_delta_high_minus_low", "mean"),
    )
    enriched = pairs.merge(predicted, on=pair_keys, how="left")
    summaries = []
    for dataset, block in enriched.dropna(subset=["predicted_delta_left_minus_right"]).groupby("dataset"):
        group = block.groupby("match_key", as_index=False).agg(
            observed=("delta_y_left_minus_right", "median"), predicted=("predicted_delta_left_minus_right", "median")
        )
        nonzero = group[group.observed.ne(0) & group.predicted.ne(0)]
        summaries.append({
            "dataset": dataset, "endpoint": DISPLAY[dataset], "matched_pairs_with_shared_outer_model": len(block),
            "matched_sequence_context_groups": len(group),
            "group_level_delta_spearman": float(spearmanr(group.observed, group.predicted).statistic) if len(group)>2 else np.nan,
            "group_level_direction_agreement": float((np.sign(nonzero.observed)==np.sign(nonzero.predicted)).mean()) if len(nonzero) else np.nan,
            "direction_evaluable_groups": len(nonzero),
            "direction_agreeing_groups": int((np.sign(nonzero.observed)==np.sign(nonzero.predicted)).sum()),
            "group_level_delta_mae": float(np.mean(np.abs(group.observed-group.predicted))),
            "validation_scope": "same source-protocol outer partition; not an exact-sequence-grouped experiment",
        })
    edit_pair_rows = enriched[
        enriched.single_edit_axis.ne("") & enriched.predicted_normalized_delta_high_minus_low.notna()
    ]
    # One edit axis can occur several times inside the same sequence/context group
    # against different unchanged chemistry backgrounds. Collapse those first so a
    # densely enumerated group does not receive more statistical weight.
    edit_groups = edit_pair_rows.groupby(["dataset", "single_edit_axis", "sequence"], as_index=False).agg(
        observed_group_delta=("normalized_delta_y_high_minus_low", "median"),
        predicted_group_delta=("predicted_normalized_delta_high_minus_low", "median"),
    )
    edit_context = edit_groups.groupby(["dataset", "single_edit_axis"], as_index=False).agg(
        sequence_backgrounds=("sequence", "nunique"),
        observed_median_delta=("observed_group_delta", "median"),
        observed_q25=("observed_group_delta", lambda values: float(np.quantile(values,.25))),
        observed_q75=("observed_group_delta", lambda values: float(np.quantile(values,.75))),
        predicted_median_delta=("predicted_group_delta", "median"),
        predicted_min_delta=("predicted_group_delta", "min"),
        predicted_max_delta=("predicted_group_delta", "max"),
        predicted_positive_fraction=("predicted_group_delta", lambda values: float((values>0).mean())),
    )
    edit_context = edit_context[edit_context.sequence_backgrounds.ge(3)].sort_values(
        ["dataset", "sequence_backgrounds"], ascending=[True, False]
    )
    if not edit_context.empty:
        edit_context.insert(1, "endpoint", edit_context.dataset.map(DISPLAY))
        edit_context["interpretation_boundary"] = (
            "model response across observed sequence/assay contexts; variation or sign change is context dependence, not a causal interaction"
        )
    return enriched, pd.DataFrame(summaries), edit_context


def interaction_kind(feature: str) -> str | None:
    if feature.startswith("pint_motif_") or feature.startswith("scint_motif_"):
        return "motif×chemistry_or_position"
    if feature.startswith("pint_base_") and "__mod_" in feature:
        return "base×sugar×position"
    if feature.startswith("pint_base_") and "__link_" in feature:
        return "base×backbone×position"
    if feature.startswith("scint_") or feature.startswith("pint_mod_transition_") or feature.startswith("pint_link_transition_"):
        return "chemistry×regional_context"
    if feature.startswith("ucontext__") and any(token in feature for token in ("_sugar_", "_link_")):
        return "chemistry×landmark_position"
    return None


def humanize_feature(feature: str) -> str:
    text = feature
    replacements = {
        "pint_base_": "碱基 ", "pint_motif_": "短序列 ", "__mod_": " × 糖 ",
        "__link_": " × 骨架 ", "__region_": " × 区域 ", "region_": "区域 ",
        "wing5": "5′翼区", "wing3": "3′翼区", "gap_mid": "缺口中部", "gap": "DNA缺口",
        "ucontext__anchor_": "标志位置 ", "ucontext__pool_": "区域汇总 ",
        "_sugar_": " 的糖 ", "_link_": " 的骨架 ", "pint_mod_transition_": "相邻糖转换 ",
        "pint_link_transition_": "相邻骨架转换 ", "scint_": "序列–化学交互 ", "_": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def model_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(RUN / "explanations/global_heldout_feature_ranking.csv")
    features["interaction_kind"] = features.feature.map(interaction_kind)
    interactions = features[features.interaction_kind.notna() & features.mean_absolute_contribution.gt(0)].copy()
    interactions["plain_language_zh"] = interactions.feature.map(humanize_feature)
    interactions = interactions.sort_values(["dataset", "mean_absolute_contribution"], ascending=[True, False])
    interactions["interaction_rank"] = interactions.groupby("dataset").cumcount()+1
    interactions = interactions[interactions.interaction_rank.le(5)]

    full = pd.read_csv(RUN / "summary.csv")[["dataset", "spearman"]].rename(columns={"spearman": "full_spearman"})
    deletion = pd.read_csv(RUN / "family_ablation/family_ablation_endpoint_metrics.csv").merge(full, on="dataset")
    deletion["spearman_loss_when_removed"] = deletion.full_spearman-deletion.spearman
    deletion["evidence_type"] = "outer-held-out refitted family deletion; functional dependence, not causality"
    return interactions, deletion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/endpoint_interpretability_landscape_v1")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_frames, summaries, trends = [], [], []
    for dataset in ENDPOINT_ORDER:
        frame = frame_for(dataset)
        pairs, summary = measured_pairs(dataset, frame)
        summaries.append(summary)
        if not pairs.empty:
            pair_frames.append(pairs)
            trend = linear_trends(pairs, frame, dataset)
            if not trend.empty:
                trends.append(trend)
    summary_frame = pd.DataFrame(summaries)
    pair_frame = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    trend_frame = pd.concat(trends, ignore_index=True) if trends else pd.DataFrame()
    pair_frame, recovery_frame, edit_context_frame = add_source_protocol_oof_recovery(pair_frame)
    examples = [select_pair_examples(block) for _, block in pair_frame.groupby("dataset", sort=False)]
    example_frame = pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()
    if not recovery_frame.empty:
        summary_frame = summary_frame.merge(
            recovery_frame[["dataset", "matched_pairs_with_shared_outer_model", "matched_sequence_context_groups"]].rename(
                columns={"matched_sequence_context_groups": "oof_recovery_sequence_context_groups"}
            ),
            on="dataset", how="left",
        )
    for column in ("matched_pairs_with_shared_outer_model", "oof_recovery_sequence_context_groups"):
        if column not in summary_frame:
            summary_frame[column] = 0
        summary_frame[column] = summary_frame[column].fillna(0).astype(int)
    interactions, deletion = model_evidence()
    summary_frame.to_csv(args.out_dir / "dataset_direct_chemistry_summary.csv", index=False)
    pair_frame.to_csv(args.out_dir / "dataset_context_matched_chemistry_pairs.csv.gz", index=False, compression="gzip")
    example_frame.to_csv(args.out_dir / "dataset_pair_examples.csv", index=False)
    trend_frame.to_csv(args.out_dir / "dataset_within_sequence_linear_trends.csv", index=False)
    recovery_frame.to_csv(args.out_dir / "source_protocol_oof_pair_recovery_by_endpoint.csv", index=False)
    edit_context_frame.to_csv(args.out_dir / "single_edit_context_dependence.csv", index=False)
    interactions.to_csv(args.out_dir / "model_top_joint_descriptors.csv", index=False)
    deletion.to_csv(args.out_dir / "model_family_deletion_by_endpoint.csv", index=False)
    manifest = {
        "schema_version": "endpoint_interpretability_landscape_v1",
        "endpoints": len(summary_frame), "source_rows": int(summary_frame.rows.sum()),
        "matched_pairs": int(summary_frame.matched_chemistry_pairs.sum()),
        "same_sequence_required": True,
        "atlas_recorded_context_matched": True,
        "oligogym_row_context_available": False,
        "direct_trend_scope": "measured descriptive association; exact replicates median-collapsed and repeated-edit summaries weighted once per base sequence",
        "oof_pair_scope": "both records predicted by the same source-protocol outer model; not a new exact-sequence-grouped validation",
        "model_scope": "source-protocol OOF pair recovery, outer-held-out TreeSHAP aggregate and refitted family deletion",
        "causal_claim": False,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(summary_frame.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
