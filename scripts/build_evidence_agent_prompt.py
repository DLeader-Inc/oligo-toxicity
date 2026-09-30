#!/usr/bin/env python3
"""Build a provenance-constrained prompt for an independent evidence sub-agent.

The prompt deliberately separates model-derived observations from biological
claims.  It is a hand-off artifact: a caller can run the retrieval script first
and give this prompt plus the reviewed/candidate evidence ledger to another
agent without allowing it to silently promote a keyword hit into a mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_prompt(packet: dict, evidence_path: Path | list[Path] | None = None) -> str:
    prediction = packet.get("prediction", {})
    entry = packet.get("entry", {})
    mechanisms = packet.get("mechanisms", [])
    attributions = packet.get("attributions", [])
    evidence = packet.get("evidence", [])
    neighbors = packet.get("neighbors", [])
    secondary_models = packet.get("secondary_models", [])
    packet_json = json.dumps(packet, ensure_ascii=False, indent=2)
    evidence_json = ""
    evidence_paths = (
        evidence_path
        if isinstance(evidence_path, list)
        else [evidence_path] if evidence_path is not None else []
    )
    evidence_sections = []
    for path in evidence_paths:
        if path.exists():
            evidence_sections.append(
                f"### {path}\n{path.read_text(encoding='utf-8')}"
            )
    evidence_json = "\n\n".join(evidence_sections)

    lines = [
        "# Independent oligonucleotide toxicity evidence review",
        "",
        "You are an independent biomedical evidence reviewer. Do not optimize for",
        "agreement with the model. Treat every model output as an observation, not",
        "as causal proof. Return both supporting and contradictory evidence.",
        "",
        "## Required claim separation",
        "",
        "1. Prediction claim: what endpoint was predicted, by which artifact, descriptor",
        "   family, training mode, rows, groups, validation protocol and response direction.",
        "2. Model-explanation claim: which same-model features, contributions, attention",
        "   or gradients were observed. Call attention/gradient from another model",
        "   secondary_model evidence; never call it the XGBoost explanation.",
        "3. Evidence claim: what a reviewed paper, patent or database record directly",
        "   supports, contradicts or provides only as context. Include URL/identifier",
        "   and a table/figure/page/section locator when available.",
        "4. Mechanism claim: label it established_context, supported_candidate,",
        "   novel_candidate or unresolved. Give alternatives and a falsification test.",
        "",
        "## Hard constraints",
        "",
        "- Do not convert feature importance, similarity, attention, a rule score or a",
        "  keyword hit into causality.",
        "- Do not promote a patent discovery into a training label.",
        "- A similar measured training row is analogue/applicability evidence only;",
        "  never present it as the reason the model predicted the value or as proof",
        "  that the query and neighbor share a mechanism.",
        "- Do not write a citation without a stable URL/identifier.",
        "- If the exact entry was in training, say the result is retrospective.",
        "- Report grouped-validation Spearman, R-squared and RMSE/y_std together",
        "  for every primary or secondary model. Also retain raw RMSE and y_std;",
        "  never infer numeric predictive utility from Spearman alone.",
        "- State missing assay details, species, dose, route, chemistry and endpoint.",
        "- Use confidence as a heuristic evidence/reliability assessment, never as a",
        "  calibrated probability unless a calibration artifact is explicitly present.",
        "",
        "## Entry and prediction context",
        "",
        f"- sequence: `{entry.get('sequence', '')}`",
        f"- target: `{entry.get('target', '')}`",
        f"- endpoint: `{prediction.get('endpoint', '')}`",
        f"- prediction: `{prediction.get('value', '')}`",
        f"- model family: `{prediction.get('model_family', '')}`",
        f"- descriptor family: `{prediction.get('descriptor_family', '')}`",
        f"- representation: `{prediction.get('representation', '')}`",
        f"- training mode: `{prediction.get('training_mode', '')}`",
        f"- validation: `{prediction.get('validation_protocol', '')}`",
        f"- training rows/groups: `{prediction.get('training_rows', '')}` / `{prediction.get('validation_groups', '')}`",
        f"- grouped Spearman: `{prediction.get('metrics', {}).get('spearman', '')}`",
        f"- grouped R-squared: `{prediction.get('metrics', {}).get('r2', '')}`",
        f"- grouped RMSE/y_std: `{prediction.get('metrics', {}).get('rmse_over_y_std', '')}`",
        f"- grouped RMSE / y_std: `{prediction.get('metrics', {}).get('rmse', '')}` / `{prediction.get('metrics', {}).get('y_std', '')}`",
        "",
        "## Required output",
        "",
        "Return these sections in order:",
        "1. What the model predicted and what it did not establish.",
        "2. Same-model attribution summary with signed values and feature provenance.",
        "3. Training-analog audit: distinguish exact membership, sequence similarity and chemistry similarity; do not infer shared causality.",
        "4. Evidence ledger: claim, relation (supports/contradicts/context/uncertain),",
        "   source, URL/identifier, locator, assay context and evidence strength.",
        "5. Mechanism hypotheses with status, alternatives and falsification experiments.",
        "6. Missing data and the smallest useful next experiment.",
        "7. A short audit note saying whether the conclusion is prospective or retrospective.",
        "",
        "## Machine-readable inputs",
        "",
        "### Explanation packet",
        "```json",
        packet_json,
        "```",
        "",
        "### Candidate/reviewed evidence ledger",
        "```text",
        evidence_json or "No external ledger supplied; retrieve candidates before making literature claims.",
        "```",
        "",
        "## Mechanism hypotheses currently supplied by the model",
        "",
    ]
    for item in mechanisms:
        lines.append(
            f"- `{item.get('hypothesis_id', '')}` / {item.get('status', '')}: "
            f"{item.get('claim', '')} (support={item.get('confidence', '')})"
        )
    lines.extend(["", "## Top attribution records", ""])
    for item in attributions[:20]:
        lines.append(
            f"- rank {item.get('rank', '')}: `{item.get('feature', '')}` value="
            f"`{item.get('feature_value', '')}` contribution=`{item.get('contribution', '')}`; "
            f"{item.get('interpretation', '')}"
        )
    if neighbors:
        lines.extend(["", "## Ranked measured training analogs", ""])
        for item in neighbors:
            lines.append(
                f"- rank {item.get('rank', '')}: sequence=`{item.get('sequence', '')}`, "
                f"chemistry=`{item.get('modification_pattern', '')}`, measured=`{item.get('measured_value', '')}`, "
                f"sequence_identity=`{item.get('sequence_identity', '')}`, "
                f"chemistry_identity=`{item.get('chemistry_identity', '')}`, "
                f"source row=`{item.get('source_row_index', '')}`, "
                f"source_scope=`{item.get('source_scope', '')}`, "
                f"row_level_source_status=`{item.get('row_level_source_status', '')}`, "
                f"source_urls=`{item.get('source_urls', '')}`"
            )
    if secondary_models:
        lines.extend(["", "## Secondary-model attribution records", ""])
        for item in secondary_models:
            metrics = item.get("metrics", {})
            lines.append(
                f"- `{item.get('model_id', '')}` role=`{item.get('role', '')}` "
                f"representation=`{item.get('representation', '')}` method=`{item.get('attribution_method', '')}`; "
                f"Spearman=`{metrics.get('spearman', '')}` R-squared=`{metrics.get('r2', '')}` "
                f"RMSE/y_std=`{metrics.get('rmse_over_y_std', '')}` "
                f"(RMSE=`{metrics.get('rmse', '')}`, y_std=`{metrics.get('y_std', '')}`); "
                f"claim_scope=`{item.get('claim_scope', '')}`"
            )
    if evidence:
        lines.extend(["", "## Evidence already embedded in packet", ""])
        for item in evidence:
            lines.append(
                f"- `{item.get('evidence_id', '')}` [{item.get('relation', '')}] "
                f"{item.get('statement', '')} ({item.get('url', '')})"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--evidence-ledger", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_prompt(packet, args.evidence_ledger), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
