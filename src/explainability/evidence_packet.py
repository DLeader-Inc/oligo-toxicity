"""Typed, serializable evidence packets for oligonucleotide predictions."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


EvidenceOrigin = Literal[
    "measured",
    "curated",
    "model_derived",
    "rule_derived",
    "retrieved",
    "generated_hypothesis",
]
EvidenceRelation = Literal["supports", "contradicts", "context", "uncertain"]


@dataclass
class EvidenceItem:
    evidence_id: str
    category: str
    origin: EvidenceOrigin
    relation: EvidenceRelation
    statement: str
    source_title: str
    citation: str = ""
    url: str = ""
    local_artifact: str = ""
    locator: str = ""
    evidence_strength: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributionItem:
    rank: int
    feature: str
    contribution: float
    direction: Literal["raises_prediction", "lowers_prediction", "neutral"]
    family: str
    interpretation: str
    feature_value: float | None = None
    provenance_evidence_ids: list[str] = field(default_factory=list)
    causal_claim: bool = False


@dataclass
class ConfidenceComponent:
    name: str
    score: float
    weight: float
    basis: str
    caveats: list[str] = field(default_factory=list)


@dataclass
class ConfidenceAssessment:
    score: float
    level: Literal["high", "moderate", "low", "very_low"]
    interpretation: str
    calibrated_probability: bool
    components: list[ConfidenceComponent]


@dataclass
class PredictionSource:
    endpoint: str
    value: float
    model_family: str
    model_artifact: str
    descriptor_family: str
    representation: str
    training_mode: str
    validation_protocol: str
    training_rows: int
    validation_groups: int
    metrics: dict[str, float]
    artifact_sha256: str = ""
    dataset_snapshot_sha256: str = ""
    feature_schema_sha256: str = ""
    response_direction: str = ""
    label_unit: str = ""
    uncertainty_interval: dict[str, Any] = field(default_factory=dict)


@dataclass
class NeighborEvidence:
    rank: int
    dataset: str
    source_row_index: int
    sequence: str
    helm: str
    modification_pattern: str
    target: str
    measured_value: float | str
    sequence_identity: float
    chemistry_identity: float
    combined_similarity: float
    exact_sequence_match: bool
    exact_input_match: bool
    citation: str = ""
    local_artifact: str = ""
    normalized_levenshtein_identity: float | None = None
    source_scope: str = "dataset_level"
    row_level_source_status: str = "unresolved"
    source_urls: list[str] = field(default_factory=list)
    response_direction: str = ""
    label_unit: str = ""


@dataclass
class SecondaryModelExplanation:
    model_id: str
    role: Literal["secondary_model"]
    representation: str
    prediction: float | None
    attribution_method: str
    model_artifact: str
    performance_artifact: str
    attribution_artifacts: dict[str, str]
    metrics: dict[str, float] = field(default_factory=dict)
    claim_scope: str = "secondary_model_behavior_not_primary_model_explanation_or_causality"
    limitations: list[str] = field(default_factory=list)


@dataclass
class MechanismHypothesis:
    hypothesis_id: str
    name: str
    status: Literal["established_context", "supported_candidate", "novel_candidate", "unresolved"]
    claim: str
    confidence: float
    confidence_kind: str = "heuristic_hypothesis_support"
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    alternative_explanations: list[str] = field(default_factory=list)
    falsification_tests: list[str] = field(default_factory=list)
    literature_queries: list[str] = field(default_factory=list)
    patent_queries: list[str] = field(default_factory=list)
    counterevidence_queries: list[str] = field(default_factory=list)
    evidence_review_status: str = "candidate_not_independently_reviewed"


@dataclass
class ExplanationPacket:
    schema_version: str
    entry: dict[str, Any]
    prediction: PredictionSource
    confidence: ConfidenceAssessment
    attributions: list[AttributionItem]
    mechanisms: list[MechanismHypothesis]
    evidence: list[EvidenceItem]
    limitations: list[str]
    neighbors: list[NeighborEvidence] = field(default_factory=list)
    secondary_models: list[SecondaryModelExplanation] = field(default_factory=list)
    integrity: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload + "\n", encoding="utf-8")
        return payload


def confidence_level(score: float) -> Literal["high", "moderate", "low", "very_low"]:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "moderate"
    if score >= 0.40:
        return "low"
    return "very_low"


def aggregate_confidence(components: list[ConfidenceComponent]) -> ConfidenceAssessment:
    if not components:
        score = 0.0
    else:
        total_weight = sum(max(0.0, component.weight) for component in components)
        if total_weight <= 0:
            score = 0.0
        else:
            score = sum(
                max(0.0, min(1.0, component.score)) * max(0.0, component.weight)
                for component in components
            ) / total_weight
    score = float(max(0.0, min(1.0, score)))
    return ConfidenceAssessment(
        score=score,
        level=confidence_level(score),
        interpretation=(
            "Evidence-based reliability score combining grouped validation, applicability domain, "
            "data support, model-selection stability, and provenance completeness."
        ),
        calibrated_probability=False,
        components=components,
    )
