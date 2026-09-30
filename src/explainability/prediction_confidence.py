"""Transparent reliability components for toxicity explanation packets."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from src.explainability.evidence_packet import ConfidenceComponent, aggregate_confidence


def sequence_identity(left: str, right: str) -> float:
    """Return normalized Levenshtein identity in [0, 1]."""
    left = (left or "").upper().replace("U", "T")
    right = (right or "").upper().replace("U", "T")
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_left != char_right),
                )
            )
        previous = current
    distance = previous[-1]
    return float(max(0.0, 1.0 - distance / max(len(left), len(right))))


def nearest_sequence_identity(query: str, references: Iterable[str]) -> float:
    return max((sequence_identity(query, ref) for ref in references), default=0.0)


def robust_feature_domain_score(query: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Return `(score, nearest_distance)` after robust per-feature scaling.

    The score is a heuristic applicability measure, not a probability. Constant
    columns are ignored so sparse one-hot feature blocks do not dominate.
    """
    query = np.asarray(query, dtype=float).reshape(1, -1)
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] != query.shape[1]:
        return 0.0, float("inf")
    median = np.nanmedian(reference, axis=0)
    q25 = np.nanquantile(reference, 0.25, axis=0)
    q75 = np.nanquantile(reference, 0.75, axis=0)
    scale = q75 - q25
    variable = np.isfinite(scale) & (scale > 1e-9)
    if not np.any(variable):
        return 0.5, 0.0
    ref_scaled = (reference[:, variable] - median[variable]) / scale[variable]
    query_scaled = (query[:, variable] - median[variable]) / scale[variable]
    distances = np.sqrt(np.nanmean((ref_scaled - query_scaled) ** 2, axis=1))
    nearest = float(np.nanmin(distances)) if len(distances) else float("inf")
    score = math.exp(-max(0.0, nearest)) if math.isfinite(nearest) else 0.0
    return float(max(0.0, min(1.0, score))), nearest


def validation_score(spearman: float, n_groups: int) -> float:
    correlation = max(0.0, min(1.0, float(spearman)))
    group_factor = math.sqrt(min(max(int(n_groups), 0), 20) / 20.0)
    return correlation * group_factor


def data_support_score(n_rows: int, n_groups: int) -> float:
    row_factor = min(1.0, math.log10(max(n_rows, 1) + 1) / math.log10(3001))
    group_factor = math.sqrt(min(max(int(n_groups), 0), 20) / 20.0)
    return row_factor * group_factor


def build_confidence(
    *,
    spearman: float,
    n_rows: int,
    n_groups: int,
    sequence_identity_score: float,
    chemistry_seen: bool,
    feature_domain_score: float,
    selection_stability: float,
    provenance_completeness: float,
    exact_input_in_training: bool = False,
    exact_sequence_in_training: bool = False,
):
    applicability = (
        0.45 * max(0.0, min(1.0, sequence_identity_score))
        + 0.20 * float(bool(chemistry_seen))
        + 0.35 * max(0.0, min(1.0, feature_domain_score))
    )
    components = [
        ConfidenceComponent(
            name="grouped_validation",
            score=validation_score(spearman, n_groups),
            weight=0.35,
            basis=f"Spearman={spearman:.3f}; held-out groups={n_groups}",
            caveats=["A grouped-CV correlation is endpoint-level evidence, not an individual probability."],
        ),
        ConfidenceComponent(
            name="applicability_domain",
            score=applicability,
            weight=0.25,
            basis=(
                f"nearest sequence identity={sequence_identity_score:.3f}; "
                f"chemistry_seen={chemistry_seen}; feature-domain score={feature_domain_score:.3f}"
            ),
        ),
        ConfidenceComponent(
            name="data_support",
            score=data_support_score(n_rows, n_groups),
            weight=0.15,
            basis=f"training rows={n_rows}; independent groups={n_groups}",
        ),
        ConfidenceComponent(
            name="selection_stability",
            score=max(0.0, min(1.0, selection_stability)),
            weight=0.15,
            basis="Frequency with which the descriptor family was selected in strict nested outer folds.",
        ),
        ConfidenceComponent(
            name="provenance_completeness",
            score=max(0.0, min(1.0, provenance_completeness)),
            weight=0.10,
            basis="Fraction of prediction, data, feature and rule claims linked to auditable artifacts/citations.",
        ),
        ConfidenceComponent(
            name="prospective_independence",
            score=(0.25 if exact_input_in_training else 0.65 if exact_sequence_in_training else 1.0),
            weight=0.10,
            basis=(
                "exact input is a training member; interpretation is retrospective"
                if exact_input_in_training
                else "exact molecular sequence occurs in training; chemistry/input may differ"
                if exact_sequence_in_training
                else "no exact molecular-sequence training match"
            ),
            caveats=(
                ["Training membership demonstrates pipeline consistency, not prospective validation."]
                if exact_input_in_training
                else []
            ),
        ),
    ]
    assessment = aggregate_confidence(components)
    # Applicability and provenance cannot compensate for weak generalization.
    # Gate the weighted evidence score by grouped-validation quality.
    validation_gate = 0.5 + 0.5 * components[0].score
    assessment.score = float(assessment.score * validation_gate)
    from src.explainability.evidence_packet import confidence_level

    assessment.level = confidence_level(assessment.score)
    assessment.interpretation += " The final score is conservatively gated by grouped-validation quality."
    if exact_input_in_training:
        assessment.interpretation += " This entry is a training member, so the assessment is retrospective."
    return assessment
