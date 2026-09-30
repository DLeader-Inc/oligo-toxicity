"""Versioned semantic descriptor families and AutoML route definitions.

The descriptor families are deliberately disjoint. Registry one-hot is a
representation. Reliably row-joined Atlas dose/schedule columns are appended to
every route as an always-on exposure-context block rather than treated as a
selectable molecular descriptor family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.benchmark_aso_atlas2_project_representation import (
    experimental_conditions,
    project_features,
    seqchem_dataframe_for_helms,
)
from src.features.position_reencoding_features import dataframes_for_helms
from src.features.universal_oligo_representation import (
    universal_context_names,
    universal_core_names,
    universal_ohe_feature_names,
)


SCHEMA_VERSION = "motif_chemical_modification_routes_v1"


@dataclass(frozen=True)
class RouteSpec:
    name: str
    components: tuple[str, ...]
    weights: tuple[float, ...]
    route_rank: int
    description: str

    @property
    def model_count(self) -> int:
        return len(self.components)


ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec("registry_only", ("registry",), (1.0,), 1, "Registry one-hot representation"),
    RouteSpec("motif_only", ("motif",), (1.0,), 2, "sequence/motif descriptor family"),
    RouteSpec("chemistry_only", ("chemistry",), (1.0,), 3, "chemical-modification descriptor family"),
    RouteSpec("motif_chemistry", ("motif_chemistry",), (1.0,), 4, "early concatenation of both descriptor families"),
    RouteSpec("registry_motif", ("registry_motif",), (1.0,), 5, "Registry plus motif family"),
    RouteSpec("registry_chemistry", ("registry_chemistry",), (1.0,), 6, "Registry plus chemical-modification family"),
    RouteSpec("registry_motif_chemistry", ("registry_motif_chemistry",), (1.0,), 7, "Registry plus both descriptor families"),
    RouteSpec("semantic_late_25_75", ("motif", "chemistry"), (0.25, 0.75), 8, "25:75 motif/chemistry late fusion"),
    RouteSpec("semantic_late_50_50", ("motif", "chemistry"), (0.50, 0.50), 9, "50:50 motif/chemistry late fusion"),
    RouteSpec("semantic_late_75_25", ("motif", "chemistry"), (0.75, 0.25), 10, "75:25 motif/chemistry late fusion"),
    RouteSpec("legacy_fixed_50_50", ("legacy_branch_a", "legacy_branch_b"), (0.50, 0.50), 11, "current A/B fixed fusion control"),
)


def descriptor_family(name: str) -> str:
    """Map a named feature to one non-overlapping semantic family."""

    value = str(name)
    if value.startswith("assay__"):
        return "assay_context"
    if value.startswith("uohe__"):
        return "registry"
    if value.startswith("scint_"):
        return "chemical_modification"
    if value.startswith("pint_"):
        if "pint_motif_" in value and "__region_" in value:
            return "motif"
        return "chemical_modification"
    if value.startswith("ucore__"):
        motif_tokens = (
            "__kmer_", "__n_CpG", "__n_TCC", "__n_TGC", "__n_toxic",
            "__has_G", "__n_GG", "__n_TT", "__n_TG", "__gc_content",
        )
        return "motif" if any(token in value for token in motif_tokens) else "chemical_modification"
    if value.startswith("ucontext__"):
        motif_tokens = ("_base_", "gc_fraction", "global_length_")
        return "motif" if any(token in value for token in motif_tokens) else "chemical_modification"
    raise ValueError(f"unmapped AutoML feature: {value}")


def _condition_names(frame: pd.DataFrame) -> list[str]:
    return (
        ["assay__dosage_mg_per_kg", "assay__num_doses", "assay__dosing_period_days"]
        if "dosage_mg_per_kg" in frame
        else []
    )


def _with_conditions(
    matrix: np.ndarray,
    names: Iterable[str],
    conditions: np.ndarray,
    condition_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    values = np.column_stack([matrix, conditions]).astype(np.float32)
    return values, [*map(str, names), *condition_names]


def build_route_matrices(
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], pd.DataFrame]:
    """Build every unique component matrix used by the declared route grid."""

    features = project_features(frame)
    helms = frame["x"].astype(str).tolist()
    registry = features["raw"].astype(np.float32)
    core_names = list(universal_core_names())
    context_names = list(universal_context_names())
    seqchem_names = list(seqchem_dataframe_for_helms(helms[:1]).columns)
    position_names = list(dataframes_for_helms(helms[:1])["position_interactions"].columns)
    branch_b_names = [*core_names, *context_names, *seqchem_names]
    branch_a_names = [*universal_ohe_feature_names(), *position_names]
    if len(branch_a_names) != features["branch_a"].shape[1]:
        raise RuntimeError("legacy branch A feature schema mismatch")
    if len(branch_b_names) != features["branch_b"].shape[1]:
        raise RuntimeError("legacy branch B feature schema mismatch")

    descriptor_matrix = np.column_stack(
        [
            features["branch_b"],
            features["branch_a"][:, registry.shape[1] :],
        ]
    ).astype(np.float32)
    descriptor_names = [*branch_b_names, *position_names]
    families = [descriptor_family(name) for name in descriptor_names]
    motif_indices = [i for i, family in enumerate(families) if family == "motif"]
    chemistry_indices = [
        i for i, family in enumerate(families) if family == "chemical_modification"
    ]
    if set(motif_indices).intersection(chemistry_indices):
        raise RuntimeError("descriptor families overlap")
    if len(motif_indices) + len(chemistry_indices) != len(descriptor_names):
        raise RuntimeError("descriptor family mapping is incomplete")

    motif = descriptor_matrix[:, motif_indices]
    chemistry = descriptor_matrix[:, chemistry_indices]
    motif_names = [descriptor_names[i] for i in motif_indices]
    chemistry_names = [descriptor_names[i] for i in chemistry_indices]
    conditions = experimental_conditions(frame).astype(np.float32)
    condition_names = _condition_names(frame)
    molecular = {
        "registry": (registry, list(universal_ohe_feature_names())),
        "motif": (motif, motif_names),
        "chemistry": (chemistry, chemistry_names),
        "motif_chemistry": (
            np.column_stack([motif, chemistry]),
            [*motif_names, *chemistry_names],
        ),
        "registry_motif": (
            np.column_stack([registry, motif]),
            [*universal_ohe_feature_names(), *motif_names],
        ),
        "registry_chemistry": (
            np.column_stack([registry, chemistry]),
            [*universal_ohe_feature_names(), *chemistry_names],
        ),
        "registry_motif_chemistry": (
            np.column_stack([registry, motif, chemistry]),
            [*universal_ohe_feature_names(), *motif_names, *chemistry_names],
        ),
        "legacy_branch_a": (features["branch_a"], branch_a_names),
        "legacy_branch_b": (features["branch_b"], branch_b_names),
    }
    matrices: dict[str, np.ndarray] = {}
    schemas: dict[str, list[str]] = {}
    for component, (matrix, names) in molecular.items():
        matrices[component], schemas[component] = _with_conditions(
            matrix, names, conditions, condition_names
        )

    manifest_rows = [
        {
            "feature": name,
            "family": descriptor_family(name),
            "schema_version": SCHEMA_VERSION,
        }
        for name in [*universal_ohe_feature_names(), *descriptor_names, *condition_names]
    ]
    return matrices, schemas, pd.DataFrame(manifest_rows).drop_duplicates("feature")


def route_by_name(name: str) -> RouteSpec:
    try:
        return next(route for route in ROUTES if route.name == name)
    except StopIteration as error:
        raise KeyError(name) from error


def route_feature_count(route: RouteSpec, schemas: dict[str, list[str]]) -> int:
    return sum(len(schemas[component]) for component in route.components)
