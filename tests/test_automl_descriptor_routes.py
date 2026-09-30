from __future__ import annotations

import pandas as pd

from scripts.benchmark_registry_nine_endpoints import earlier_frame
from src.features.automl_descriptor_routes import (
    ROUTES,
    SCHEMA_VERSION,
    build_route_matrices,
    descriptor_family,
)


def test_semantic_mapping_is_explicit() -> None:
    assert descriptor_family("ucore__kmer_TCC") == "motif"
    assert descriptor_family("pint_motif_TGC__region_gap_mid") == "motif"
    assert descriptor_family("pint_motif_TGC__mod_L") == "chemical_modification"
    assert descriptor_family("scint_L_fraction") == "chemical_modification"
    assert descriptor_family("uohe__raw_pos_01_sugar_lna") == "registry"
    assert descriptor_family("assay__dosage_mg_per_kg") == "assay_context"


def test_route_grid_has_declared_fusions() -> None:
    names = {route.name for route in ROUTES}
    assert {"semantic_late_25_75", "semantic_late_50_50", "semantic_late_75_25"} <= names
    assert "legacy_fixed_50_50" in names


def test_built_families_are_disjoint_and_conditions_are_fixed() -> None:
    frame = earlier_frame("cytotoxicity").iloc[:8].reset_index(drop=True)
    matrices, schemas, manifest = build_route_matrices(frame)
    motif = set(schemas["motif"])
    chemistry = set(schemas["chemistry"])
    assert motif
    assert chemistry
    assert motif.isdisjoint(chemistry)
    assert matrices["motif"].shape[1] == len(schemas["motif"])
    assert manifest.schema_version.eq(SCHEMA_VERSION).all()
    assert not manifest.feature.duplicated().any()

