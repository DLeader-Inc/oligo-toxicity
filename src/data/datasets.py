"""
Dataset registry and loaders for OligoGym toxicity datasets.

Centralizes the metadata for all 5 toxicity datasets so downstream code
can iterate over them uniformly. Wraps OligoGym's ``DatasetDownloader`` so
downloads happen once and are cached locally by the underlying library.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass(frozen=True)
class ToxicityDataset:
    """Metadata for a single OligoGym toxicity dataset."""

    oligogym_key: str       # Key used by oligogym.data.DatasetDownloader
    model_key: str          # Short name for our output files
    endpoint: str           # Human-readable endpoint description
    chemistry: str          # Dominant chemistry class
    mechanism: str          # Toxicity mechanism per our taxonomy (A1/B1/...)
    citation: str           # Source paper / citation
    canonical_endpoint_id: str = ""
    assay_receptor: str = ""
    response_direction: str = ""
    label_unit: str = ""
    context_dose: str = ""
    context_dose_unit: str = ""
    context_exposure_time: str = ""
    context_exposure_time_unit: str = ""
    context_cell_density: str = ""
    context_cell_line: str = ""
    context_species: str = ""
    context_delivery: str = ""
    context_route: str = ""
    context_assay: str = ""
    context_scope: str = ""
    context_status: str = ""
    context_note: str = ""
    context_source: str = ""


# ── The 5 toxicity datasets (see ANALYSIS_REPORT.md for details) ──────────
TOXICITY_DATASETS: List[ToxicityDataset] = [
    ToxicityDataset(
        oligogym_key="Cytotox LNA",
        model_key="cytotoxicity",
        endpoint="Caspase 3/7 (HeLa, hepatotoxicity proxy)",
        chemistry="LNA gapmer",
        mechanism="A1 + B2 + B3 + C1 (off-target RNase H1, motifs, G4, LNA)",
        citation="Papargyri et al. 2020, Mol. Ther. Nucleic Acids 19:706-717; doi:10.1016/j.omtn.2019.12.011",
        canonical_endpoint_id="hela_caspase_3_7_cytotoxicity",
        response_direction="higher value indicates higher cytotoxic response",
        label_unit="unitless min-max-scaled mean caspase activation",
        context_dose="100",
        context_dose_unit="nM",
        context_exposure_time="24",
        context_exposure_time_unit="hours",
        context_cell_line="HeLa",
        context_species="human",
        context_delivery="Lipofectamine 2000",
        context_route="in vitro transfection",
        context_assay="caspase 3/7",
        context_scope="dataset_constant",
        context_status="fixed_endpoint_contract",
        context_note="Packaged rows are a 100 nM HeLa caspase screen; dose-response rows reported in the paper are not separate packaged labels.",
        context_source="Papargyri et al. 2020; OligoGym metadata",
    ),
    ToxicityDataset(
        oligogym_key="Neurotox LNA",
        model_key="neurotoxicity_lna",
        endpoint="Ca2+ oscillation score (primary rat cortical neurons; 25 uM)",
        chemistry="LNA gapmer",
        mechanism="B4 (Ca2+ dysregulation, AMPA receptor)",
        citation="Hagedorn et al. 2022, Nucleic Acid Therapeutics 32:151-162; doi:10.1089/nat.2021.0071",
        canonical_endpoint_id="lna_neurotoxicity",
        response_direction=(
            "lower raw calcium-oscillation score is associated with greater "
            "acute neurotoxicity; higher score is associated with lower "
            "acute neurotoxicity"
        ),
        label_unit="FLIPR calcium oscillation score",
        context_dose="25",
        context_dose_unit="uM",
        context_cell_line="primary rat cortical neurons",
        context_species="rat",
        context_route="in vitro incubation",
        context_assay="FLIPR calcium oscillation",
        context_scope="dataset_constant",
        context_status="fixed_endpoint_contract",
        context_note=(
            "Packaged y is the untransformed calcium-oscillation score; lower "
            "scores were associated with worse acute mouse ICV tolerability. "
            "The related mouse ICV tolerability subset is not the packaged label."
        ),
        context_source="Hagedorn et al. 2022; OligoGym metadata",
    ),
    ToxicityDataset(
        oligogym_key="Neurotox MOE",
        model_key="neurotoxicity_moe",
        endpoint="FOB behavioral score (mouse ICV)",
        chemistry="MOE gapmer",
        mechanism="B4 (in vivo neurotoxicity)",
        citation="13 Ionis CNS patents (compiled into OligoGym)",
        canonical_endpoint_id="moe_icv_fob_neurotoxicity",
        response_direction="higher value indicates greater behavioral toxicity",
        label_unit="FOB behavioral score",
        context_exposure_time="3",
        context_exposure_time_unit="hours",
        context_species="mouse",
        context_route="intracerebroventricular",
        context_assay="functional observational battery",
        context_scope="source_level_not_row_linked",
        context_status="needs_patent_row_join",
        context_note="OligoGym reports 13 patent sources and rounded 3h FOB labels, but packaged rows lack compound/source-table IDs needed for dose/protocol joins.",
        context_source="OligoGym metadata; 13 patent URLs",
    ),
    ToxicityDataset(
        oligogym_key="TLR7",
        model_key="tlr7_immunotox",
        endpoint="TLR8 potentiation (NF-kB reporter, HEK293-TLR8; relative to R848=1.0)",
        chemistry="2'OMe gapmer",
        mechanism="B1 (innate immune activation)",
        citation="Alharbi et al. 2020, Nucleic Acids Res. 48:7052",
        canonical_endpoint_id="tlr8_potentiation",
        assay_receptor="TLR8",
        response_direction="higher value indicates stronger TLR8 potentiation",
        label_unit="relative reporter response; R848=1.0",
        context_dose="100",
        context_dose_unit="nM",
        context_cell_line="HEK293-TLR8",
        context_species="human",
        context_route="in vitro reporter assay",
        context_assay="NF-kB luciferase reporter with R848 challenge; TLR8",
        context_scope="dataset_constant",
        context_status="fixed_endpoint_contract",
        context_note="Packaged endpoint keeps the 100 nM TLR8 potentiation condition; intended ASO transcript target is not the assay receptor.",
        context_source="Alharbi et al. 2020; OligoGym metadata",
    ),
    ToxicityDataset(
        oligogym_key="TLR8",
        model_key="tlr8_immunotox",
        endpoint="TLR7 inhibition (NF-kB reporter, HEK293-TLR7; relative to R848=100)",
        chemistry="2'OMe gapmer",
        mechanism="B1 (innate immune activation)",
        citation="Alharbi et al. 2020, Nucleic Acids Res. 48:7052",
        canonical_endpoint_id="tlr7_inhibition",
        assay_receptor="TLR7",
        response_direction="lower residual reporter value indicates stronger TLR7 inhibition",
        label_unit="relative reporter response; R848=100",
        context_dose="100",
        context_dose_unit="nM",
        context_cell_line="HEK293-TLR7",
        context_species="human",
        context_route="in vitro reporter assay",
        context_assay="NF-kB luciferase reporter with R848 challenge; TLR7",
        context_scope="dataset_constant",
        context_status="fixed_endpoint_contract",
        context_note="Packaged endpoint keeps the 100 nM TLR7 inhibition condition; intended ASO transcript target is not the assay receptor.",
        context_source="Alharbi et al. 2020; OligoGym metadata",
    ),
]


def get_dataset(model_key: str) -> ToxicityDataset:
    """Look up a dataset by legacy model key or canonical endpoint id."""
    for ds in TOXICITY_DATASETS:
        if (
            ds.model_key == model_key
            or ds.canonical_endpoint_id == model_key
            or ds.oligogym_key == model_key
        ):
            return ds
    raise KeyError(f"Unknown model_key: {model_key}. Choose from "
                   f"{[d.model_key for d in TOXICITY_DATASETS]} or canonical endpoint ids "
                   f"{[d.canonical_endpoint_id for d in TOXICITY_DATASETS]}")


def load_oligogym(oligogym_key: str) -> pd.DataFrame:
    """Download (cached) and load an OligoGym dataset.

    Returns a DataFrame with columns: ``x`` (HELM), ``y`` (the packaged
    endpoint-specific training label), ``y_raw`` (source-scale label),
    ``targets`` (gene), ``smiles``, and ``fasta``. The relationship between
    ``y`` and ``y_raw`` is endpoint specific; it is not universally 0-1
    normalization.
    """
    # Release adaptation: prefer the SHA-256-verified, fixed-revision download.
    # This preserves packaged rows and prevents a moving upstream download.
    from pathlib import Path
    import hashlib
    import json
    release_root = Path(__file__).resolve().parents[2]
    registry_path = release_root / "reproduce/source_downloads.json"
    if registry_path.exists():
        source_files = {
            "Cytotox LNA": "Papargyri_2020_1_processed.csv.gz",
            "Neurotox LNA": "Hagedorn_2022_1_processed.csv.gz",
            "Neurotox MOE": "MOE_Neurotox_1_processed.csv.gz",
            "TLR7": "Alharbi_2020_1_processed.csv.gz",
            "TLR8": "Alharbi_2020_2_processed.csv.gz",
        }
        if oligogym_key in source_files:
            filename = source_files[oligogym_key]
            cached = release_root / "data/external/oligogym" / filename
            if not cached.exists():
                raise FileNotFoundError("Run python reproduce/download_data.py first")
            expected = next(row["downloaded_sha256"] for row in json.loads(registry_path.read_text()) if row["file"] == filename)
            if hashlib.sha256(cached.read_bytes()).hexdigest() != expected:
                raise ValueError("Source data hash mismatch: " + filename)
            return pd.read_csv(cached)
    from oligogym.data import DatasetDownloader  # imported lazily
    downloader = DatasetDownloader()
    return downloader.download(dataset_key=oligogym_key, verbose=0).data.copy()


def toxicity_context_record(dataset: ToxicityDataset | str) -> dict[str, object]:
    """Return normalized experimental context for one toxicity endpoint.

    Target identifiers and source IDs are intentionally excluded from the
    context hash by ``experimental_context_id``. The returned scope/status
    fields say whether the context is row-level trainable, a dataset constant,
    or source-level information that still needs a reliable row join.
    """

    ds = get_dataset(dataset) if isinstance(dataset, str) else dataset
    from src.features.experimental_context_features import (  # imported lazily
        experimental_context_id,
        normalize_context_record,
    )

    normalized = normalize_context_record(
        {
            "dose": ds.context_dose,
            "dose_unit": ds.context_dose_unit,
            "exposure_time": ds.context_exposure_time,
            "exposure_time_unit": ds.context_exposure_time_unit,
            "cell_density": ds.context_cell_density,
            "cell_line": ds.context_cell_line,
            "species": ds.context_species,
            "delivery": ds.context_delivery,
            "route": ds.context_route,
            "assay": ds.context_assay,
            "source_id": "",
            "target_group": "",
        }
    )
    normalized["experimental_context_id"] = experimental_context_id(normalized)
    normalized["experimental_context_scope"] = ds.context_scope
    normalized["experimental_context_status"] = ds.context_status
    normalized["experimental_context_note"] = ds.context_note
    normalized["experimental_context_source"] = ds.context_source
    normalized["model_key"] = ds.model_key
    normalized["oligogym_key"] = ds.oligogym_key
    normalized["canonical_endpoint_id"] = ds.canonical_endpoint_id
    normalized["assay_receptor"] = ds.assay_receptor
    return normalized


def append_toxicity_context(
    frame: pd.DataFrame,
    dataset: ToxicityDataset | str,
) -> pd.DataFrame:
    """Attach endpoint-level experimental context columns to a data frame."""

    context = toxicity_context_record(dataset)
    output = frame.copy()
    for key, value in context.items():
        output[key] = value
    return output


def load_oligogym_with_context(model_key: str) -> pd.DataFrame:
    """Load one toxicity dataset and append its context contract columns."""

    ds = get_dataset(model_key)
    return append_toxicity_context(load_oligogym(ds.oligogym_key), ds)


def load_all_toxicity_datasets() -> dict:
    """Load all 5 toxicity datasets. Returns ``{model_key: DataFrame}``."""
    return {ds.model_key: load_oligogym(ds.oligogym_key) for ds in TOXICITY_DATASETS}


def summary_table() -> pd.DataFrame:
    """Return a tidy DataFrame summarizing all 5 datasets (no data loading)."""
    return pd.DataFrame([{
        "model_key": ds.model_key,
        "oligogym_key": ds.oligogym_key,
        "endpoint": ds.endpoint,
        "chemistry": ds.chemistry,
        "mechanism": ds.mechanism,
        "citation": ds.citation,
        "canonical_endpoint_id": ds.canonical_endpoint_id,
        "assay_receptor": ds.assay_receptor,
        "response_direction": ds.response_direction,
        "label_unit": ds.label_unit,
        "context_scope": ds.context_scope,
        "context_status": ds.context_status,
        "context_source": ds.context_source,
    } for ds in TOXICITY_DATASETS])
