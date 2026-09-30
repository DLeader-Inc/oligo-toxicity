"""Reviewed literature provenance for the selected two-branch descriptors.

The papers motivate descriptor concepts. They do not define the project's
feature dimensions, fitted XGBoost effects, or causal interpretation.
"""

from __future__ import annotations

from typing import Any


CATALOG: dict[str, dict[str, Any]] = {
    "lit:aso_atlas2:2026": {
        "category": "dataset_and_provenance",
        "statement": (
            "ASO Atlas 2.0 provides structured antisense oligonucleotide records "
            "extracted from patents together with cohort-construction metadata."
        ),
        "source_title": "ASO Atlas 2.0 structured antisense oligonucleotide patent extraction dataset",
        "citation": "ASO Atlas 2.0, version 2.0.2; dataset DOI 10.57967/hf/8687.",
        "url": "https://doi.org/10.57967/hf/8687",
        "locator": "Dataset card, version 2.0.2, and cohort construction source",
        "claim_scope": "dataset/provenance context only; patent records do not establish mechanism",
    },
    "lit:oligogym:2025": {
        "category": "dataset_and_representation",
        "statement": (
            "OligoGym curates standardized oligonucleotide datasets and supplies "
            "natural-base, SMILES, and HELM representations for benchmarking."
        ),
        "source_title": "OligoGym: Curated Datasets and Benchmarks for Oligonucleotide Drug Discovery",
        "citation": "Rotrattanadumrong R, De Donno C. NeurIPS Datasets and Benchmarks Track. 2025.",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2025/hash/4927dd556ff6e49671e3a77ac1f12487-Abstract-Datasets_and_Benchmarks_Track.html",
        "locator": "Sections 3-4 and dataset documentation",
        "claim_scope": "dataset standardization and representation context",
    },
    "lit:helm:2012": {
        "category": "representation",
        "statement": (
            "HELM is a hierarchical notation for complex biomolecules that can "
            "record polymer monomers and connections."
        ),
        "source_title": "HELM: A Hierarchical Notation Language for Complex Biomolecule Structure Representation",
        "citation": "Zhang T et al. J Chem Inf Model. 2012. doi:10.1021/ci3001925.",
        "url": "https://doi.org/10.1021/ci3001925",
        "locator": "Abstract and notation definition",
        "claim_scope": "input notation only",
    },
    "lit:papargyri:2020": {
        "category": "sequence_chemistry_architecture",
        "statement": (
            "Papargyri et al. systematically varied LNA number and wing placement "
            "in 768 gapmers under common experimental conditions and reported "
            "architecture-dependent pharmaceutical properties."
        ),
        "source_title": "Chemical Diversity of Locked Nucleic Acid-Modified Antisense Oligonucleotides Allows Optimization of Pharmaceutical Properties",
        "citation": "Papargyri N et al. Mol Ther Nucleic Acids. 2020;19:706-717. doi:10.1016/j.omtn.2019.12.011.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6965521/",
        "locator": "Methods and Results; LNA architecture screen",
        "claim_scope": "architecture and position motivate features; paper does not define project vectors",
    },
    "lit:hagedorn:2022": {
        "category": "neurotoxicity_sequence_context",
        "statement": (
            "Hagedorn et al. linked sequence features and neuronal calcium-oscillation "
            "responses to acute mouse ICV tolerability and reported a five-feature "
            "sequence model with greater than 80% classification accuracy in its study."
        ),
        "source_title": "Acute Neurotoxicity of Antisense Oligonucleotides After Intracerebroventricular Injection Into Mouse Brain Can Be Predicted from Sequence Features",
        "citation": "Hagedorn PH et al. Nucleic Acid Ther. 2022;32:151-162. doi:10.1089/nat.2021.0071.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9221153/",
        "locator": "Abstract, Methods, and sequence-feature Results",
        "claim_scope": "study-level sequence association; not a universal mechanism",
    },
    "lit:alharbi:2020": {
        "category": "tlr_sequence_context",
        "statement": (
            "Alharbi et al. screened 192 2'-O-methyl gapmer ASOs and sequence "
            "mutants, observing sequence-dependent and chemistry-dependent effects "
            "on TLR7 inhibition and TLR8 potentiation."
        ),
        "source_title": "Rational design of antisense oligonucleotides modulating the activity of TLR7/8 agonists",
        "citation": "Alharbi AS et al. Nucleic Acids Res. 2020;48:7052-7065. doi:10.1093/nar/gkaa523.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7367172/",
        "locator": "Abstract and 192-ASO screen Results",
        "claim_scope": "TLR assay context and observed sequence/chemistry associations",
    },
    "lit:bhamra:2025": {
        "category": "sequence_chemistry_position_interaction",
        "statement": (
            "Bhamra et al. report that ASO toxicity can depend on the joint context "
            "of nucleotide sequence, sugar/backbone chemistry, and modification position."
        ),
        "source_title": "Toxicity of Antisense Oligonucleotides is Determined by the Synergistic Interplay of Chemical Modifications and Nucleotide Sequences, Not by Either Factor Alone",
        "citation": "Bhamra J et al. ChemBioChem. 2025;26:e202500584. doi:10.1002/cbic.202500584.",
        "url": "https://doi.org/10.1002/cbic.202500584",
        "locator": "Abstract; Introduction; experimental survey and Discussion",
        "claim_scope": "interaction premise only; the 237D and 456D project vectors are project-engineered",
    },
    "lit:burdick:2014": {
        "category": "sequence_motif",
        "statement": (
            "Burdick et al. tested 2- to 5-mer counts in 71 3-8-3 LNA gapmers "
            "and reported TGC and TCC associations with mouse hepatotoxicity, "
            "including motif-containing non-toxic counterexamples."
        ),
        "source_title": "Sequence motifs associated with hepatotoxicity of locked nucleic acid-modified antisense oligonucleotides",
        "citation": "Burdick AD et al. Nucleic Acids Res. 2014;42:4882-4891. doi:10.1093/nar/gku142.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4005641/",
        "locator": "Classification model Methods and Discussion",
        "claim_scope": "TGC/TCC association in the reported 3-8-3 LNA mouse study; not universal causality",
    },
    "lit:migawa:2019": {
        "category": "backbone_position",
        "statement": (
            "Migawa et al. evaluated site-specific PS replacement with alkyl "
            "phosphonate linkages and related placement to protein interactions "
            "and therapeutic profile."
        ),
        "source_title": "Site-specific replacement of phosphorothioate with alkyl phosphonate linkages enhances the therapeutic profile of gapmer ASOs by modulating interactions with cellular proteins",
        "citation": "Migawa MT et al. Nucleic Acids Res. 2019;47:5465-5479. doi:10.1093/nar/gkz247.",
        "url": "https://doi.org/10.1093/nar/gkz247",
        "locator": "Abstract and site-specific linkage experiments",
        "claim_scope": "backbone position design prior; not a fitted effect for benchmark rows",
    },
    "lit:anderson:2021": {
        "category": "backbone_position",
        "statement": (
            "Anderson et al. evaluated mesyl-phosphoramidate placement in gapmer "
            "ASOs and measured biochemical and in-vivo property changes."
        ),
        "source_title": "Towards next generation antisense oligonucleotides: mesylphosphoramidate modification improves therapeutic index and duration of effect of gapmer antisense oligonucleotides",
        "citation": "Anderson BA et al. Nucleic Acids Res. 2021;49:9026-9041. doi:10.1093/nar/gkab718.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8450106/",
        "locator": "Abstract and linkage-placement experiments",
        "claim_scope": "backbone chemistry and placement context; not a fitted effect for benchmark rows",
    },
    "lit:cns_motor_phenotype:2024": {
        "category": "endpoint_alternative_context",
        "statement": (
            "A CNS motor-phenotype study reported that reducing PS content improved "
            "gapmer tolerability and that formulation also affected acute phenotypes."
        ),
        "source_title": "Quantifying and mitigating motor phenotypes induced by antisense oligonucleotides in the central nervous system",
        "citation": "Ferguson CM et al. Mol Ther. 2024. doi:10.1016/j.ymthe.2024.10.024.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11638874/",
        "locator": "Abstract and PS-content/formulation experiments",
        "claim_scope": "endpoint-level alternative context; not direct evidence for the benchmark query sequence",
    },
    "lit:divalent_formulation:2024": {
        "category": "endpoint_alternative_context",
        "statement": (
            "A mouse CNS-delivery study reported that concentrated oligonucleotides "
            "bound divalent ions and that adding Ca2+ and/or Mg2+ to formulation "
            "prevented acute seizures without changing distribution or efficacy."
        ),
        "source_title": "Preventing acute neurotoxicity of CNS therapeutic oligonucleotides with the addition of Ca2+ and Mg2+ in the formulation",
        "citation": "Ferguson CM et al. Mol Ther Nucleic Acids. 2024;35:102359. doi:10.1016/j.omtn.2024.102359.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11567125/",
        "locator": "Abstract and ion-binding/formulation experiments",
        "claim_scope": "formulation alternative; not direct evidence for the benchmark query sequence or dose",
    },
}


DATASET_EVIDENCE_IDS = {
    "cytotoxicity": ["lit:oligogym:2025", "lit:papargyri:2020"],
    "neurotoxicity_lna": ["lit:oligogym:2025", "lit:hagedorn:2022"],
    "neurotoxicity_moe": ["lit:oligogym:2025"],
    "tlr7_immunotox": ["lit:oligogym:2025", "lit:alharbi:2020"],
    "tlr8_immunotox": ["lit:oligogym:2025", "lit:alharbi:2020"],
}


ATLAS_DATASETS = {"mouse_hepatic", "rat_hepatic", "mouse_neuro", "rat_neuro"}


ENDPOINT_CONTEXT_EVIDENCE_IDS = {
    "cytotoxicity": [],
    "neurotoxicity_lna": [],
    "neurotoxicity_moe": [
        "lit:cns_motor_phenotype:2024",
        "lit:divalent_formulation:2024",
    ],
    "tlr7_immunotox": [],
    "tlr8_immunotox": [],
}


def source_ids_for_feature(feature: str, dataset: str) -> list[str]:
    """Return reviewed sources that motivate a feature concept."""

    name = str(feature)
    ids: list[str] = []
    if name.startswith("uohe__"):
        ids.extend(["lit:helm:2012", "lit:oligogym:2025"])
    elif "__pint_" in name or name.startswith("pint_"):
        ids.extend(["lit:papargyri:2020", "lit:bhamra:2025"])
    elif name.startswith("ucontext__"):
        ids.extend(["lit:papargyri:2020", "lit:bhamra:2025"])
    elif name.startswith("seqchem__") or name.startswith("scint_"):
        ids.append("lit:bhamra:2025")
    elif name.startswith("ucore__"):
        ids.append("lit:oligogym:2025")

    if "TCC" in name or "TGC" in name:
        ids.append("lit:burdick:2014")
    if "link" in name.lower() or "PS" in name or "PO" in name:
        ids.extend(["lit:migawa:2019", "lit:anderson:2021"])
    if dataset == "neurotoxicity_lna":
        ids.append("lit:hagedorn:2022")
    if dataset in {"tlr7_immunotox", "tlr8_immunotox"}:
        ids.append("lit:alharbi:2020")
    if dataset in ATLAS_DATASETS:
        ids.append("lit:aso_atlas2:2026")
    return list(dict.fromkeys(ids))


def describe_feature(feature: str, value: float | None) -> str:
    """Describe a named project feature without assigning a mechanism."""

    name = str(feature)
    value_text = "missing" if value is None else f"{value:g}"
    if name.startswith("uohe__raw_pos_"):
        return f"Fixed-position raw HELM token indicator/count; observed value={value_text}."
    if "__pint_" in name or name.startswith("pint_"):
        return f"Project-computed position interaction derived deterministically from HELM; value={value_text}."
    if name.startswith("ucontext__anchor_"):
        return f"Project-computed landmark anchor token derived from HELM; value={value_text}."
    if name.startswith("ucontext__pool_"):
        return f"Project-computed regional pooled composition derived from HELM; value={value_text}."
    if name.startswith("ucore__"):
        return f"Project-computed global architecture, composition, or motif descriptor; value={value_text}."
    if name.startswith("seqchem__") or name.startswith("scint_"):
        return f"Project-computed sequence-by-chemistry interaction descriptor; value={value_text}."
    return f"Named project feature derived from the submitted molecular representation; value={value_text}."


def describe_feature_zh(feature: str, value: float | None) -> str:
    """用中文说明具名项目特征，不把模型特征写成生物机制。"""

    name = str(feature)
    value_text = "缺失" if value is None else f"{value:g}"
    if name.startswith("uohe__raw_pos_"):
        return f"固定位置的原始 HELM token 指示量或计数；观测值={value_text}。"
    if "__pint_" in name or name.startswith("pint_"):
        return f"由 HELM 确定性计算的位置交互特征；观测值={value_text}。"
    if name.startswith("ucontext__anchor_"):
        return f"由 HELM 计算的关键位置锚点 token；观测值={value_text}。"
    if name.startswith("ucontext__pool_"):
        return f"由 HELM 计算的区域汇总组成特征；观测值={value_text}。"
    if name.startswith("ucore__"):
        return f"项目计算的全局架构、组成或 motif 描述符；观测值={value_text}。"
    if name.startswith("seqchem__") or name.startswith("scint_"):
        return f"项目计算的序列与化学交互描述符；观测值={value_text}。"
    return f"由提交的分子表示计算的具名项目特征；观测值={value_text}。"


def dataset_source_scope(dataset: str) -> dict[str, Any]:
    """State whether a benchmark row can be resolved to its source publication."""

    if dataset == "neurotoxicity_moe":
        return {
            "source_scope": "source_collection_level",
            "row_level_source_status": "unresolved_missing_compound_table_identifier",
            "source_urls": [
                "https://github.com/Roche/OligoGym",
                "https://patents.google.com/patent/WO2020172559A1",
                "https://patents.google.com/patent/WO2016168592A2",
                "https://patents.google.com/patent/WO2021021673A1",
                "https://patents.google.com/patent/WO2015153800A2",
                "https://patents.google.com/patent/WO2020205463A1",
                "https://patents.google.com/patent/WO2020006267A1",
                "https://patents.google.com/patent/WO2020023737A1",
                "https://patents.google.com/patent/WO2020243292A1",
                "https://patents.google.com/patent/WO2020106996A1",
                "https://patents.google.com/patent/WO2022026589A1",
                "https://patents.google.com/patent/WO2022165122A1",
                "https://patents.google.com/patent/WO2021174019A1",
                "https://patents.google.com/patent/WO2022032060A2",
            ],
            "note": (
                "OligoGym attributes the endpoint to 13 CNS patents, but packaged "
                "rows lack a stable compound/table identifier. No individual row or "
                "neighbor may be assigned to one patent without an additional join."
            ),
        }
    source_id = (
        "lit:aso_atlas2:2026"
        if dataset in ATLAS_DATASETS
        else DATASET_EVIDENCE_IDS[dataset][-1]
    )
    return {
        "source_scope": "publication_dataset_level",
        "row_level_source_status": "dataset_publication_resolved_row_locator_not_recorded",
        "source_urls": [CATALOG[source_id]["url"]],
        "note": (
            "The benchmark dataset is resolved to a source publication, but this "
            "packet does not claim a supplementary-table row locator unless one is recorded."
        ),
    }
