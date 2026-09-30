# Oligonucleotide toxicity: prediction and interpretable evidence

Code for nine retrospective regression endpoints: HeLa caspase-3/7 response,
LNA neuronal calcium score, MOE mouse intracerebroventricular functional
observational battery (FOB) score, TLR8 potentiation, TLR7 inhibition, and
mouse/rat alanine aminotransferase (ALT) and FOB responses.

The primary model uses the complete expert descriptor fusion, a fixed
`current_balanced` XGBoost recipe, and a training-label-only scale rule.
The search policies are sensitivity controls, not the primary claim.
The three descriptor groups are encoding groups, not independent biological
causes: positional encoding and regional sequence descriptors also contain
chemical information. A small additional engineered-chemistry contribution
does not mean chemical modification is unimportant.

## Status and scope

This snapshot contains archived evidence and executable scientific entry
points. In a fresh Python 3.11 environment, 48 tests passed, the original data
and 45 partitions were rebuilt, and the primary statistics were recalculated.
All nine native model ensembles reproduced their serialization reference
predictions. A complete training rerun was not performed on the release host,
which lacked the original memory/GPU headroom. Recomputing statistics from
saved predictions is distinct from refitting every experiment.

- Code destination: https://github.com/DLeader-Inc/oligo-toxicity
- Weight destination: https://huggingface.co/DLeader/oligo-toxicity
- Original data: seven fixed-revision links and SHA-256 values in
  [`reproduce/source_downloads.json`](reproduce/source_downloads.json).
- Frozen tables, row-level explanations and provenance: `results/supporting/`.
- Compatibility adaptations and source hashes: `reproduce/source_inventory.json`.

## Environment

Use Python 3.11. The CPU environment below supports inference and result
verification. The original training used CUDA. Neural replays and GPU/CPU
refits need not be bitwise identical across runtimes.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
mkdir -p external
git clone https://github.com/Roche/OligoGym external/OligoGym
git -C external/OligoGym checkout 97f5b9f58d9e445a8ecb0218037af7465c3708c0
pip install --no-deps ./external/OligoGym
python reproduce/download_data.py
python -m pytest -q tests
python reproduce/verify_inputs.py
python reproduce/verify_results.py
python reproduce/figures.py
```

OligoGym is installed without dependencies intentionally: its original
dependency pins differ from the numerical environment used in this study.
Only the required feature/split code is used here, except when explicitly
running the separately named source replay protocol. This is not a claim of
bitwise reproduction of the source authors' unknown seeds/checkpoints.

## Model inference

Download the fixed weight revision with `python reproduce/download_models.py`.
It checks every downloaded weight and metadata file against the published
hash manifest. Input is one HELM oligonucleotide in 5′→3′ order.
For mouse/rat ALT, provide `dosage_mg_per_kg`, `num_doses`, and
`dosing_period_days` inside `assay_context`. These values must reflect the
experiment, not arbitrary defaults. Other endpoints have fixed assay
contexts and must not be interpreted as general clinical safety scores.

```bash
python reproduce/predict.py --models models --input query.json --output prediction.json
```

Example input (synthetic syntax example, not an experimentally validated design):

```json
{"endpoint":"moe_icv_fob_neurotoxicity","helm":"RNA1{[moe](A)[sp].[moe](C)[sp].d(G)[sp].d(T)[sp].d(A)[sp].d(C)[sp].[moe](G)[sp].[moe](T)}$$$$"}
```

Prediction returns the endpoint-specific original response scale, direction,
five seed predictions, and signed TreeSHAP feature contributions on the
fitted model scale. Each seed combines two descriptor branches at weights
0.5/0.5. For log1p fits, inverse transformation occurs after branch fusion
and before averaging seeds. TreeSHAP contributions are not transformed into
raw-scale causal effects. The five-seed spread is not a calibrated interval.

## Training and interpretation

The original source-native validation is retained: five fixed-seed
nucleobase-cluster holdouts for OligoGym and earliest-patent five-fold
GroupKFold for Atlas. Assigned-group and exact-HELM overlap are checked.
This is not exact-base-sequence-disjoint validation or an external cohort.
See [`reproduce/TRAINING.md`](reproduce/TRAINING.md) for the staged workflow,
controls, resource requirements, and distinction between regenerated and
archived evidence.

## Licences and limitations

Project-owned code is MIT. Third-party code, original data, saved labels,
derived evidence and weights are **not** covered by the code MIT licence.
Read [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). MOE mouse weights use
Apache-2.0. TLR7/TLR8 weights use CC BY-NC 4.0 and are not offered for
commercial use. The other six endpoint weights use CC BY 4.0. There is no
single unrestricted licence for the entire mixed model collection.

These are retrospective research models, not clinical decision tools.
Attribution, similarity and literature evidence generate testable hypotheses,
not proof that a chemical change will improve safety.
