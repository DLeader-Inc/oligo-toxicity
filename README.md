# Oligonucleotide toxicity: prediction and interpretable evidence

Code for nine retrospective regression endpoints: HeLa caspase-3/7 response,
LNA neuronal calcium score, MOE mouse intracerebroventricular functional
observational battery (FOB) score, TLR8 potentiation, TLR7 inhibition, and
mouse/rat alanine aminotransferase (ALT) and FOB responses.

The primary model uses the complete expert descriptor fusion, a fixed
`current_balanced` XGBoost recipe, and a training-label-only scale rule.
The three descriptor groups are encoding groups, not independent biological
causes: positional encoding and regional sequence descriptors also contain
chemical information. A small additional engineered-chemistry contribution
does not mean chemical modification is unimportant.

## What this workflow reproduces

Follow steps 1–4 to download the original data and reproduce the published
primary tables from the saved predictions. Step 5 downloads and runs the
published models. Step 6 is a separate, optional training run using the same
fixed recipe and seeds; it does not introduce new experiments.

The completed reproduction audit found:

- The three primary comparison tables match when recomputed from archived
  predictions, with numerical checks at `rtol=1e-10, atol=1e-10`.
- Refitting the primary recipe on CPU gave mean Spearman **0.603018**, compared
  with **0.602890** in the archived run. Each endpoint changed by less than
  **0.0074**. These are similar results, not a claim of bitwise equality across
  CPU/GPU environments.
- The updated environment and entry points passed **86 tests**. All nine
  freshly exported ensembles reproduced their retained inference references
  exactly in the tested CPU environment. This tests inference consistency,
  not a second full training run.

The original published evidence remains the reference. Do not replace the
archived tables or change a seed to make a new run look closer to them.

- Code destination: https://github.com/DLeader-Inc/oligo-toxicity
- Weight destination: https://huggingface.co/DLeader/oligo-toxicity
- Original data: seven fixed-revision links and SHA-256 values in
  [`reproduce/source_downloads.json`](reproduce/source_downloads.json).
- Frozen tables, row-level explanations and provenance: `results/supporting/`.
- Compatibility adaptations and source hashes: `reproduce/source_inventory.json`.

## 1. Clone the code

Use a new directory so an earlier training cache cannot be mistaken for a
fresh run. All commands below assume a Linux shell at the repository root.

```bash
git clone https://github.com/DLeader-Inc/oligo-toxicity.git
cd oligo-toxicity
git rev-parse HEAD
```

Record the printed code revision with your results. This README describes the
files in its own revision. An older checkout may not contain the preparation
and environment-check entry points used below.

## 2. Create an isolated CPU environment

Use Python 3.11 in a new isolated environment. CPU supports inference, result
verification and training on a host with adequate resources. The original
training used CUDA. Neural replays and GPU/CPU
refits need not be bitwise identical across runtimes.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
mkdir -p external
git clone https://github.com/Roche/OligoGym external/OligoGym
git -C external/OligoGym checkout 97f5b9f58d9e445a8ecb0218037af7465c3708c0
# Import the pinned source with the project's compatibility profile.
export PYTHONPATH="$PWD/external/OligoGym${PYTHONPATH:+:$PYTHONPATH}"
python reproduce/check_environment.py
```

The pinned OligoGym source is imported directly, without changing its code or
installing its incompatible all-feature distribution metadata. Its original
dependency pins differ from this study's numerical environment. This project's
requirements declare the tested subset; TabPFN, Lightning and other unused
upstream features are not part of this compatibility profile. Keep the source
checkout and set `PYTHONPATH` in each new shell or service environment.
`check_environment.py` reports actual import versions, distribution metadata,
source commit and `pip check` separately, and fails on mismatches. Do not modify
an existing shared environment to make it pass.
Only the required feature/split code is used here, except when explicitly
running the separately named source replay protocol. This is not a claim of
bitwise reproduction of the source authors' unknown seeds/checkpoints.

Expected result: the environment check exits with status 0, all import/version
checks pass, and `pip check` reports `No broken requirements found.` The
CPU build of PyTorch is sufficient. No GPU or API token is required.

When returning in a new shell, run these commands again before other steps:

```bash
cd /absolute/path/to/oligo-toxicity
. .venv/bin/activate
export PYTHONPATH="$PWD/external/OligoGym${PYTHONPATH:+:$PYTHONPATH}"
```

## 3. Download data and verify inputs

```bash
python reproduce/download_data.py
python reproduce/prepare_inputs.py
python -m pytest -q tests
python reproduce/verify_inputs.py
```

The downloader verifies seven original files against fixed-revision SHA-256
values. The loaders create nine endpoint datasets. Preparation verifies or
rebuilds the required clustering file from those inputs, without fitting models
or copying old predictions. A differing existing file is rejected, not silently
overwritten. Downloads need Internet access; cached, verified data can be reused.

Expected checks:

| Check | Expected result |
|---|---|
| Project tests | 86 passed in the verified revision |
| Endpoint datasets | 9 |
| Evaluation partitions | 45 |
| Paired test records | 14,450, including repeat appearances across partitions |
| Assigned-group and exact-HELM overlap | 0 |
| Partitions with canonical-base-sequence overlap | 18; reported, not removed |

Input checks are saved to `rerun/input_verification.json`. The existing
comparison uses the original source grouping, not an independent external
cohort or an exact-base-sequence-disjoint evaluation.

## 4. Reproduce the published tables and figures

Run this step **before** refitting into any archived output location:

```bash
python reproduce/verify_results.py
python reproduce/figures.py
```

`verify_results.py` recalculates the comparison from the archived row-level
predictions. It checks every column in these published tables and exits
nonzero on a mismatch:

- `source_vs_final_endpoint_effects.csv`
- `source_vs_final_pairing_audit.csv`
- `source_vs_final_system_tests.csv`

Read `rerun/primary_statistics/verification.json` for the checks and
`rerun/primary_statistics/source_vs_final_endpoint_effects.csv` for the
endpoint scores. All three table records should have `matched: true`.
The figure command writes eight numerical figures as PNG, SVG and PDF under
`rerun/figures/`, with a separate `verification.json`.

These commands reproduce the archived evidence; they do not retrain models.
Figures read `results/supporting/`, so they remain archived-result figures
even after an optional refit. Do not label them as newly trained results.

## 5. Download models and run a prediction

Download the fixed weight revision with `python reproduce/download_models.py`.
It checks every downloaded weight and metadata file against the published
hash manifest. Input is one HELM oligonucleotide in 5′→3′ order.
For mouse/rat ALT, provide `dosage_mg_per_kg`, `num_doses`, and
`dosing_period_days` inside `assay_context`. These values must reflect the
experiment, not arbitrary defaults. Other endpoints have fixed assay
contexts and must not be interpreted as general clinical safety scores.

```bash
python reproduce/download_models.py
python reproduce/predict.py --models models --input query.json --output prediction.json
python reproduce/predict.py --models models --input query.json --explain --output explanation.json
```

Before running the prediction commands, save the following as `query.json`
at the repository root (synthetic syntax example, not a validated design):

```json
{"endpoint":"moe_icv_fob_neurotoxicity","helm":"RNA1{[moe](A)[sp].[moe](C)[sp].d(G)[sp].d(T)[sp].d(A)[sp].d(C)[sp].[moe](G)[sp].[moe](T)}$$$$"}
```

Prediction returns the endpoint-specific original response scale, direction,
five seed predictions, and signed TreeSHAP feature contributions on the
fitted model scale. Each seed combines two descriptor branches at weights
0.5/0.5. For log1p fits, inverse transformation occurs after branch fusion
and before averaging seeds. TreeSHAP contributions are not transformed into
raw-scale causal effects. The five-seed spread is not a calibrated interval.

`--explain` returns the numerical evidence packet, evidence-bound Chinese and
English text, and a fully assembled prompt for a downstream agent. Member
counts, sequences, model scales, feature values and contributions come from
the current prediction, not a manuscript case. The complete feature matrix
supports arithmetic checks; the agent prompt carries top features and summaries.
`reproduce/explain.py` can also render a saved production prediction packet.
Validation checks branch fusion, per-member inverse transformation, full SHAP
sums, top-feature summaries and exact template fidelity. It does not certify
free-form LLM prose or biological mechanisms. Measured neighbors, membership,
correlation clusters and reviewed literature are explicitly absent unless a
separate evidence workflow supplies them. No specific chemical edit is claimed
to improve an assay merely from feature importance.

The historical `build_paper_interpretability_agent.py` remains a fixed research
case template. It is not used by this production explanation endpoint.

The downloader pins the model repository revision in
`reproduce/model_release.json` and checks all 107 weight/metadata files. It
refuses to overwrite an existing file with a different hash. Prediction checks
weight hashes, feature order and TreeSHAP reconstruction before returning a
score. The optional explanation is not required to reproduce the score tables.

## 6. Optional refit with the published recipe

Use a host with adequate memory and disk. CPU training is supported; a GPU is
optional. The launcher requires at least 20 GiB available host memory and
20 GiB free disk, caps the process tree at 48 GiB RSS, and applies an additional
6 GiB free-VRAM requirement only when CUDA is requested. It performs a bounded
probe before training and records telemetry under `rerun/resources/`.

### Keep the archived reference separate

After completing steps 1–4 in a fresh checkout, move only the three run
directories below aside. This preserves the archived predictions and prevents
resuming an old fit. It leaves the other archived analysis inputs in place.
Run this block **once**, not when resuming an interrupted refit. `mkdir` without
`-p` deliberately stops the block if the backup directory already exists.

```bash
(
  set -e
  mkdir archived_primary_comparison
  mv outputs/expert_rule_scale_v1/expert_rule_scale archived_primary_comparison/primary
  mv outputs/oligogym_neurips2025_same_split archived_primary_comparison/source_oligogym
  mv outputs/aso_atlas2_four_stage_factorial_v1/full archived_primary_comparison/source_atlas
)
```

Do not delete `results/supporting/` or the remaining `outputs/` directories.
They contain the frozen references and additional inputs required by the
existing reporting code. Stop and inspect any failed preparation command
before continuing.

### Run the established comparison

First inspect the commands without launching training:

```bash
python reproduce/workflow.py primary --device cpu
python reproduce/workflow.py source_oligogym --device cpu
python reproduce/workflow.py source_atlas --device cpu
```

Then execute the same fixed stages:

```bash
python reproduce/workflow.py primary --device cpu --run
python reproduce/workflow.py source_oligogym --device cpu --run
python reproduce/workflow.py source_atlas --device cpu --run
```

Run them sequentially and stop if a command fails. `source_oligogym` prepares
its clustering prerequisite automatically. To resume, rerun the same command
with the same configuration and existing new-run directory. To test a genuinely
independent refit, use another clean checkout and the same preparation sequence,
rather than reusing completed checkpoints. Never use a different device or
library version in a partially completed run.

### Recalculate and compare the new primary table

```bash
python reproduce/verify_inputs.py
python scripts/analyze_expert_rule_paper.py --out-dir rerun/refit_primary_statistics
```

The new primary scores are in
`rerun/refit_primary_statistics/source_vs_final_endpoint_effects.csv`. Compare
them with the untouched publication reference, rather than running the strict
archived-table equality check on newly fitted predictions:

```bash
python - <<'PY'
import pandas as pd

old = pd.read_csv("results/supporting/source_vs_final_endpoint_effects.csv")
new = pd.read_csv("rerun/refit_primary_statistics/source_vs_final_endpoint_effects.csv")
cols = ["dataset", "final_spearman", "source_spearman"]
comparison = old[cols].merge(new[cols], on="dataset", suffixes=("_published", "_refit"), validate="one_to_one")
assert len(comparison) == 9
for model in ("final", "source"):
    comparison[model + "_difference"] = comparison[model + "_spearman_refit"] - comparison[model + "_spearman_published"]
comparison.to_csv("rerun/refit_primary_statistics/published_vs_refit.csv", index=False)
print(comparison.to_string(index=False))
print("Primary mean Spearman:", new.final_spearman.mean())
PY
```

The analysis entry point also emits auxiliary tables using archived inputs
that were not refitted in these three stages. Only its `source_vs_final_*`
comparison tables are fully refreshed by the workflow above. This workflow
does not claim to retrain every auxiliary analysis or mechanism example.

To export your new full-data models, download the published catalogue if not
already present, then export only your own trusted training artifacts:

```bash
python reproduce/download_models.py
python reproduce/export_refit.py --run-dir outputs/expert_rule_scale_v1/expert_rule_scale --output rerun/native_models
python reproduce/predict.py --models rerun/native_models --input query.json --output rerun/refit_prediction.json
```

The export checks agreement with the fitted models before saving native
XGBoost files. It does not replace the published models under `models/`.
Do not load untrusted downloaded joblib/pickle files.

## Fixed random seeds and expected agreement

Our models do not use an unspecified random seed. The primary workflow uses
five deterministic seed members, single-threaded XGBoost (`n_jobs=1`) and
`tree_method="hist"`. The seed schedule is implemented in
`scripts/benchmark_xgb_robust_nested.py::_seed_values`:

```text
seed = 100003 + 10007 * dataset_index + 101 * partition + 7919 * member_index
```

`dataset_index` follows the fixed `EARLIER + ATLAS` order, `member_index` starts
at zero, and full-data refits use partition identifier 999. Within a member,
branch seeds use the fixed `499 * branch_index` offset. Saved selection and
model manifests record the seed schedule. Do not replace it with an arbitrary
`--seed 42` or change the dataset order. Source replay has its own fixed seed
schedule in the source-replay entry points; it is not an assertion that the
source authors used these seeds.

There are three different checks:

| Reproduction target | Expected agreement |
|---|---|
| Recompute tables from the same saved predictions | Numerical equality at the stated tolerance |
| Predict with the same native weights and inputs | Identical outputs in the verified environment |
| Fit again with the same seeds | Check predictions and scores in the same pinned environment; fixed seeds alone do not guarantee bitwise equality across hardware or software builds |

Save the code revision, `check_environment.py` output, model manifests, and
resource logs alongside a refit. Matching means alone are not sufficient to
claim every prediction or weight is identical. The completed CPU refit above
supports similar published scores; it does not establish universal cross-device
bitwise determinism.

## Troubleshooting

- **Import or dependency errors:** use a fresh Python 3.11 environment, restore
  the `PYTHONPATH` command, and run `check_environment.py`. Do not silently
  substitute newer numerical packages or install the incompatible full upstream
  dependency set into this environment.
- **Hash mismatch:** stop and inspect the differing file. The tools intentionally
  do not overwrite it. Do not bypass the hash check to obtain a pass.
- **Resource preflight failure:** use a larger CPU host or suitable GPU host.
  A CPU device is valid; insufficient available memory is a separate issue.
- **A run finishes immediately:** inspect the checkpoint logs. It may have
  resumed completed work rather than independently refitted models.
- **Strict table check fails after refitting:** use the new-versus-published
  comparison above. `verify_results.py` is for archived predictions, not an
  assertion that retraining must recreate every archived digit.

## Validation context

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
