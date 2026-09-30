# Reproduction map

## Three different verification tasks

1. `download_data.py` retrieves seven fixed-revision upstream files and checks
   their hashes. The loaders create nine endpoint frames without changing row
   order, chemistry filters, labels or patent grouping.
2. `verify_results.py` recomputes the 14,450-row, 45-partition primary comparison
   from frozen source/final predictions and checks the published tables.
3. Refitting below creates new outer and full-data members from the source
   inputs. CPU/GPU and neural initialization differences can change predictions.
   Recomputed statistics are not evidence that a complete refit has been run.

## Resource safety

Before training, inspect RAM, swap, GPU memory, disk and current processes.
On the study machine the required headroom is at least 20 GiB host available,
at most 48 GiB process-tree RSS, at least 6 GiB unused GPU memory, and at least
20 GiB free disk. The original experiments used a larger GPU host. A small
CPU-only verification host cannot reproduce those formal runs safely.

The workflow launcher below checks these requirements and runs a bounded
probe followed by the formal command under the resource guard. It preserves
telemetry and checkpoints. Never run the formal command after a failed probe.
The launcher defaults to printing commands. `--run` explicitly starts work.
Use a fresh checkout or preserve a separate copy of archived results before
refitting into their conventional output directories.

```bash
python reproduce/workflow.py primary --device cuda
python reproduce/workflow.py primary --device cuda --run
```

## Stage order

| Stage | Launcher name | Output and purpose |
|---|---|---|
| Source data | `python reproduce/download_data.py` | Exact packaged CSV and Atlas Parquet files |
| Primary grouped evaluation and full-data refit | `primary` | `outputs/expert_rule_scale_v1/expert_rule_scale` |
| OligoGym paper-best local replay | `source_oligogym` | Same fixed-seed cluster holdouts, not a recovered original paper seed |
| Atlas source and XGBoost controls | `source_atlas` | Source patent folds and four representation/model routes |
| Fixed expert identity | `fixed_expert` | Scale-only baseline |
| Expert nested scale | `expert_nested_scale` | Identity/log1p selected inside training folds |
| Descriptor and scale search | `route_scale` | Search sensitivity, not the primary system |
| Full search | `full_automl` | Descriptor/scale/capacity sensitivity |
| Fixed and selected capacity | other `expert_rule_*` choices | Capacity controls with the rule scale |

`python reproduce/workflow.py --help` lists the accepted stage names.
Controls use the same source-native outer partitions and five seeds. The
rule scale is fitted only to training labels. Do not select a primary recipe
using outer-test results or present full-data refits as validation models.

After completing the policies, the existing numerical scripts are:

```bash
python scripts/analyze_expert_rule_scale.py
python scripts/analyze_expert_rule_capacity.py
python scripts/analyze_expert_rule_paper.py --out-dir rerun/primary_statistics
```

After the primary full-data refit, native inference files can be generated with:

```bash
python reproduce/download_models.py
python reproduce/export_refit.py --run-dir outputs/expert_rule_scale_v1/expert_rule_scale --output rerun/native_models
```

The downloaded catalogue supplies the endpoint contracts. New weights are
generated only from your own trusted refits, checked against their training
manifests, then exported without Python pickle. The archive's full-data
weights remain separate from newly refitted weights and outer-test evidence.

The primary script also consumes the historical stage table. That table's
provenance is `build_integrated_auto_system_report._stage_metrics`: OligoGym
source-XGBoost/project-XGBoost predictions, Atlas author/project-XGBoost
predictions, and `benchmark_registry_nine_endpoints.py` representation stages.
The frozen intermediate table is included so its controls remain auditable.
These sequential historical differences are not additive shares of the
primary gain and, for Atlas ALT, the representation contrast also changes
scale. They must not be described as pure isolated descriptor effects.

## Attribution and chemistry evidence

After a primary refit, the corresponding outer-model paths in its newly
written manifests are local and valid. Do not point the exporters at the
full-data models. On a resource-checked host, run the SHAP bounded probe
under `scripts/run_with_resource_guard.py` before its full export:

```bash
python scripts/export_feature_automl_explanations.py --run-dir outputs/expert_rule_scale_v1/expert_rule_scale --global-only --probe
python scripts/export_feature_automl_explanations.py --run-dir outputs/expert_rule_scale_v1/expert_rule_scale --global-only
python scripts/run_nested_automl_family_ablation.py --run-dir outputs/expert_rule_scale_v1/expert_rule_scale --device cuda --seed-count 5
python scripts/build_tlr8_automl_case.py --run-dir outputs/expert_rule_scale_v1/expert_rule_scale
python scripts/analyze_endpoint_interpretability_landscape.py --out-dir rerun/endpoint_interpretability
```

The commands above identify the scientific entry points; formal jobs must
be wrapped by the guard, just as training is. Family deletion refits models
and is not a cheap plotting operation. `export_reader_case_evidence_v11.py`
reconstructs the HeLa comparison from partition 3, with membership and
canonical-sequence exclusion assertions. Its `--paper-dir` input must contain
the archived chemistry-rich case manifest in `supporting/`.

The interpretation chain is descriptor definition → signed/absolute local
contribution → training-set correlation cluster → molecular concept →
literature-supported hypothesis and counterevidence. The source-reviewed
mechanism ledger and narrative audit are frozen research records, not newly
verified by rerunning a string template. `build_paper_interpretability_agent.py`
records deterministic evidence assembly; it is not an independent human
expert evaluation or proof of autonomous mechanism discovery.

The manuscript figures and reporting mappings are calculated from
`results/supporting/`. Historical figure source scripts are included with
their original version suffixes. They document provenance, not a promise
that an old caption or old figure label is the current manuscript wording.
Current figure rendering will be recorded separately in the release
verification manifest.
