"""Bounded two-row inference from archived models; no fitting or selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_registry_nine_endpoints import earlier_frame
from src.features.automl_descriptor_routes import build_route_matrices

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-dir', type=Path, required=True)
    ap.add_argument('--probe', action='store_true')
    args = ap.parse_args()
    out = args.paper_dir / 'supporting'
    run = ROOT / 'outputs/expert_rule_scale_v1/expert_rule_scale'
    selection_path = run / 'fold_checkpoints/cytotoxicity__partition3__selection.json'
    selection = json.loads(selection_path.read_text())
    assert selection['selected_transform'] == 'identity'
    membership = json.loads(Path(selection['membership_path']).read_text())
    case = json.loads((out/'chemistry_rich_case_manifest.json').read_text())['cytotoxicity']
    frame = earlier_frame('cytotoxicity').iloc[[710, 718]].copy()
    for row, side in zip([710, 718], ['left', 'right']):
        rec = case[side]
        assert row not in membership['train_row_indices']
        assert hashlib.sha256(rec['sequence'].encode()).hexdigest() not in membership['train_sequence_sha256']
        assert hashlib.sha256(frame.loc[row, 'x'].encode()).hexdigest() not in membership['train_helm_sha256']
    matrices, schemas, _ = build_route_matrices(frame.reset_index(drop=True))
    details, reconciliation, model_hashes = [], [], []
    members = [1] if args.probe else [1, 2, 3, 4, 5]
    for member in members:
        combined, features = {}, {}
        baseline, prediction = np.zeros(2), np.zeros(2)
        for record in selection['models']:
            if int(record['member']) != member:
                continue
            component, weight = record['component'], float(record['weight'])
            model = joblib.load(record['path'])
            model.set_params(n_jobs=1, device='cpu')
            matrix = matrices[component]
            booster = model.get_booster()
            booster.set_param({'nthread': 1, 'device': 'cpu'})
            contributions = booster.predict(xgb.DMatrix(matrix), pred_contribs=True)
            prediction += weight * booster.predict(xgb.DMatrix(matrix))
            baseline += weight * contributions[:, -1]
            for index, name in enumerate(schemas[component]):
                combined[name] = combined.get(name, np.zeros(2)) + weight * contributions[:, index]
                features[name] = matrix[:, index]
            model_hashes.append({'path': record['path'], 'sha256': sha(record['path'])})
            del model, booster
        for index, row in enumerate([710, 718]):
            reconstructed = baseline[index] + sum(v[index] for v in combined.values())
            assert abs(prediction[index] - reconstructed) < 2e-6
            reconciliation.append(dict(member=member, row_index=row, prediction=prediction[index],
                baseline=baseline[index], reconstructed=reconstructed,
                residual=prediction[index]-reconstructed, model_scale='y', outer_partition=3))
        for name, values in combined.items():
            details.append(dict(member=member, feature=name, left_value=float(features[name][0]),
                right_value=float(features[name][1]), left_contribution=values[0],
                right_contribution=values[1], difference=values[0]-values[1]))
    recon = pd.DataFrame(reconciliation)
    result = {'probe': args.probe, 'new_training': False, 'members': len(members),
              'prediction': recon.groupby('row_index').prediction.mean().to_dict(),
              'maximum_reconstruction_residual': float(recon.residual.abs().max()),
              'selection_sha256': sha(selection_path), 'models': model_hashes}
    if not args.probe:
        expected = pd.read_csv(run/'fold_checkpoints/cytotoxicity__partition3__predictions.csv')
        prediction_column = 'pred_policy' if 'pred_policy' in expected else 'pred_automl_v2'
        for row, value in result['prediction'].items():
            assert np.isclose(value, expected.set_index('row_index').loc[row, prediction_column], atol=2e-6)
        pd.DataFrame(details).to_csv(out/'hela_heldout_feature_contrasts_v11.csv.gz', index=False)
        recon.to_csv(out/'hela_heldout_reconciliation_v11.csv', index=False)
        summary = pd.DataFrame(details).groupby('feature').agg(
            left_value=('left_value', 'first'), right_value=('right_value', 'first'),
            mean_difference=('difference', 'mean'), positive_fraction=('difference', lambda x: float((x>0).mean())),
            negative_fraction=('difference', lambda x: float((x<0).mean())))
        summary['absolute_mean_difference'] = summary.mean_difference.abs()
        summary.sort_values('absolute_mean_difference', ascending=False).to_csv(out/'hela_heldout_feature_summary_v11.csv')
        local = pd.read_csv(run/'tlr8_case/case_feature_contributions.csv.gz')
        assert not local.duplicated(['outer_partition','member','feature']).any()
        local.to_csv(out/'tlr8_local_complete_contributions_v11.csv.gz', index=False)
        tlr_summary = local.groupby('feature').agg(value=('feature_value','first'),
            mean_contribution=('signed_contribution','mean'),
            negative_fraction=('signed_contribution',lambda x:float((x<0).mean())))
        tlr_summary['absolute_mean_contribution'] = tlr_summary.mean_contribution.abs()
        tlr_summary.sort_values('absolute_mean_contribution',ascending=False).to_csv(out/'tlr8_local_feature_summary_v11.csv')
        (out/'reader_case_export_manifest_v11.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
