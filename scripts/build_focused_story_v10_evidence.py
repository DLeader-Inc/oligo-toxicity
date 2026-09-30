"""Reclassify existing feature attribution by implemented molecular meaning."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_endpoint_interpretability_landscape import ORDER, LABEL_EN, LABEL_ZH, style

AXES = ['sequence', 'sugar', 'backbone', 'modified_base', 'architecture', 'combined', 'assay']
EN = ['Base patterns\nand composition', 'Sugar type\nand placement', 'Backbone\nand placement', 'Modified\nbases', 'Length and\nwing–gap architecture', 'Combined sequence,\nchemistry and region', 'Recorded\nassay inputs']
ZH = ['碱基模式\n与组成', '糖类型\n与位置', '骨架类型\n与位置', '修饰\n碱基', '长度与\n翼区–缺口结构', '序列、化学\n与区域的组合', '已记录的\n测定输入']


def molecular_axis(name: str) -> str:
    """One reporting axis per column; explicit combinations precede constituents."""
    s = name.lower()
    if s.startswith('assay__'):
        return 'assay'
    if any(x in s for x in ['auxiliary_', 'overflow_', 'canvas_truncated']):
        return 'encoding_context'
    if 'symbolic_' in s:
        return 'sugar'
    if s == 'scint_gap_pos2_or_3_neutral':
        return 'backbone'
    if s.endswith('_exists') or 'global_length_' in s or s in ['ucore__seq_len', 'scint_seq_len']:
        return 'architecture'
    if 'gc_fraction' in s:
        return 'sequence'
    if s.startswith(('pint_base_', 'pint_motif_')):
        return 'combined'
    if s.startswith('scint_motif_'):
        return 'sequence' if re.fullmatch(r'scint_motif_[acgt]+_count', s) else 'combined'
    if s.startswith('scint_tgt_gap_'):
        return 'combined'
    if any(x in s for x in ['m5c', '5mc', '5mec', 'basechem_']):
        return 'modified_base'
    if any(x in s for x in ['link', 'phosphate', 'ps_fraction', 'mspa', 'mop']) or re.search(r'_(ps|po|mp)(_|$)', s):
        return 'backbone'
    if any(x in s for x in ['sugar', 'mod_pos_', 'mod_transition', 'max_mod_run', 'mod_3prime', 'mod_5prime', 'modified_in_gap', 'bna_wing']):
        return 'sugar'
    if re.search(r'(?:n_|frac_)(?:lna|moe|ome|dna|rna|cet)', s):
        return 'sugar'
    if s.startswith('scint_') and re.search(r'_(?:d|l|m|o|r|e)_(?:count|fraction)$', s):
        return 'sugar'
    if any(x in s for x in ['gap_length', 'gap_fraction', 'wing5_len', 'wing3_len', 'gap_01_exists', 'gap_02_exists', 'gap_03_exists', 'has_explicit_dna_gap']):
        return 'architecture'
    if s.startswith('ucore__') and any(x in s for x in ['kmer_', 'n_', 'has_', 'gc_content', 'seq_len']):
        return 'sequence'
    if s == 'scint_seq_len' or '_base_' in s:
        # Regional bases occur in a chemistry-defined region.
        return 'combined' if s.startswith('ucontext__') and any(x in s for x in ['gap', 'wing', 'pool_other']) else 'sequence'
    return 'unmapped'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper-dir', type=Path, required=True)
    args = parser.parse_args()
    support = args.paper_dir / 'supporting'
    feature = pd.read_csv(support / 'global_heldout_feature_ranking.csv')
    provenance = pd.read_csv(support / 'descriptor_provenance_database.csv')
    mapping = pd.DataFrame({'feature': sorted(feature.feature.unique())})
    mapping['molecular_axis_v10'] = mapping.feature.map(molecular_axis)
    mapping['positional_or_regional'] = mapping.feature.str.contains(r'pos_|region|wing|gap|anchor|pool_|transition')
    mapping = mapping.merge(provenance[['feature', 'implementation_file']], on='feature', how='left', validate='one_to_one')
    mapping.loc[mapping.feature.str.startswith('assay__'), 'implementation_file'] = 'scripts/benchmark_feature_automl_nested.py'
    mapping.to_csv(support / 'molecular_descriptor_mapping_v10.csv', index=False)
    missing = mapping[mapping.molecular_axis_v10.eq('unmapped')]
    if len(missing):
        raise ValueError('Unmapped descriptors: ' + ', '.join(missing.feature))
    joined = feature.merge(mapping, on='feature', validate='many_to_one')
    summary = joined.groupby(['dataset', 'model_scale', 'molecular_axis_v10'], as_index=False).agg(
        mean_absolute_feature_mass=('mean_absolute_contribution', 'sum'),
        mean_signed_contribution=('mean_signed_contribution', 'sum'),
        features=('feature', 'size'))
    summary['normalized_absolute_mass'] = summary.mean_absolute_feature_mass / summary.groupby(['dataset', 'model_scale']).mean_absolute_feature_mass.transform('sum')
    summary.to_csv(support / 'endpoint_molecular_information_v10.csv', index=False)
    assert np.allclose(summary.groupby(['dataset', 'model_scale']).normalized_absolute_mass.sum(), 1)
    assert feature.groupby('dataset').model_scale.nunique().eq(1).all(), 'Plot needs separate rows for mixed scales'
    old = pd.read_csv(support / 'mechanism_axis_v2_manifest.csv')
    mapping.merge(old[['feature', 'mechanism_axis_v2']], on='feature', how='left').to_csv(support / 'molecular_mapping_change_audit_v10.csv', index=False)
    deletion = pd.read_csv(support / 'model_family_deletion_by_endpoint.csv')
    grid = summary.pivot(index='dataset', columns='molecular_axis_v10', values='normalized_absolute_mass').reindex(index=ORDER, columns=AXES).fillna(0) * 100
    hidden = summary[summary.molecular_axis_v10.eq('encoding_context')].normalized_absolute_mass
    assert hidden.eq(0).all(), 'Nonzero encoding context must be shown as an extra figure category'
    loss = deletion.pivot(index='dataset', columns='ablation_family', values='spearman_loss_when_removed').reindex(index=ORDER, columns=['registry', 'motif', 'chemical_modification'])
    style()
    for zh in (False, True):
        labels = LABEL_ZH if zh else LABEL_EN
        row_labels = [labels[x] + (' †' if 'hepatic' in x else '') for x in ORDER]
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 7.4), gridspec_kw={'width_ratios': [7, 3], 'wspace': .15})
        im = ax[0].imshow(grid.to_numpy(), cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
        ax[0].set_yticks(range(9), row_labels)
        ax[0].set_xticks(range(7), ZH if zh else EN, rotation=35, ha='right')
        ax[0].set_title('A  分子信息贡献（%）' if zh else 'A  Molecular attribution (%)', fontsize=12)
        for i in range(9):
            for j in range(7):
                v = grid.iloc[i, j]
                ax[0].text(j, i, f'{v:.1f}', ha='center', va='center', color='white' if v > 50 else '#172b35', fontsize=10)
        limit = max(abs(loss.to_numpy()).max(), .01)
        ax[1].imshow(loss.to_numpy(), cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        ax[1].set_yticks(range(9), [''] * 9)
        ax[1].set_xticks(range(3), ['原始位置\n编码', '序列模式\n信息块', '显式化学\n信息块'] if zh else ['Raw positional\nblock', 'Sequence-pattern\nblock', 'Explicit chemistry\nblock'], rotation=35, ha='right')
        ax[1].set_title('B  删除后的排序损失' if zh else 'B  Spearman loss\nafter removal', fontsize=11)
        for i in range(9):
            for j in range(3):
                v = loss.iloc[i, j]
                ax[1].text(j, i, f'{v:+.3f}', ha='center', va='center', color='white' if abs(v) > limit*.6 else '#172b35', fontsize=9)
        note = ('每行贡献合计为 100%；† ALT 在 log1p 模型尺度上解释，其余为原尺度。\n糖与骨架固定的 TLR 数据仍有序列–区域组合变化；贡献不等于化学替换效应。B 正值表示删除后变差。' if zh else
                'Attribution sums to 100% within each row; † ALT uses log1p, others raw scale.\nTLR chemistry is fixed: combined features locate bases in fixed regions.\nPositive B values indicate worse ranking after removal.')
        fig.text(.12, .015, note, fontsize=9)
        fig.subplots_adjust(bottom=.29, top=.9)
        fig.savefig(args.paper_dir / 'figures' / ('endpoint_molecular_information_zh.png' if zh else 'endpoint_molecular_information.png'), bbox_inches='tight')
        plt.close(fig)
    manifest = {
        'mapping_version': 'v10_definition_audited', 'features': len(mapping), 'unmapped': 0,
        'aggregation': 'sum of mean absolute feature contributions; not absolute value of a signed category sum',
        'normalization': 'within endpoint and fitted model scale; chemistry, positional and sequence annotations not double counted',
        'feature_ranking_sha256': hashlib.sha256((support/'global_heldout_feature_ranking.csv').read_bytes()).hexdigest(),
        'assay_columns': mapping[mapping.molecular_axis_v10.eq('assay')].feature.tolist(),
        'encoding_context': 'Auxiliary/overflow/canvas flags retained in CSV; zero attribution in all nine endpoints, omitted from figure',
        'no_new_training': True,
    }
    (support/'molecular_information_manifest_v10.json').write_text(json.dumps(manifest, indent=2)+'\n')
    oracle = json.loads((support/'full_data_oracle_case_explanations.json').read_text())['cytotoxicity']
    left, right = [pd.DataFrame(x['descriptors']).set_index('feature') for x in oracle]
    common = left.index.intersection(right.index)
    contrast = pd.DataFrame({'feature': common})
    for side, frame in [('left', left), ('right', right)]:
        contrast[side+'_value'] = frame.loc[common, 'feature_value'].to_numpy()
        contrast[side+'_signed_contribution'] = frame.loc[common, 'mean_signed_contribution'].to_numpy()
    contrast['signed_contribution_difference'] = contrast.left_signed_contribution - contrast.right_signed_contribution
    contrast['molecular_axis_v10'] = contrast.feature.map(molecular_axis)
    contrast['scope'] = 'intersection of exported top-100 descriptors for each molecule; not a complete prediction decomposition'
    contrast.sort_values('signed_contribution_difference', ascending=False).to_csv(support/'hela_case_common_feature_contrasts_v10.csv', index=False)
    print(grid.round(2).to_string())


if __name__ == '__main__':
    main()
