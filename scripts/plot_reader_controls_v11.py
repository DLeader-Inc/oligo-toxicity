#!/usr/bin/env python3
"""Reporting-only V11 figures; retain measured historical contrasts and expose their limits."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'docs/papers/oligo_toxicity_journal_v11_reader_revised'
FIG = PAPER / 'figures'
ORDER = ['cytotoxicity','neurotoxicity_lna','neurotoxicity_moe','tlr7_immunotox','tlr8_immunotox','mouse_hepatic','rat_hepatic','mouse_neuro','rat_neuro']
LABELS = {
 'en':['HeLa caspase','LNA calcium','MOE mouse FOB','TLR8 potentiation','TLR7 inhibition','Mouse ALT','Rat ALT','Mouse FOB','Rat FOB'],
 'zh':['HeLa 半胱天冬酶','LNA 钙振荡','MOE 小鼠 FOB','TLR8 增强','TLR7 抑制','小鼠 ALT','大鼠 ALT','小鼠 FOB','大鼠 FOB']}

def style(lang):
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family': next((n for n in ['Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei','DejaVu Sans'] if n in names), 'DejaVu Sans'), 'font.size':11, 'axes.unicode_minus':False, 'svg.fonttype':'none'})

def save(fig, name, lang):
    stem = name + ('_zh' if lang == 'zh' else '')
    for ext in ['png','svg']:
        fig.savefig(FIG / f'{stem}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def controls(lang):
    d = pd.read_csv(PAPER/'supporting/endpoint_performance_component_controls.csv').set_index('dataset').loc[ORDER]
    z = lang == 'zh'; x=np.arange(9); w=.24
    fig, ax = plt.subplots(2,1,figsize=(12.5,11),gridspec_kw={'height_ratios':[1,1.12]})
    fig.subplots_adjust(hspace=.72, bottom=.17, top=.94)
    ax[0].bar(x-w/2,d.source_spearman,w,label='论文最佳配方本地复跑' if z else 'Paper-best local replay',color='#b65f45')
    ax[0].bar(x+w/2,d.final_spearman,w,label='完整专家系统' if z else 'Complete expert system',color='#177c8f')
    ax[0].set_title('A. 相同测试行：最终性能比较' if z else 'A. Final performance on the same test rows',loc='left')
    ax[0].set_ylabel('Spearman ρ'); ax[0].set_ylim(0,.87); ax[0].legend(loc='upper left',ncol=2,fontsize=10)
    labels = ['XGBoost 配方','历史表示／融合（ALT 同时改尺度）','仅改变尺度'] if z else ['XGB recipe','Historical representation/fusion (ALT also scale)','Scale-only']
    for offset,col,label,color in zip([-w,0,w],['model_recipe','expert_descriptor_fusion','rule_log1p'],labels,['#536d86','#177c8f','#d38a42']):
        ax[1].bar(x+offset,d[col],w,label=label,color=color)
    ax[1].set_title('B. 历史配方差值与单独尺度对照：不是提升来源的分解' if z else 'B. Historical recipe differences and scale-only control: not a gain decomposition',loc='left')
    ax[1].set_ylabel('Spearman 差值' if z else 'ΔSpearman'); ax[1].axhline(0,color='#444',lw=.8)
    ax[1].legend(loc='upper left',bbox_to_anchor=(0,1.01),fontsize=9); ax[1].set_ylim(-.01,.29)
    for a in ax:
        a.set_xticks(x,LABELS[lang],rotation=30,ha='right'); a.grid(axis='y',alpha=.2); a.set_axisbelow(True)
    note = ('B 图各项起点不同，不能相加得到 A 图的总提升。XGBoost 配方对照不是 CNN／Transformer 替换实验。\n历史表示／融合对照的种子约定不同；两个 ALT 终点还同时由原尺度改为 log1p，不能视为纯描述符增益。' if z else 'Panel B uses different starting points and is not additive to panel A. XGB recipe is not a CNN/Transformer replacement.\nHistorical representation/fusion uses different seed conventions; both ALT endpoints also change identity to log1p, so this is not a pure descriptor gain.')
    fig.text(.08,.035,note,fontsize=10,va='bottom',linespacing=1.7)
    save(fig,'endpoint_performance_and_controls',lang)

def architecture(lang):
    z=lang=='zh'; fig,ax=plt.subplots(figsize=(12.5,7.5));ax.set_xlim(0,12);ax.set_ylim(0,7);ax.axis('off')
    boxes = [
      (0.2,4.6,3.5,1.5,'输入：可溯源记录\n序列与修饰、测定条件\n模型预测、归因、实测近邻\n作者审阅的论文证据' if z else 'INPUT: TRACEABLE RECORDS\nSequence, chemistry and assay context\nPredictions, attributions, measured neighbours\nAuthor-reviewed literature evidence'),
      (4.25,4.6,3.5,1.5,'确定性证据汇集\n连接记录、核对成员关系\n汇总相关描述符与数值加和\n保存输入哈希及证据包' if z else 'DETERMINISTIC ASSEMBLY\nJoin records; check model membership\nSummarize related features and accounting\nSave input hashes and evidence packet'),
      (8.3,4.6,3.5,1.5,'预写双语模板\n把已核对字段填入解释\n执行十项文本内容检查\n保存叙述和检查记录' if z else 'PREWRITTEN BILINGUAL TEMPLATE\nInsert evidence fields into explanation\nRun ten text-content checks\nSave narrative and check records'),
      (4.25,1.65,3.5,1.5,'作者科学审阅\n解释描述符对应的分子变化\n区分模型依赖与实验事实\n核查文献、局限和替代解释' if z else 'AUTHOR SCIENTIFIC REVIEW\nExplain the molecular feature changes\nSeparate model dependence from observation\nCheck sources, limits and alternatives'),
      (8.3,1.65,3.5,1.5,'论文中的解释\n具体模型线索与证据边界\n可证伪的实验问题\n不是已验证的因果机制' if z else 'MANUSCRIPT INTERPRETATION\nSpecific model clues and evidence limits\nFalsifiable experimental questions\nNot a validated causal mechanism')]
    for x,y,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',fc='#edf4f5',ec='#177c8f',lw=1.2));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10.5,linespacing=1.65)
    for start,end in [((3.75,5.35),(4.2,5.35)),((7.8,5.35),(8.25,5.35)),((10.05,4.5),(6,3.2)),((7.8,2.4),(8.25,2.4))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.7,'color':'#536d86'})
    ax.set_title('本文案例实际执行的证据流程' if z else 'Evidence workflow actually executed for the archived case',fontsize=16,pad=14)
    ax.text(6,.45,'作者审阅属于科学解释工作，不是自动运行轨迹中的推理步骤。\n文本检查通过不等于独立语义验证；本案例未展示自主多智能体推理或有效自动修复。' if z else 'Author review is scientific interpretation, not a reasoning step demonstrated by the automated trace.\nText-check PASS is not independent semantic validation; this case does not demonstrate autonomous multi-agent reasoning or effective automatic repair.',ha='center',fontsize=10,linespacing=1.7)
    save(fig,'evidence_agent_architecture',lang)

if __name__=='__main__':
    FIG.mkdir(exist_ok=True)
    for language in ['en','zh']:
        style(language);controls(language);architecture(language)
