#!/usr/bin/env python3
"""Reporting-only V12 figures from archived paired metrics and case records."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'docs/papers/oligo_toxicity_journal_v12_literature_focused'
ORDER = ['cytotoxicity','neurotoxicity_lna','neurotoxicity_moe','tlr7_immunotox','tlr8_immunotox','mouse_hepatic','rat_hepatic','mouse_neuro','rat_neuro']
LABELS = {'en':['HeLa caspase','LNA calcium','MOE mouse FOB','TLR8 potentiation','TLR7 inhibition','Mouse ALT','Rat ALT','Mouse FOB','Rat FOB'], 'zh':['HeLa 半胱天冬酶','LNA 钙振荡','MOE 小鼠 FOB','TLR8 增强','TLR7 抑制','小鼠 ALT','大鼠 ALT','小鼠 FOB','大鼠 FOB']}

def save(fig, name, lang):
    stem = PAPER/'figures'/(name + ('_zh' if lang == 'zh' else ''))
    for ext in ('png','svg','pdf'):
        fig.savefig(str(stem)+'.'+ext, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def ranking(lang):
    z=lang=='zh'
    d=pd.read_csv(PAPER/'supporting/endpoint_performance_component_controls.csv').set_index('dataset').loc[ORDER]
    fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained'); x=np.arange(9); w=.34
    ax.bar(x-w/2,d.source_spearman,w,color='#b65f45',label='论文最佳配方本地复跑' if z else 'Paper-best local replay')
    ax.bar(x+w/2,d.final_spearman,w,color='#177c8f',label='完整专家系统' if z else 'Complete expert system')
    ax.set_xticks(x,LABELS[lang],rotation=30,ha='right'); ax.set_ylim(0,.91)
    ax.set_ylabel('Spearman ρ'); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    ax.legend(loc='upper left',ncol=2,fontsize=10)
    for i,r in enumerate(d.itertuples()):
        ax.text(i,r.final_spearman+.018,f'+{r.delta_spearman:.3f}',ha='center',fontsize=9)
    save(fig,'endpoint_ranking_v12',lang)

def cases(lang):
    z=lang=='zh'
    local=pd.read_csv(PAPER/'supporting/hela_heldout_feature_contrasts_v11.csv.gz').groupby('feature').difference.mean()
    pred=pd.read_csv(PAPER/'supporting/hela_heldout_reconciliation_v11.csv').groupby('row_index').prediction.mean()
    fig=plt.figure(figsize=(11.5,9.2),layout='constrained')
    gs=fig.add_gridspec(3,2,height_ratios=[1.5,1.7,1.3])
    a=fig.add_subplot(gs[0,:]); a.set_xlim(-2.8,20.6); a.set_ylim(-.9,2.6); a.axis('off')
    a.set_title('A. HeLa：碱基不变，只改变第 12 位糖' if z else 'A. HeLa: identical bases, one sugar edit at residue 12',loc='left',fontsize=13)
    for y,pattern,label in [(1.2,'LDDLDDDDDDLLL','710'),(0,'LDDLDDDDDDLDL','718')]:
        a.text(-.8,y+.35,label,ha='right',va='center')
        for i,(b,s) in enumerate(zip('TACTGCCTTCTTA',pattern),1):
            a.add_patch(Rectangle((i-1,y),.83,.78,fc='#cee5e3' if s=='L' else '#f0e6d5',ec='#b65f45' if i==12 else '#ffffff',lw=2))
            a.text(i-.58,y+.5,b,ha='center',fontsize=11);a.text(i-.58,y+.15,s,ha='center',fontsize=9)
            if y==1.2:a.text(i-.58,y+1.03,str(i),ha='center',fontsize=8)
    a.text(14,.65,'L = LNA; D = DNA\n' + ('两者均为全 PS 骨架\n末端 LLL → LDL' if z else 'All-PS backbone in both\nTerminal LLL → LDL'),va='center',linespacing=1.8)
    b=fig.add_subplot(gs[1,0]);x=np.arange(2);w=.3
    for shift,vals,color,label in [(-w/2,[.256,.078],'#b65f45','实测' if z else 'Measured'),(w/2,pred.loc[[710,718]].values,'#177c8f','留出预测' if z else 'Held-out prediction')]:
        bars=b.bar(x+shift,vals,w,color=color,label=label);b.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    b.set_xticks(x,['710','718']);b.set_ylim(0,.37);b.legend(fontsize=9)
    b.set_ylabel('归一化 caspase 响应' if z else 'Normalized caspase response')
    b.set_title('B. 同一个留出模型比较两条分子' if z else 'B. Both molecules predicted by the same held-out model',loc='left',fontsize=10)
    c=fig.add_subplot(gs[1,1]); labs=['相邻 LNA 对','LNA 比例','最长修饰连续段','未改变的位置 3 C'] if z else ['Adjacent LNA pairs','LNA fraction','Longest modified run','Unchanged position-3 C']
    vals=local.loc[['pint_mod_transition_L_L','ucore__frac_LNA','ucore__max_mod_run','uohe__raw_pos_03_base_C']].values
    c.barh(np.arange(4),vals,color='#177c8f');c.invert_yaxis();c.set_yticks(np.arange(4),labs,fontsize=9);c.set_xlim(0,.07)
    c.set_title('C. 预测差值的部分归因：710 − 718' if z else 'C. Selected contributions to predicted difference: 710 − 718',loc='left',fontsize=10)
    c.set_xlabel('同模型贡献差值' if z else 'Same-model contribution difference',fontsize=10)
    d=fig.add_subplot(gs[2,:]);d.axis('off');d.set_xlim(-2.8,20.6);d.set_ylim(-1.1,2.2)
    d.set_title('D. TLR8：化学设计固定，研究区域序列' if z else 'D. TLR8: fixed chemistry, a question about regional sequence',loc='left',fontsize=13)
    seq='GTCCGTACCTCCACCCACCG'
    for i,base in enumerate(seq,1):
        d.add_patch(Rectangle((i-1,.4),.83,.65,fc='#cee5e3' if i<=5 or i>15 else '#f0e6d5',ec='#b65f45' if i==1 or 9<=i<=12 else 'white',lw=1.5));d.text(i-.58,.72,base,ha='center',va='center',fontsize=11)
        d.text(i-.58,1.25,str(i),ha='center',fontsize=8)
    for center,label in [(2.2,'OMe 5′ 翼区' if z else 'OMe 5′ wing'),(9.7,'DNA 缺口' if z else 'DNA gap'),(17.2,'OMe 3′ 翼区' if z else 'OMe 3′ wing')]:d.text(center,.05,label,ha='center',fontsize=10)
    d.text(9,-.63,'首位 G 与位置 9–12 的 CTCC 是模型线索；不是已测得的替换效应。' if z else 'First G and CTCC at positions 9–12 are model clues, not measured replacement effects.',ha='center',fontsize=10)
    save(fig,'case_molecular_questions_v12',lang)

if __name__=='__main__':
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none'})
    for lang in ('en','zh'):
        ranking(lang);cases(lang)
