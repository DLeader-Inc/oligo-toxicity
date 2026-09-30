#!/usr/bin/env python3
"""Page-width figures: 6.2-inch canvases, minimum 8-point labels, audited boxes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text
from matplotlib.legend import Legend
import numpy as np
import pandas as pd
from plot_endpoint_interpretability_landscape import ORDER, LABEL_EN, CASE_ORDER
from build_focused_story_v10_evidence import AXES

ROOT=Path(__file__).resolve().parents[1]
TEAL,RUST='#177c8f','#b65f45'
SHORT=['HeLa caspase','LNA calcium','MOE mouse FOB','TLR8 potentiation','TLR7 inhibition','Mouse ALT','Rat ALT','Mouse FOB','Rat FOB']
AUDIT=[]

def finish(fig,out,name,boxes=()):
    fig.canvas.draw();r=fig.canvas.get_renderer();bounds=fig.bbox
    # Matplotlib retains tick-label objects outside the view interval but does
    # not render them. Exclude these inactive objects from containment checks.
    for ax in fig.axes:
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            lo,hi=sorted(limits)
            for tick in axis.get_major_ticks():
                if not lo-1e-9<=tick.get_loc()<=hi+1e-9:
                    tick.label1.set_visible(False);tick.label2.set_visible(False)
    texts=[x for x in fig.findobj(Text) if x.get_visible() and x.get_text().strip()]
    for t in texts:
        assert t.get_fontsize()>=8,(name,t.get_text(),t.get_fontsize())
        b=t.get_window_extent(r)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=bounds.x1+1 and b.y1<=bounds.y1+1,(name,'canvas clipping',t.get_text(),tuple(b.extents))
    for patch,t in boxes:
        p=patch.get_window_extent(r);b=t.get_window_extent(r)
        assert p.contains(b.x0,b.y0) and p.contains(b.x1,b.y1),(name,'box overflow',t.get_text())
    for legend in fig.findobj(Legend):
        b=legend.get_window_extent(r)
        assert b.x0>=0 and b.y0>=0 and b.x1<=bounds.x1 and b.y1<=bounds.y1,(name,'legend clipping')
    for ext in ('png','pdf','svg'):fig.savefig(out/(name+'.'+ext),dpi=240,facecolor='white')
    AUDIT.append({'figure':name,'width_inches':6.2,'height_inches':float(fig.get_figheight()),'minimum_source_font_pt':min(t.get_fontsize() for t in texts),'minimum_at_6inch_pt':min(t.get_fontsize() for t in texts)*6/6.2,'diagram_boxes_checked':len(boxes),'canvas_containment':'PASS'})
    plt.close(fig)

def panel(ax,letter):ax.text(0,1.045,letter,transform=ax.transAxes,fontsize=10,fontweight='bold')

def diagram(out,name,labels,kind):
    fig,ax=plt.subplots(figsize=(6.2,5.6));fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.02);ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off');boxes=[]
    if kind=='system':
        positions=[(2,8.7,6,1),(0.1,5.7,4.5,2),(5.4,5.7,4.5,2),(2,3.65,6,1.1),(1.8,.4,6.4,1.85)]
        edges=[((3.7,8.6),(2.3,7.8)),((6.3,8.6),(7.7,7.8)),((2.4,4.85),(2.3,5.6)),((7.6,4.85),(7.7,5.6)),((.9,5.6),(2,2.35)),((9.1,5.6),(8,2.35))]
    else:
        positions=[(2,8.7,6,1),(2,6.7,6,1),(2,4.7,6,1),(2,2.7,6,1),(2,.7,6,1)]
        edges=[((5,y),(5,y-1)) for y in (8.6,6.6,4.6,2.6)]
    for (x,y,w,h),s in zip(positions,labels):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor='#edf4f5',edgecolor=TEAL);ax.add_patch(p)
        t=ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=9,linespacing=1.45);boxes.append((p,t))
    for a,b in edges:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.3,'color':'#536d86'})
    finish(fig,out,name,boxes)

def ranking(data,out):
    d=pd.read_csv(data/'endpoint_performance_component_controls.csv').set_index('dataset').loc[ORDER]
    fig,ax=plt.subplots(figsize=(6.2,4.7));fig.subplots_adjust(left=.25,right=.96,bottom=.12,top=.9)
    y=np.arange(9);ax.barh(y+.18,d.source_spearman,.34,color=RUST,label='Paper-best local replay');ax.barh(y-.18,d.final_spearman,.34,color=TEAL,label='Integrated system')
    ax.set_yticks(y,SHORT);ax.invert_yaxis();ax.set_xlim(0,.9);ax.set_xlabel('Spearman ρ');ax.legend(loc='lower left',bbox_to_anchor=(-.3,1.01),ncol=2,fontsize=8)
    for i,row in enumerate(d.itertuples()):ax.text(row.final_spearman+.015,i-.18,f'+{row.delta_spearman:.3f}',va='center',fontsize=8)
    finish(fig,out,'endpoint_ranking_v15')

def molecular(data,out):
    d=pd.read_csv(data/'endpoint_molecular_information_v10.csv');grid=d.pivot(index='dataset',columns='molecular_axis_v10',values='normalized_absolute_mass').reindex(index=ORDER,columns=AXES).fillna(0)*100
    loss=pd.read_csv(data/'model_family_deletion_by_endpoint.csv').pivot(index='dataset',columns='ablation_family',values='spearman_loss_when_removed').reindex(index=ORDER,columns=['registry','motif','chemical_modification'])
    fig,axs=plt.subplots(2,1,figsize=(6.2,7.25));fig.subplots_adjust(left=.25,right=.975,top=.94,bottom=.06,hspace=.52)
    for j,(ax,frame) in enumerate(zip(axs,[grid,loss])):
        limit=max(abs(frame.to_numpy()).max(),.01)
        ax.imshow(frame,cmap='YlGnBu' if j==0 else 'RdBu_r',vmin=0 if j==0 else -limit,vmax=100 if j==0 else limit,aspect='auto')
        ax.set_yticks(range(9),[s+(' †' if 'hepatic' in k else '') for s,k in zip(SHORT,ORDER)])
        labels=['Sequence','Sugar','Backbone','Modified\nbases','Wing–gap\nlayout','Combined','Assay'] if j==0 else ['Positional encoding','Sequence patterns','Explicit chemistry']
        ax.set_xticks(range(len(labels)),labels);panel(ax,'AB'[j])
        for row in range(9):
            for col in range(frame.shape[1]):
                v=frame.iloc[row,col];white=v>50 if j==0 else abs(v)>limit*.6
                ax.text(col,row,f'{v:.1f}' if j==0 else f'{v:+.3f}',ha='center',va='center',fontsize=8,color='white' if white else '#172b35')
    finish(fig,out,'endpoint_molecular_information_v15')

def cases(data,out):
    local=pd.read_csv(data/'hela_heldout_feature_contrasts_v11.csv.gz').groupby('feature').difference.mean();pred=pd.read_csv(data/'hela_heldout_reconciliation_v11.csv').groupby('row_index').prediction.mean()
    fig=plt.figure(figsize=(6.2,7.1));gs=fig.add_gridspec(3,2,height_ratios=[1.35,1.5,1.05],left=.13,right=.97,top=.96,bottom=.05,hspace=.58,wspace=1.08)
    a=fig.add_subplot(gs[0,:]);a.set(xlim=(-2.1,15),ylim=(-.5,2.7));a.axis('off');panel(a,'A')
    for y,pat,row in [(1.2,'LDDLDDDDDDLLL','710'),(0,'LDDLDDDDDDLDL','718')]:
        a.text(-1.9,y+.4,row,va='center',fontsize=8);a.text(-.65,y+.4,'5′',va='center',fontsize=8);a.text(13.1,y+.4,'3′',va='center',fontsize=8)
        for i,(base,sugar) in enumerate(zip('TACTGCCTTCTTA',pat)):
            a.add_patch(Rectangle((i,y),.9,.84,fc='#cee5e3' if sugar=='L' else '#f0e6d5',ec=RUST if i==11 else 'white',lw=1.5))
            a.text(i+.45,y+.54,base,ha='center',va='center',fontsize=8);a.text(i+.45,y+.19,sugar,ha='center',va='center',fontsize=8)
            if y>0:a.text(i+.45,y+1.08,str(i+1),ha='center',fontsize=8)
    b=fig.add_subplot(gs[1,0]);x=np.arange(2)
    for shift,vals,color,label in [(-.17,[.256,.078],RUST,'Measured'),(.17,pred.loc[[710,718]].values,TEAL,'Held-out')]:
        bars=b.bar(x+shift,vals,.32,color=color,label=label);b.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
    b.set_xticks(x,['710','718']);b.set_ylim(0,.38);b.set_ylabel('Normalized caspase response');b.set_xlabel('Data row');b.legend(fontsize=8,loc='upper right',bbox_to_anchor=(1.1,1.2));panel(b,'B')
    c=fig.add_subplot(gs[1,1]);keys=['pint_mod_transition_L_L','ucore__frac_LNA','ucore__max_mod_run','uohe__raw_pos_03_base_C'];c.barh(range(4),local.loc[keys],color=TEAL);c.set_yticks(range(4),['Adjacent\nLNA pairs','LNA fraction','Longest\nmodified run','Position-3 C']);c.invert_yaxis();c.set_xlim(0,.07);c.set_xticks([0,.03,.06]);c.set_xlabel('Contribution difference\n(710 − 718)');panel(c,'C')
    d=fig.add_subplot(gs[2,:]);d.axis('off');d.set(xlim=(-1.2,21),ylim=(-.3,1.8));panel(d,'D')
    for i,base in enumerate('GTCCGTACCTCCACCCACCG'):
        d.add_patch(Rectangle((i,.4),.9,.65,fc='#cee5e3' if i<5 or i>=15 else '#f0e6d5',ec=RUST if i==0 or 8<=i<=11 else 'white',lw=1.2));d.text(i+.45,.73,base,ha='center',va='center',fontsize=8);d.text(i+.45,1.22,str(i+1),ha='center',fontsize=8)
    d.text(-1.05,.73,'5′',va='center');d.text(20.1,.73,'3′',va='center')
    for x,t in [(2.4,'OMe'),(9.9,'DNA'),(17.4,'OMe')]:d.text(x,.05,t,ha='center')
    finish(fig,out,'hela_heldout_reader_case_v15')

def landscape(data,out):
    summary=pd.read_csv(data/'dataset_direct_chemistry_summary.csv').set_index('dataset').loc[ORDER]
    deletion=pd.read_csv(data/'model_family_deletion_by_endpoint.csv');deletion=deletion[deletion.ablation_family.eq('chemical_modification')].set_index('dataset').loc[ORDER]
    recovery=pd.read_csv(data/'source_protocol_oof_pair_recovery_by_endpoint.csv').set_index('dataset')
    fig,axs=plt.subplots(3,1,figsize=(6.2,7.4));fig.subplots_adjust(left=.26,right=.95,top=.96,bottom=.065,hspace=.6)
    values=[summary.eligible_row_fraction*100,deletion.spearman_loss_when_removed,[recovery.loc[k,'group_level_direction_agreement']*100 if k in recovery.index else np.nan for k in ORDER]]
    for i,(ax,v) in enumerate(zip(axs,values)):
        ax.barh(range(9),np.nan_to_num(v),color=TEAL if i!=1 else [RUST if x>0 else '#94a3b8' for x in v]);ax.set_yticks(range(9),SHORT);ax.invert_yaxis();panel(ax,'ABC'[i]);ax.axvline(0,color='black',lw=.5)
        ax.set_xlabel(['Rows with a same-sequence chemical comparator (%)','Spearman loss after chemistry-block deletion','Direction recovery (%)'][i])
        if i!=1:
            ax.set_xlim(0,112)
            ax.set_xticks([0,25,50,75,100])
            for j,val in enumerate(v):ax.text(1 if not np.isfinite(val) else val+1,j,'No contrast' if not np.isfinite(val) else f'{val:.1f}',va='center',fontsize=8)
    finish(fig,out,'endpoint_interpretability_landscape_v15')

def matched(data,out):
    frame=pd.read_csv(data/'single_edit_context_dependence.csv');fig,axs=plt.subplots(4,1,figsize=(6.2,8.0));fig.subplots_adjust(left=.29,right=.89,top=.94,bottom=.055,hspace=.7)
    for j,(ax,k) in enumerate(zip(axs,CASE_ORDER)):
        b=frame[frame.dataset.eq(k)].sort_values('sequence_backgrounds',ascending=False).head(6).iloc[::-1];y=np.arange(len(b))
        ax.errorbar(b.observed_median_delta,y+.11,xerr=np.vstack([b.observed_median_delta-b.observed_q25,b.observed_q75-b.observed_median_delta]),fmt='o',color='#2563eb',markersize=4,capsize=2,label='Measured median and IQR')
        ax.errorbar(b.predicted_median_delta,y-.11,xerr=np.vstack([b.predicted_median_delta-b.predicted_min_delta,b.predicted_max_delta-b.predicted_median_delta]),fmt='s',color='#f97316',markersize=4,capsize=2,label='Model median and range')
        ax.set_yticks(y,[x.replace('sugar@','Sugar ').replace('linkage@','Linkage ').replace('2′-O-methyl','OMe') for x in b.single_edit_axis]);ax.axvline(0,color='black',lw=.6);ax.set_xlabel('Response difference (chemical contrast)');panel(ax,'ABCD'[j])
        for i,n in enumerate(b.sequence_backgrounds):ax.text(1.02,i,f'n={n}',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.997),fontsize=8,ncol=2)
    finish(fig,out,'matched_edit_context_cases_v15')

def joint(data,out):
    frame=pd.read_csv(data/'model_top_joint_descriptors.csv')
    for page in range(3):
        fig,axs=plt.subplots(3,1,figsize=(6.2,6.5));fig.subplots_adjust(left=.52,right=.97,top=.95,bottom=.07,hspace=.6)
        for j,(ax,k) in enumerate(zip(axs,ORDER[page*3:page*3+3])):
            b=frame[frame.dataset.eq(k)].sort_values('mean_absolute_contribution').tail(5);labels=b.feature.str.replace('pint_','',regex=False).str.replace('scint_','',regex=False)
            ax.barh(range(len(b)),b.normalized_absolute_mass*100,color='#7c3aed');ax.set_yticks(range(len(b)),labels);ax.tick_params(axis='y',labelsize=8);ax.set_xlabel('Absolute TreeSHAP mass (%)');panel(ax,chr(65+page*3+j))
        finish(fig,out,f'model_joint_descriptors_by_endpoint_v15_part{page+1}')

def controls(data,out):
    d=pd.read_csv(data/'endpoint_performance_component_controls.csv').set_index('dataset').loc[ORDER]
    fig,axs=plt.subplots(2,1,figsize=(6.2,7.5));fig.subplots_adjust(left=.25,right=.95,top=.91,bottom=.065,hspace=.48);y=np.arange(9)
    for shift,col,color,label in [(+.17,'source_spearman',RUST,'Paper-best local replay'),(-.17,'final_spearman',TEAL,'Integrated system')]:axs[0].barh(y+shift,d[col],.32,color=color,label=label)
    for shift,col,color,label in [(-.24,'model_recipe','#536d86','XGBoost recipe'),(0,'expert_descriptor_fusion',TEAL,'Representation/fusion'),(.24,'rule_log1p','#d38a42','Scale-only')]:axs[1].barh(y+shift,d[col],.23,color=color,label=label)
    for i,ax in enumerate(axs):ax.set_yticks(y,SHORT);ax.invert_yaxis();ax.set_xlabel('Spearman ρ' if i==0 else 'ΔSpearman');panel(ax,'AB'[i]);ax.legend(loc='lower left',bbox_to_anchor=(-.3,1.065),ncol=2 if i==0 else 3,fontsize=8)
    finish(fig,out,'endpoint_performance_and_controls_v15')

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'docs/papers/oligo_toxicity_author_revision_v14/supporting');p.add_argument('--out-dir',type=Path,default=ROOT/'docs/papers/oligo_toxicity_author_revision_v15/figures');args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.labelsize':8,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    diagram(args.out_dir,'system_workflow_v15',['Molecular structure and assay context','Branch A: positional features\nMonomer/linkage encoding\n+ local position interactions\nXGBoost','Branch B: engineered features\nMotifs, composition, sugars,\nbackbone, sequence × chemistry\nXGBoost','Training-label scale rule\nIdentity or log1p','Equal-weight branch average\nInverse transform to recorded scale\nAverage predictions over five seeds'],'system')
    diagram(args.out_dir,'evidence_workflow_v15',['Traceable inputs','Deterministic assembly','Explanation template','AI-assisted source review','Manuscript interpretation'],'evidence')
    diagram(args.out_dir,'evidence_agent_workflow_v15',['Individual descriptors','Training-set correlation groups','Molecular concepts','Information families','Testable mechanism hypothesis'],'concepts')
    for fn in (ranking,molecular,cases,landscape,matched,joint,controls):fn(args.data_dir,args.out_dir)
    (args.out_dir/'FIGURE_PAGE_SIZE_AUDIT.json').write_text(json.dumps(AUDIT,indent=2)+'\n')

if __name__=='__main__':main()
