"""Build manuscript artifacts using saved outputs only. Does not import torch."""
from supplement_tables import build_supplement_tables
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pilot
import json,time,hashlib,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import study_budget as budget
P=ROOT/'paper'; F=P/'figures'; D=P/'data'

def main():
    F.mkdir(exist_ok=True); D.mkdir(exist_ok=True)
    original=json.loads((ROOT/'results/full-study/evaluation.json').read_text())
    audit=json.loads((ROOT/'results/publication-task-audit/paired-task-audit.json').read_text())
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,n):
        n+=1  # Figure 1 is the authorized public clinical illustration.
        fig.savefig(F/f'Figure{n}.png',dpi=600,bbox_inches='tight'); fig.savefig(F/f'Figure{n}.pdf',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,4.4)); ax.set(xlim=(0,10),ylim=(0,7)); ax.axis('off')
    def box(x,y,w,h,text):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='white',edgecolor='.3',lw=.8)); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(x,y,u,v): ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='->',mutation_scale=10,color='.25'))
    box(.1,5.6,2.5,1.1,'480 phantoms\nAbsent / line / ring\nSplit before generation')
    box(3.1,5.6,3.0,1.1,'Coupled elastic scattering\n1 / 3 MHz; full / limited view\nReference SNR 20 / 40 dB')
    box(6.7,5.6,3.1,1.1,'Matched-filter score maps\n128 x 128, known background\n3,840 derived images')
    arrow(2.65,6.15,3.05,6.15); arrow(6.15,6.15,6.65,6.15)
    box(.2,3.25,2.7,1.15,'Direct US\n128 x 128\nOne-channel classifier')
    box(3.6,3.25,2.7,1.15,'Bicubic + unsharp\n256 x 256\nOne-channel classifier')
    box(7,3.25,2.7,1.15,'Learned SR map\n256 x 256\nUS + map classifier')
    for cx in (1.55,4.95,8.35): arrow(8.25,5.55,cx,4.45)
    box(7,1.6,2.7,.9,'Map localization\n150 um matching radius\nFalse candidates / image')
    arrow(8.35,3.2,8.35,2.55)
    box(.2,.3,6.1,1.3,'Classification endpoints\nPresence ROC AUC and validation-threshold sensitivity\nLine / ring balanced accuracy\nSame held-out phantoms across all conditions')
    for cx in (1.55,4.95): arrow(cx,3.2,cx,1.65)
    arrow(7,3.65,5.9,1.65)
    ax.text(8.35,.8,'Joint endpoint audit\nClassification success does not\nvalidate map localization',ha='center',va='center',fontsize=8)
    save(fig,1)
    names=[f"{c['frequency_hz']//1000000} MHz\n{c['view_noise']}" for c in original['conditions']]; x=np.arange(8)
    fig,axes=plt.subplots(2,2,figsize=(7.1,6.2),layout='constrained')
    for method,color,offset,marker in [('direct','black',-.13,'o'),('enhanced','.55',0,'^'),('xray_sr','#0072B2',.13,'s')]:
        for idx,metric in enumerate((0,3)):
            value=np.array([c['models'][method]['metrics'][metric] for c in original['conditions']]); ci=np.array([c['models'][method]['ci95'][metric] for c in original['conditions']])
            axes[0,idx].errorbar(x+offset,value,yerr=np.maximum(0,np.stack((value-ci[:,0],ci[:,1]-value))),fmt=marker,color=color,capsize=1.5,markersize=3,label={'direct':'Direct US','enhanced':'Enhanced','xray_sr':'US + SR map'}[method])
    axes[0,0].set_ylabel('(a) Presence ROC AUC'); axes[0,1].set_ylabel('(b) Pattern balanced accuracy')
    for ax in axes[0]: ax.set_ylim(.25,1.06); ax.axhline(.5,color='.7',ls=':',lw=.8)
    axes[0,0].legend(fontsize=7,loc='lower left')
    mismatch=[c['fixed_05']['fraction_detected_positive_with_no_matched_particle'] for c in audit['conditions']]
    axes[1,0].bar(x,mismatch,color='#0072B2',width=.55)
    for k,c in enumerate(audit['conditions']):
        d=c['fixed_05']; axes[1,0].text(k,mismatch[k]+.03,f"{d['detected_positive_with_no_matched_particle']}/{d['detected_positive_cases']}",ha='center',fontsize=6)
    axes[1,0].set(ylabel='(c) No matched particle among detection TPs',ylim=(0,1.18))
    axes[1,1].plot(x,[c['fixed_05']['fp_per_positive_image'] for c in audit['conditions']],'s-',color='#0072B2',markersize=3,label='Positive phantoms')
    axes[1,1].plot(x,[c['fixed_05']['fp_per_negative_image'] for c in audit['conditions']],'o--',color='black',markersize=3,label='Negative phantoms')
    axes[1,1].set_ylabel('(d) False candidate centroids / image'); axes[1,1].legend(fontsize=7)
    for ax in axes.flat:
        ax.set_xticks(x,names,fontsize=6); ax.grid(axis='y',alpha=.2); ax.set_xlabel('Frequency / view / reference SNR (dB)',fontsize=8)
    save(fig,2)
    fig,axes=plt.subplots(2,4,figsize=(7.1,4.1),layout='constrained')
    for k,(ax,c) in enumerate(zip(axes.flat,original['conditions'])):
        values=c['sr_froc']; fp=[v['fp_per_image'] for v in values]; sens=[v['sensitivity'] for v in values]
        ax.plot(fp,sens,'o-',lw=.7,ms=3,color='#0072B2'); ax.plot(fp[4],sens[4],'s',ms=5,color='black')
        ax.set_xscale('symlog',linthresh=.03); ax.set(xlim=(-.002,30),ylim=(-.03,1.03))
        ax.set_xticks([0,.1,1,10],['0','0.1','1','10'])
        ax.set_title(names[k].replace('\n',' / '),fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=.2)
        if k%4==0: ax.set_ylabel('Particle sensitivity',fontsize=8)
        if k>=4: ax.set_xlabel('False candidates / image',fontsize=8)
    save(fig,3)
    tables=[]
    def add(title,headers,rows,note): tables.append({'title':title,'headers':headers,'rows':rows,'note':note})
    add('Acquisition and model specification',['Parameter','Fixed specification'],[
      ['Phantoms and split','480; 160 per class; per class 100 training / 30 validation / 30 test'],
      ['Target','Five 100 um hydroxyapatite cylinders; absent, line, or ring'],
      ['Geometry','Line: -1,-0.5,0,0.5,1 mm; ring radius sqrt(0.5) mm; shared centroid and RMS radius'],
      ['Nuisance variation','Rotation 0–2pi; x/y shift ±2 mm; sound speed ±1%; attenuation ±10%; gain ±5%'],
      ['Background','c = 1540 m/s; density = 1041 kg/m^3; alpha0 = 8.635 Np/m/MHz^1.2'],
      ['Solid','density = 3180 kg/m^3; Young modulus = 6 GPa; Poisson ratio = 0.27'],
      ['Acquisition','110 mm radius; 32 full / 16 limited elements; 0.3 mm arc width'],
      ['Channels','992 full / 240 limited; exclude co-located transmitter/receiver'],
      ['Pulse and sampling','1 / 3 MHz; 80% -6 dB bandwidth; 20 MHz sampling; 4096 samples'],
      ['Frequency and angular grid','0.1–8 MHz; 1618 FFT bins; angular orders -16 to 16'],
      ['Noise','Reference SNR 20 / 40 dB; shared absolute noise calibration'],
      ['Noise reference','Separate nominal fat reference: c = 1440.2 m/s; alpha0 = 4.3578 Np/m/MHz^1.2; 0.22 m path at 1 MHz'],
      ['Map grids','128 x 128 at 64 um; SR occupancy 256 x 256 at 32 um'],
      ['Training','Adam 0.001; batch 8; 30 epochs; seed 20261115; validation-best checkpoint'],
      ['Localization','Four-connected components; intensity-weighted centroids; 150 um one-to-one matching']
    ],'The background is known to the inverse model. Noise labels are referenced to a fixed diametric signal, not lesion-specific SNR. Acquisition views do not have matched total transmission counts.')
    rows=[]
    for c in original['conditions']:
        rr=[f"{c['frequency_hz']//1000000} MHz {c['view_noise']}"]
        for m in ('direct','enhanced','xray_sr'):
            v=c['models'][m]; lo,hi=v['ci95'][0]; rr.append(f"{v['metrics'][0]:.3f} ({lo:.3f}–{hi:.3f})")
        rows.append(rr)
    add('Presence detection ROC AUC',['Condition','Direct US','Enhanced','US + SR map'],rows,'Parentheses are 95% stratified phantom-bootstrap intervals, conditional on the fitted models; 90 test phantoms per condition. Saturated empirical intervals are not zero population uncertainty.')
    rows=[]
    for c in audit['conditions']:
        d=c['fixed_05']; rows.append([f"{c['frequency_hz']//1000000} MHz {c['view_noise']}",f"{d['detected_positive_cases']}/60",f"{d['matched_particles']}/300",f"{d['detected_positive_with_no_matched_particle']}/{d['detected_positive_cases']}",f"{d['fp_per_positive_image']:.3f}",f"{d['fp_per_negative_image']:.3f}"])
    add('Paired detection and localization in the US plus SR pathway',['Condition','Detection TP','Matched particles','TP with no match','FP / positive','FP / negative'],rows,'SR threshold 0.5 and 150 um matching. TP means correctly detected positive phantom. FP denotes unmatched candidate centroids. Positive/negative denominators are 60/30; the paired counts and FP strata are exploratory saved-output analyses.')
    tables=json.loads(re.sub(r'(?<![A-Za-z0-9])-(?=\d)','−',json.dumps(tables,ensure_ascii=False)))
    (D/'tables.json').write_text(json.dumps(tables,indent=2),encoding='utf-8',newline='\r\n')
    (D/'supplement-tables.json').write_text(json.dumps(build_supplement_tables(original,audit),indent=2),encoding='utf-8',newline='\r\n')
    result={'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'results/full-study/evaluation.json',ROOT/'results/publication-task-audit/paired-task-audit.json']},'gpu_calls':0,'figures':3,'tables':3}
    (D/'figure-build.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':
    start=time.perf_counter()
    try: main()
    finally: budget.record('cpu_manuscript_figures_tables',time.perf_counter()-start)
