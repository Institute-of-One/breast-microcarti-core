"""Non-generative display of one licensed clinical mammogram; no network inference."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pilot
import numpy as np
import pydicom
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import study_budget as budget

def main():
    cache=ROOT/'.cache/public-clinical'; paper=ROOT/'paper'
    d=pydicom.dcmread(cache/'clinical.dcm'); raw=d.pixel_array.astype(np.float32)
    annotated=np.asarray(Image.open(cache/'annotation.jpg').convert('RGB'))
    assert raw.shape==annotated.shape[:2] and str(d.PhotometricInterpretation)=='MONOCHROME2'
    blue=(annotated[:,:,2]>150)&(annotated[:,:,0]<120)&(annotated[:,:,1]<120)
    yy,xx=np.where(blue); assert len(xx)>100
    x0,x1=max(0,int(xx.min())-90),min(raw.shape[1],int(xx.max())+91)
    y0,y1=max(0,int(yy.min())-90),min(raw.shape[0],int(yy.max())+91)
    roi=raw[y0:y1,x0:x1]; lo,hi=np.percentile(roi,[1,99.7])
    spacing=[float(x) for x in d.get('PixelSpacing',d.get('ImagerPixelSpacing'))]
    assert spacing==[.07,.07]
    fig,axes=plt.subplots(1,3,figsize=(7.1,3.7),gridspec_kw={'width_ratios':[.85,1,1]},layout='constrained')
    axes[0].imshow(raw,cmap='gray',vmin=0,vmax=4095,interpolation='nearest')
    axes[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#E69F00',lw=1))
    axes[1].imshow(roi,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
    axes[2].imshow(annotated[y0:y1,x0:x1],interpolation='nearest')
    for k,ax in enumerate(axes):
        ax.axis('off'); ax.text(.02,.98,f'({chr(97+k)})',transform=ax.transAxes,ha='left',va='top',fontsize=10,color='white',bbox={'facecolor':'black','edgecolor':'none','pad':2})
    n=5/spacing[1]; axes[1].plot([20,20+n],[roi.shape[0]-30]*2,color='white',lw=2)
    axes[1].text(20,roi.shape[0]-40,'5 mm',color='white',fontsize=8,va='bottom')
    fig.savefig(paper/'figures/Figure1.png',dpi=600,bbox_inches='tight'); fig.savefig(paper/'figures/Figure1.pdf',bbox_inches='tight'); plt.close(fig)
    provenance=json.loads((paper/'sources/clinical-illustration-provenance.json').read_text())
    provenance['display_processing']={'roi_xyxy':[x0,y0,x1,y1],'full_window':[0,4095],
        'roi_window':[float(lo),float(hi)],'roi_window_rule':'raw ROI 1st to 99.7th percentiles',
        'pixel_spacing_mm':spacing,'annotation_processing':'crop only; original source blue/red marks retained',
        'annotation_alignment':'same dimensions; source version3 correspondence, not new registration validation',
        'no_generative_editing':True,'new_patient_or_model_inference':False,'visual_check_required':True}
    (paper/'sources/clinical-illustration-provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance['display_processing'],indent=2))

if __name__=='__main__':
    t=time.perf_counter()
    try: main()
    finally: budget.record('cpu_public_clinical_figure_display',time.perf_counter()-t)
