"""Single frozen test evaluation, paired phantom bootstrap and SR localization."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label,center_of_mass
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from study_models import StudyModel
import cluster_physics as cp
import study_budget as budget

OUT=Path(__file__).resolve().parent/'results/full-study'
METHODS=('direct','enhanced','xray_sr')

def auc(y,s):
    pos=y>0; n=pos.sum(); m=len(y)-n
    return float((rankdata(s)[pos].sum()-n*(n+1)/2)/(n*m)) if n and m else float('nan')

def metrics(y,p,threshold):
    positive=y>0; pred=np.argmax(p[:,1:],axis=1)+1
    return np.array([auc(y,1-p[:,0]),np.mean((1-p[positive,0])>=threshold),
                     np.mean((1-p[~positive,0])>=threshold),
                     .5*(np.mean(pred[y==1]==1)+np.mean(pred[y==2]==2))])

def localize(prob,row,threshold):
    components,n=label(prob>=threshold)
    # Eight-connected? scipy default is four-connected; fixed before evaluation.
    centers=np.array(center_of_mass(prob,components,np.arange(1,n+1))) if n else np.empty((0,2))
    xy=(centers[:,::-1]-127.5)*32e-6
    truth=cp.positions(row['pattern'],row['shift_m'],row['angle']) if row['class'] else np.empty((0,2))
    matches=0
    if len(xy) and len(truth):
        distance=np.linalg.norm(xy[:,None,:]-truth[None,:,:],axis=2)
        # Maximize valid matches before minimizing distance (dummy assignments allowed).
        a,b=linear_sum_assignment(np.where(distance<=150e-6,distance,1000.))
        matches=int(np.sum(distance[a,b]<=150e-6))
    return matches,len(truth),len(xy)-matches

def main():
    assert (OUT/'training-complete.json').exists()
    if (OUT/'evaluation.json').exists(): raise RuntimeError('Test already evaluated; preserve frozen result')
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    rows=json.loads((OUT/'phantoms.json').read_text()); norm=json.loads((OUT/'normalization.json').read_text())
    images=np.load(OUT/'images.npy',mmap_mode='r')
    test=[r['id'] for r in rows if r['split']=='test']; val=[r['id'] for r in rows if r['split']=='validation']
    predictions={}; map_counts={}
    start=time.perf_counter()
    try:
        for method in METHODS:
            budget.check(reserve=600,ongoing=time.perf_counter()-start)
            model=StudyModel(method).cuda()
            saved=torch.load(OUT/f'{method}.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(saved['state']); model.eval()
            predictions[method]={}
            for split,ids in [('validation',val),('test',test)]:
                values=np.empty((len(ids),8,3),np.float32)
                counts=np.zeros((len(ids),8,9,3),np.int32) if split=='test' and method=='xray_sr' else None
                with torch.no_grad():
                    for k,i in enumerate(ids):
                        x=np.asarray(images[i]).reshape(8,1,128,128).copy()
                        x=np.clip((x-norm['median'])/norm['iqr'],-5,5)
                        p,s=model(torch.from_numpy(x).cuda()); values[k]=p.softmax(1).cpu().numpy()
                        if counts is not None:
                            maps=s.sigmoid().cpu().numpy()[:,0]
                            for c,prob in enumerate(maps):
                                for t,threshold in enumerate(np.arange(1,10)/10): counts[k,c,t]=localize(prob,rows[i],threshold)
                predictions[method][split]=values
                np.save(OUT/f'{method}-{split}-predictions.npy',values)
                if counts is not None: map_counts=counts; np.save(OUT/'sr-localization-counts.npy',counts)
            del model
        torch.cuda.synchronize()
    finally: budget.record('gpu_test_inference_and_localization',time.perf_counter()-start)
    start=time.perf_counter()
    try:
        y=np.array([rows[i]['class'] for i in test]); vy=np.array([rows[i]['class'] for i in val])
        # Stratified paired resampling preserves all classes and the phantom cluster.
        rng=np.random.default_rng(20261115)
        samples=np.array([np.concatenate([rng.choice(np.flatnonzero(y==j),30,replace=True) for j in range(3)]) for _ in range(1000)])
        result={'clinical_claim':False,'metric_order':['presence_auc','sensitivity_at_validation_5pct_fpr','test_fpr','pattern_balanced_accuracy'],
                'test_ids':test,'validation_ids':val,'bootstrap':'1000 stratified paired phantom resamples; thresholds held fixed',
                'conditions':[],'independent_ai_review':'NOT_RUN_NO_EXTERNAL_TRANSMISSION_AUTHORIZATION'}
        null_labels=rng.permutation(np.repeat(np.arange(3),10))
        result['noise_only_control']={'null_test_ids':np.array(test)[y==0].tolist(),
                                      'independent_random_labels':null_labels.tolist(),'conditions':[]}
        for c in range(8):
            budget.check(ongoing=time.perf_counter()-start)
            entry={'frequency_hz':(1000000,3000000)[c//4],'view_noise':('full20','limited20','full40','limited40')[c%4],'models':{},'paired_differences':{}}
            boots={}
            for method in METHODS:
                p=predictions[method]['test'][:,c]
                vp=predictions[method]['validation'][:,c]
                threshold=float(np.quantile(1-vp[vy==0,0],.95))
                observed=metrics(y,p,threshold)
                b=np.array([metrics(y[ix],p[ix],threshold) for ix in samples]); boots[method]=b
                confusion=np.zeros((2,2),int)
                for a,z in zip(y[y>0]-1,np.argmax(p[y>0,1:],axis=1)): confusion[a,z]+=1
                entry['models'][method]={'threshold':threshold,'metrics':observed.tolist(),
                    'ci95':np.percentile(b,[2.5,97.5],axis=0).T.tolist(),'pattern_confusion':confusion.tolist()}
            for method in METHODS[1:]:
                delta=boots[method]-boots['direct']
                entry['paired_differences'][method+'-direct']={'metrics':(np.array(entry['models'][method]['metrics'])-entry['models']['direct']['metrics']).tolist(),
                     'ci95':np.percentile(delta,[2.5,97.5],axis=0).T.tolist()}
            delta=boots['xray_sr']-boots['enhanced']
            entry['paired_differences']['xray_sr-enhanced']={'metrics':(np.array(entry['models']['xray_sr']['metrics'])-entry['models']['enhanced']['metrics']).tolist(),
                      'ci95':np.percentile(delta,[2.5,97.5],axis=0).T.tolist()}
            entry['sr_froc']=[]
            for t in range(9):
                n=map_counts[:,c,t].sum(0)
                entry['sr_froc'].append({'threshold':(t+1)/10,'matched':int(n[0]),'truth':int(n[1]),
                                          'sensitivity':float(n[0]/n[1]),'fp_per_image':float(n[2]/len(test))})
            result['conditions'].append(entry)
            result['noise_only_control']['conditions'].append({method:metrics(null_labels,
                 predictions[method]['test'][y==0,c],entry['models'][method]['threshold']).tolist() for method in METHODS})
        result['device_hours_before_final_checks']=budget.consumed()/3600
        (OUT/'evaluation.json').write_text(json.dumps(result,indent=2))
    finally: budget.record('cpu_statistics',time.perf_counter()-start)

if __name__=='__main__': main()
