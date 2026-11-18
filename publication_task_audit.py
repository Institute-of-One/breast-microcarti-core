"""Exploratory saved-output audit; no model/threshold selection or inference."""
import pilot  # thread limits before numpy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import study_budget as budget

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/publication-task-audit'
DATA=ROOT/'results/full-study'

def summarize(counts,positive,detected):
    matched,truth,fp=counts.T
    tp=positive & detected
    return {'localization_sensitivity':float(matched.sum()/truth.sum()),
            'fp_per_image':float(fp.mean()),'fp_per_negative_image':float(fp[~positive].mean()),
            'fp_per_positive_image':float(fp[positive].mean()),
            'detected_positive_cases':int(tp.sum()),
            'detected_positive_with_no_matched_particle':int(np.sum(tp & (matched==0))),
            'fraction_detected_positive_with_no_matched_particle':float(np.mean(matched[tp]==0)) if tp.any() else None,
            'fraction_detected_positive_with_all_five_matched':float(np.mean(matched[tp]==5)) if tp.any() else None,
            'positive_cases_with_any_match':int(np.sum(positive & (matched>0))),
            'matched_particles':int(matched.sum()),'truth_particles':int(truth.sum()),
            'fp_negative_total':int(fp[~positive].sum()),'fp_positive_total':int(fp[positive].sum())}

def main():
    started=time.perf_counter(); OUT.mkdir(exist_ok=True,parents=True)
    result=json.loads((DATA/'evaluation.json').read_text())
    rows=json.loads((DATA/'phantoms.json').read_text())
    y=np.array([rows[i]['class'] for i in result['test_ids']]); positive=y>0
    counts=np.load(DATA/'sr-localization-counts.npy')
    pred=np.load(DATA/'xray_sr-test-predictions.npy')
    assert counts.shape==(90,8,9,3) and pred.shape==(90,8,3)
    assert np.all(counts[...,0]>=0) and np.all(counts[...,2]>=0)
    assert np.all(counts[...,0]<=counts[...,1])
    assert np.all(counts[positive,:,:,1]==5) and np.all(counts[~positive,:,:,1]==0)
    rng=np.random.default_rng(20261116)
    samples=np.array([np.concatenate([rng.choice(np.flatnonzero(y==j),30,replace=True) for j in range(3)]) for _ in range(1000)])
    report={'at_utc':datetime.now(timezone.utc).isoformat(),'analysis_status':'POST_HOC_EXPLORATORY_SAVED_OUTPUTS_ONLY',
            'test_ids':result['test_ids'],'sampling_unit':'phantom, stratified by class; 1000 bootstrap replicates',
            'no_new_inference_training_or_simulation':True,'conditions':[],
            'architecture_note':'Classification uses both bicubic US and SR probability map; this audit does not establish which channel caused detection success.',
            'hallucination_note':'GT-unmatched connected-component centroids are operational false localizations, not measurement/null-space hallucinations.'}
    keys=('localization_sensitivity','fp_per_image','fp_per_negative_image','fp_per_positive_image',
          'fraction_detected_positive_with_no_matched_particle','fraction_detected_positive_with_all_five_matched')
    for c,condition in enumerate(result['conditions']):
        if time.perf_counter()-started>300: raise TimeoutError('Five-minute extra CPU cap')
        budget.check(ongoing=time.perf_counter()-started)
        threshold=condition['models']['xray_sr']['threshold']
        detected=1-pred[:,c,0]>=threshold
        fixed=counts[:,c,4]
        observed=summarize(fixed,positive,detected)
        boot=[summarize(fixed[ix],positive[ix],detected[ix]) for ix in samples]
        intervals={}
        for key in keys:
            values=[b[key] for b in boot if b[key] is not None]
            intervals[key]={'ci95':np.percentile(values,[2.5,97.5]).tolist() if values else None,'valid_replicates':len(values)}
        froc=[]
        for t in range(9):
            s=summarize(counts[:,c,t],positive,detected)
            prior=condition['sr_froc'][t]
            assert abs(s['localization_sensitivity']-prior['sensitivity'])<1e-12
            assert abs(s['fp_per_image']-prior['fp_per_image'])<1e-12
            froc.append({'threshold':(t+1)/10,**s})
        by_pattern={}
        for label,name in [(1,'line'),(2,'ring')]:
            selected=y==label
            by_pattern[name]={'sensitivity':float(fixed[selected,0].sum()/fixed[selected,1].sum()),
                              'fp_per_image':float(fixed[selected,2].mean())}
        report['conditions'].append({'frequency_hz':condition['frequency_hz'],'view_noise':condition['view_noise'],
                  'fixed_05':observed,'ci95_05':intervals,'all_prespecified_thresholds':froc,'by_pattern_05':by_pattern})
    report['sources_sha256']={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in
                             ('evaluation.json','sr-localization-counts.npy','xray_sr-test-predictions.npy','phantoms.json')}
    report['plan_sha256']=hashlib.sha256((ROOT/'docs/PUBLICATION_SCOPE_PLAN_20261115.md').read_bytes()).hexdigest()
    (OUT/'paired-task-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([{'condition':str(c['frequency_hz'])+' '+c['view_noise'],**c['fixed_05']} for c in report['conditions']],indent=2))

if __name__=='__main__':
    start=time.perf_counter()
    try: main()
    finally: budget.record('cpu_publication_saved_output_audit',time.perf_counter()-start)
