"""Generate the frozen 480 phantom study, resumably, under the total cap."""
import pilot
import cluster_physics as cp
import cluster_fast
import study_reconstruct as sr
import study_budget as budget
import json
import hashlib
from pathlib import Path
import time
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'results/full-study'


def make_manifest():
    rng=np.random.default_rng(20261115)
    result=[]
    for label,pattern in enumerate(('absent','line','ring')):
        allocation=rng.permutation(160)
        for index in range(160):
            rank=int(np.flatnonzero(allocation==index)[0])
            split='train' if rank<100 else 'validation' if rank<130 else 'test'
            row={'id':len(result),'class':label,'pattern':pattern,'split':split,
                 'angle':float(rng.uniform(0,2*np.pi)),
                 'shift_m':rng.uniform(-0.002,0.002,size=2).tolist(),
                 'c_scale':float(rng.uniform(0.99,1.01)),'alpha_scale':float(rng.uniform(0.9,1.1)),
                 'gain':float(rng.uniform(0.95,1.05))}
            result.append(row)
    return result


def occupancy(row):
    result=np.zeros((256,256),np.float32)
    if row['class']==0: return result
    points=cp.positions(row['pattern'],row['shift_m'],row['angle'])
    offsets=(np.arange(16)+0.5)/16-0.5
    for x,y in points:
        cx=(x/32e-6)+127.5; cy=(y/32e-6)+127.5
        for iy in range(max(0,int(cy)-3),min(256,int(cy)+4)):
            for ix in range(max(0,int(cx)-3),min(256,int(cx)+4)):
                dx=((ix+offsets)-cx)*32e-6
                dy=((iy+offsets)-cy)*32e-6
                result[iy,ix]+=np.mean(dx[None,:]**2+dy[:,None]**2<=(50e-6)**2)
    expected=5*np.pi*(50e-6/32e-6)**2
    if abs(result.sum()/expected-1)>0.05:
        raise RuntimeError('High-resolution target area numerical gate failed')
    return np.clip(result,0,1)


def main():
    FOLDER.mkdir(exist_ok=True)
    freeze=json.loads((FOLDER/'freeze.json').read_text())
    for p,sha in freeze['files'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'Frozen input changed: {p}')
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(0.8)
    cluster_fast.install()
    cfg=pilot.load_config()
    manifest=make_manifest()
    manifest_path=FOLDER/'phantoms.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=manifest: raise RuntimeError('Manifest changed')
    else: manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    path=FOLDER/'images.npy'; target_path=FOLDER/'targets.npy'
    images=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(480,2,4,128,128))
    targets=np.lib.format.open_memmap(target_path,mode='r+' if target_path.exists() else 'w+',dtype=np.float16,shape=(480,256,256))
    complete_path=FOLDER/'completed.json'
    done=set(json.loads(complete_path.read_text()) if complete_path.exists() else [])
    ref_rms=float(np.sqrt(np.mean(pilot.reference_rf(cfg)**2)))
    sigma20=ref_rms*0.1; sigma40=ref_rms*0.01
    t_all=time.perf_counter()
    for row in manifest:
        targets[row['id']]=occupancy(row)
        for fi,fc in enumerate((1000000,3000000)):
            key=f"{row['id']}:{fi}"
            if key in done: continue
            budget.check(reserve=3600)
            bg=dict(cfg['backgrounds']['glandular'])
            bg['c']*=row['c_scale']; bg['alpha0']*=row['alpha_scale']
            if row['class']==0:
                signal=np.zeros((4096,32,32),np.float64)
            else:
                start=time.perf_counter()
                try:
                    _,signal,info=cp.cluster_rf(cfg,fc,cp.positions(row['pattern'],row['shift_m'],row['angle']),
                                               background=bg,check_budget=lambda:budget.check(reserve=3600,ongoing=time.perf_counter()-start))
                    if info['coupled_residual']>1e-10: raise RuntimeError('Coupled residual gate failed')
                    signal*=row['gain']
                finally:
                    budget.record('cpu_forward',time.perf_counter()-start,phantom=row['id'],frequency_hz=fc)
            rng=np.random.default_rng(2026111500+row['id']*10+fi)
            rf20=signal+rng.normal(0,sigma20,signal.shape)
            rf40=signal+rng.normal(0,sigma40,signal.shape)
            start=time.perf_counter()
            try:
                result,norm,info=sr.reconstruct_four(cfg,fc,bg,rf20,rf40,sigma20,sigma40,
                         check_budget=lambda:budget.check(reserve=3600,ongoing=time.perf_counter()-start))
                if not np.all(np.isfinite(result)): raise FloatingPointError('Nonfinite reconstruction')
                if info['peak_allocated_bytes']>8*1024**3: raise RuntimeError('GPU allocation limit exceeded')
                images[row['id'],fi]=result.T.reshape(4,128,128)
                images.flush()
            finally:
                budget.record('gpu_reconstruction',time.perf_counter()-start,phantom=row['id'],frequency_hz=fc)
            done.add(key)
            complete_path.write_text(json.dumps(sorted(done)),encoding='utf-8')
            status={'completed_frequency_cases':len(done),'total_frequency_cases':960,
                    'completed_images':len(done)*4,'total_images':3840,
                    'last_phantom':row['id'],'last_frequency_hz':fc,
                    'device_hours_used_including_prior':budget.consumed()/3600,
                    'segment_wall_hours':(time.perf_counter()-t_all)/3600}
            (FOLDER/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
            if len(done)%10==0 or len(done)==1:
                print(json.dumps(status),flush=True)
            del signal,rf20,rf40,result,norm
    targets.flush()
    assert len(done)==960
    (FOLDER/'generation-complete.json').write_text(json.dumps({'frequency_cases':len(done),'images':3840,
                                               'device_hours_used':budget.consumed()/3600},indent=2),encoding='utf-8')
    print('GENERATION COMPLETE',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); main()
