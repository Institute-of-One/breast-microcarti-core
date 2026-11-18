"""Frozen three-path training. Test images never enter model selection."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch.nn import functional as F
from study_models import StudyModel
import study_budget as budget

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/full-study'
METHODS=('direct','enhanced','xray_sr')

def main():
    assert json.loads((OUT/'generation-complete.json').read_text())['images']==3840
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    manifest=json.loads((OUT/'phantoms.json').read_text())
    images=np.load(OUT/'images.npy',mmap_mode='r')
    targets=np.load(OUT/'targets.npy',mmap_mode='r')
    train=np.array([r['id'] for r in manifest if r['split']=='train'])
    val=np.array([r['id'] for r in manifest if r['split']=='validation'])
    labels=np.array([r['class'] for r in manifest])
    start=time.perf_counter()
    try:
        # Exact train-only robust normalization; no validation/test statistics.
        q=np.percentile(np.asarray(images[train]),[25,50,75])
        norm={'median':float(q[1]),'iqr':float(q[2]-q[0]),'train_ids':train.tolist()}
        if norm['iqr']<=0: raise ValueError('Degenerate train normalization')
        (OUT/'normalization.json').write_text(json.dumps(norm,indent=2))
    finally: budget.record('cpu_normalization',time.perf_counter()-start)
    def dataset(ids):
        x=np.asarray(images[ids]).reshape(-1,1,128,128).copy()
        x=np.clip((x-norm['median'])/norm['iqr'],-5,5)
        y=np.repeat(labels[ids],8)
        target=np.repeat(np.asarray(targets[ids],dtype=np.float32),8,axis=0)[:,None]
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(target)
    train_data=dataset(train); val_data=dataset(val)
    for method in METHODS:
        if (OUT/f'{method}-training.json').exists(): continue
        budget.check(reserve=900)
        torch.manual_seed(20261115); np.random.seed(20261115)
        model=StudyModel(method).cuda()
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        rng=torch.Generator().manual_seed(20261115)
        best=float('inf'); history=[]
        start=time.perf_counter()
        try:
            for epoch in range(30):
                budget.check(reserve=900,ongoing=time.perf_counter()-start)
                model.train(); total=0.
                for indices in torch.randperm(len(train_data[0]),generator=rng).split(8):
                    x,y,t=(a[indices].cuda() for a in train_data)
                    opt.zero_grad(set_to_none=True)
                    p,s=model(x); loss=F.cross_entropy(p,y)
                    if s is not None: loss=loss+F.binary_cross_entropy_with_logits(s,t)
                    loss.backward(); opt.step(); total+=float(loss.detach())*len(indices)
                model.eval(); v=0.
                with torch.no_grad():
                    for indices in torch.arange(len(val_data[0])).split(8):
                        x,y,t=(a[indices].cuda() for a in val_data)
                        p,s=model(x); loss=F.cross_entropy(p,y)
                        if s is not None: loss=loss+F.binary_cross_entropy_with_logits(s,t)
                        v+=float(loss)*len(indices)
                v/=len(val_data[0]); row={'epoch':epoch+1,'train_loss':total/len(train_data[0]),'validation_loss':v}
                history.append(row)
                if v<best:
                    best=v
                    torch.save({'state':model.state_dict(),'epoch':epoch+1,'validation_loss':v,'method':method},OUT/f'{method}.pt')
                print(method,json.dumps(row),flush=True)
            torch.cuda.synchronize()
            info={'method':method,'history':history,'best_validation_loss':best,
                  'parameters':sum(p.numel() for p in model.parameters()),
                  'checkpoint_sha256':hashlib.sha256((OUT/f'{method}.pt').read_bytes()).hexdigest(),
                  'test_used_for_selection':False,'wall_seconds':time.perf_counter()-start}
            (OUT/f'{method}-training.json').write_text(json.dumps(info,indent=2))
        finally: budget.record('gpu_training',time.perf_counter()-start,method=method)
        del model,opt
    (OUT/'training-complete.json').write_text(json.dumps({'methods':METHODS,'device_hours':budget.consumed()/3600},indent=2))

if __name__=='__main__': main()
