"""Four noise/view reconstructions from one unchanged GPU template traversal."""
import gpu_reconstruct as gr
import pilot
import cluster_physics as cp
import numpy as np
import torch
import time

VARIANTS=('full20','limited20','full40','limited40')


@torch.inference_mode()
def reconstruct_four(config,fc,background,rf20,rf40,sigma20,sigma40,
                     pixel_block=256,frequency_block=128,points=None,
                     check_budget=None,progress=None):
    started=time.perf_counter(); cpu_start=time.process_time()
    freq,pulse,coeff,mask=gr.prepare(config,fc,background)
    if points is None:
        axis=(np.arange(128)-63.5)*64e-6
        xx,yy=np.meshgrid(axis,axis)
        points=np.column_stack((xx.ravel(),yy.ravel()))
    y20=np.fft.rfft(rf20,axis=0)[mask]
    y40=np.fft.rfft(rf40,axis=0)[mask]
    full=cp.channel_mask(config); limited=cp.channel_mask(config,True)
    measured=np.stack((y20*full,y20*limited,y40*full,y40*limited),axis=1)
    f=torch.as_tensor(freq,device='cuda',dtype=torch.float64)
    s=torch.as_tensor(pulse,device='cuda',dtype=torch.complex64)
    a=torch.as_tensor(coeff,device='cuda',dtype=torch.complex64)
    y=torch.as_tensor(measured.reshape(len(freq),4,-1),device='cuda',dtype=torch.complex64)
    p=torch.as_tensor(points,device='cuda',dtype=torch.float64)
    full_t=torch.as_tensor(full.ravel(),device='cuda')
    limited_t=torch.as_tensor(limited.ravel(),device='cuda')
    sigma=torch.tensor((sigma20,sigma20,sigma40,sigma40),device='cuda',dtype=torch.float64)
    output=torch.zeros((len(points),4),device='cuda',dtype=torch.float64)
    norms=torch.zeros((len(points),2),device='cuda',dtype=torch.float64)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    event0=torch.cuda.Event(enable_timing=True); event1=torch.cuda.Event(enable_timing=True)
    event0.record()
    for p0 in range(0,len(points),pixel_block):
        p1=min(p0+pixel_block,len(points))
        numerator=torch.zeros((p1-p0,4),device='cuda',dtype=torch.float64)
        energy=torch.zeros((p1-p0,2),device='cuda',dtype=torch.float64)
        for f0 in range(0,len(freq),frequency_block):
            if check_budget: check_budget()
            f1=min(f0+frequency_block,len(freq))
            t=gr.gpu_templates(f[f0:f1],s[f0:f1],a[f0:f1],p[p0:p1],background,config)
            t=t.reshape(f1-f0,p1-p0,-1)*full_t[None,None]
            left=t.conj().permute(1,0,2).reshape(p1-p0,-1)
            right=y[f0:f1].permute(1,0,2).reshape(4,-1)
            numerator+=(left@right.T).real.to(torch.float64)
            e=t.real.square()+t.imag.square()
            energy[:,0]+=e.sum(dim=(0,2),dtype=torch.float64)
            energy[:,1]+=e[:,:,limited_t].sum(dim=(0,2),dtype=torch.float64)
            del t,left,right,e
        norm=torch.sqrt((2/config['samples'])*energy)
        output[p0:p1]=(2/config['samples'])*numerator/(norm[:,[0,1,0,1]]*sigma[None]*1.001)
        norms[p0:p1]=norm
        if progress: progress(p1,len(points),time.perf_counter()-started)
    event1.record(); torch.cuda.synchronize()
    info={'wall_seconds':time.perf_counter()-started,'cuda_seconds':event0.elapsed_time(event1)/1000,
          'host_cpu_active_seconds':time.process_time()-cpu_start,
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'variants':list(VARIANTS)}
    return output.cpu().numpy(),norms.cpu().numpy(),info
