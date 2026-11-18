"""Block GPU evaluation of the frozen single-template matched-filter map.

Same frequency grid, elastic modes, finite apertures, and ridge score as v2.
Hankel H0/H1 use eight large-argument terms (kr>40 in this ROI), followed
by the exact integer-order recurrence. CPU/SciPy checks gate actual use.
"""
import pilot
import cluster_physics as cp
import cluster_fast
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import hankel1
import torch

ROOT=Path(__file__).resolve().parent


def gpu_hankel01(z, order, complex_dtype=torch.complex64):
    """DLMF large-z expansion, through k=8; z is positive real float64."""
    series=torch.ones_like(z,dtype=torch.complex128)
    term=torch.ones_like(series)
    for k in range(1,9):
        term=term*(1j*(4*order*order-(2*k-1)**2)/(8*k))/z
        series=series+term
    phase=z-order*math.pi/2-math.pi/4
    result=torch.sqrt(2/(math.pi*z))*torch.exp(1j*phase)*series
    return result.to(complex_dtype)


def gpu_signed_hankel(z, modes, complex_dtype=torch.complex64):
    positive=[gpu_hankel01(z,0,complex_dtype),gpu_hankel01(z,1,complex_dtype)]
    inverse=z.reciprocal().to(torch.float32 if complex_dtype==torch.complex64 else torch.float64)
    for n in range(1,modes):
        positive.append(2*n*inverse*positive[-1]-positive[-2])
    stacked=torch.stack(positive[:modes+1],dim=-1)
    orders=torch.arange(-modes,modes+1,device=z.device)
    signs=torch.where(orders<0,(-1.0)**orders.abs(),1.0)
    return stacked[...,orders.abs()]*signs


def gpu_propagators(frequencies,points,background,config,modes=16,quadrature=3):
    """Return F,P,N,R incoming/outgoing coefficients, coherent arc average."""
    f=frequencies
    device=f.device
    nodes,weights=np.polynomial.legendre.leggauss(quadrature)
    theta=2*math.pi*torch.arange(config['full_elements'],device=device,dtype=torch.float64)/config['full_elements']
    theta=theta[:,None]+torch.as_tensor(nodes,device=device)*config['element_width_m']/(2*config['radius_m'])
    sensors=config['radius_m']*torch.stack((torch.cos(theta),torch.sin(theta)),dim=-1)
    delta=sensors[None]-points[:,None,None]
    distance=torch.linalg.vector_norm(delta,dim=-1)
    phase=torch.atan2(delta[...,1],delta[...,0])
    k=2*math.pi*f/background['c']
    alpha=background['alpha0']*(f/1e6)**config['attenuation_power']
    z=k[:,None,None,None]*distance[None]
    radial=gpu_signed_hankel(z,modes)
    loss=torch.exp(-alpha[:,None,None,None]*distance[None]).float()
    radial=radial*loss[...,None]
    n=torch.arange(-modes,modes+1,device=device,dtype=torch.float64)
    angular=torch.exp(1j*phase[...,None]*n).to(torch.complex64)
    w=torch.as_tensor(weights/2,device=device,dtype=torch.float32)
    # F,P,R,Q,N -> F,P,N,R
    out=(radial*angular[None]*w[None,None,None,:,None]).sum(dim=3).permute(0,1,3,2)
    inc=(radial*angular.conj()[None]*w[None,None,None,:,None]).sum(dim=3).permute(0,1,3,2)
    return inc,out


def gpu_templates(frequencies,pulse,coeff,points,background,config):
    inc,out=gpu_propagators(frequencies,points,background,config)
    response=0.25j*torch.matmul(out.transpose(-2,-1),coeff[:,None,:,None]*inc)
    return pulse[:,None,None,None]*response.conj()


def prepare(config,fc,background):
    f,pulse,mask=pilot.pulse_spectrum(config,fc)
    freq=f[mask]
    coefficient,_=pilot.elastic_modes(freq,50e-6,background,config['materials']['hap'],config['modes'])
    signed=coefficient[:,np.abs(np.arange(-config['modes'],config['modes']+1))]
    return freq,pulse[mask],signed,mask


@torch.inference_mode()
def reconstruct_gpu(config,fc,background,rf,sigma,points=None,pixel_block=256,frequency_block=128,
                    max_seconds=600,progress=True):
    cpu_started=time.process_time(); wall_started=time.perf_counter()
    freq,pulse,coefficient,mask=prepare(config,fc,background)
    if points is None:
        axis=(np.arange(128)-63.5)*64e-6
        xx,yy=np.meshgrid(axis,axis)
        points=np.column_stack((xx.ravel(),yy.ravel()))
    measured=np.fft.rfft(rf,axis=0)[mask]
    device=torch.device('cuda:0')
    f_gpu=torch.as_tensor(freq,device=device,dtype=torch.float64)
    pulse_gpu=torch.as_tensor(pulse,device=device,dtype=torch.complex64)
    coefficient_gpu=torch.as_tensor(coefficient,device=device,dtype=torch.complex64)
    measured_gpu=torch.as_tensor(measured,device=device,dtype=torch.complex64)
    channels=torch.as_tensor(cp.channel_mask(config),device=device)
    measured_gpu=measured_gpu*channels[None]
    scores=torch.zeros(len(points),device=device,dtype=torch.float64)
    norms=torch.zeros_like(scores)
    points_gpu=torch.as_tensor(points,device=device,dtype=torch.float64)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    event_start=torch.cuda.Event(enable_timing=True); event_end=torch.cuda.Event(enable_timing=True)
    event_start.record()
    report_step=max(pixel_block,((len(points)//4)//pixel_block)*pixel_block)
    for p0 in range(0,len(points),pixel_block):
        p1=min(p0+pixel_block,len(points))
        numerator=torch.zeros(p1-p0,device=device,dtype=torch.float64)
        energy=torch.zeros_like(numerator)
        for f0 in range(0,len(freq),frequency_block):
            if time.perf_counter()-wall_started>max_seconds:
                raise TimeoutError('GPU benchmark wall-time allocation reached')
            f1=min(f0+frequency_block,len(freq))
            template=gpu_templates(f_gpu[f0:f1],pulse_gpu[f0:f1],coefficient_gpu[f0:f1],
                                   points_gpu[p0:p1],background,config)
            template=template*channels[None,None]
            products=(template.conj()*measured_gpu[f0:f1,None]).real
            numerator+=products.sum(dim=(0,2,3),dtype=torch.float64)
            energy+=(template.real.square()+template.imag.square()).sum(dim=(0,2,3),dtype=torch.float64)
            del products,template
        factor=2/config['samples']
        norms[p0:p1]=torch.sqrt(factor*energy)
        scores[p0:p1]=(factor*numerator)/(norms[p0:p1]*sigma*1.001)
        if progress and (p0%report_step==0 or p1==len(points)):
            torch.cuda.synchronize()
            print(f'GPU pixels {p1}/{len(points)} wall={time.perf_counter()-wall_started:.2f}s',flush=True)
    event_end.record(); torch.cuda.synchronize()
    info={'wall_seconds':time.perf_counter()-wall_started,'cuda_event_seconds':event_start.elapsed_time(event_end)/1000,
          'cpu_active_seconds':time.process_time()-cpu_started,
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
          'pixel_block':pixel_block,'frequency_block':frequency_block,'pixels':len(points),
          'frequencies':len(freq),'dtype':'complex64 with float64 phase and reductions'}
    return scores.cpu().numpy(),norms.cpu().numpy(),info


def cpu_reference(config,fc,background,rf,sigma,points):
    mask=cp.channel_mask(config)
    measured=rf[:,mask]
    scores=[]; norms=[]
    for p in points:
        _,template,_=cp.cluster_rf(config,fc,np.array([p]),background=background,both=False)
        t=template[:,mask]
        norm=float(np.linalg.norm(t))
        scores.append(float(np.sum(t*measured)/(norm*sigma*1.001)))
        norms.append(norm)
    return np.array(scores),np.array(norms)


def benchmark():
    folder=ROOT/'results/gpu-benchmark'
    freeze=json.loads((folder/'freeze.json').read_text())
    for p,expected in freeze['files'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Frozen specification changed: {p}')
    started=time.perf_counter(); cpu_started=time.process_time()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(0.80)
    config=pilot.load_config(); bg=config['backgrounds']['glandular']; fc=3000000
    cluster_fast.install()
    _,signal,forward_info=cp.cluster_rf(config,fc,cp.positions('line',(0.002,0.0)))
    sigma=float(np.sqrt(np.mean(pilot.reference_rf(config)**2)))*0.1
    rng=np.random.default_rng(20261116)
    rf=signal+rng.normal(0,sigma,signal.shape)
    np.savez_compressed(folder/'benchmark-input.npz',rf=rf,sigma=sigma)
    z=np.geomspace(40,4000,256)
    with torch.inference_mode():
        actual=gpu_signed_hankel(torch.tensor(z,device='cuda',dtype=torch.float64),16).cpu().numpy()
    expected=hankel1(np.arange(-16,17)[None],z[:,None])
    hankel_error=float(np.max(np.abs(actual-expected)/np.maximum(np.abs(expected),1e-30)))
    print(f'Hankel maximum relative error={hankel_error:.6g}',flush=True)
    if hankel_error>0.05:
        raise RuntimeError('Hankel numerical gate failed')
    indices=np.array([0,42,85,127])
    axis=(np.arange(128)-63.5)*64e-6
    ij=np.array([(y,x) for y in indices for x in indices])
    points=np.array([(axis[x],axis[y]) for y,x in ij])
    t=time.perf_counter()
    cpu_scores,cpu_norms=cpu_reference(config,fc,bg,rf,sigma,points)
    cpu_reference_seconds=time.perf_counter()-t
    gpu_scores,gpu_norms,small_info=reconstruct_gpu(config,fc,bg,rf,sigma,points,pixel_block=16,frequency_block=128,progress=False)
    score_error=float(np.linalg.norm(gpu_scores-cpu_scores)/np.linalg.norm(cpu_scores))
    norm_error=float(np.linalg.norm(gpu_norms-cpu_norms)/np.linalg.norm(cpu_norms))
    check={'hankel_max_relative_error':hankel_error,'score_relative_l2_error':score_error,'norm_relative_l2_error':norm_error,
           'cpu_reference_seconds':cpu_reference_seconds,'gpu_validation':small_info,
           'points':points.tolist(),'cpu_scores':cpu_scores.tolist(),'gpu_scores':gpu_scores.tolist(),
           'cpu_norms':cpu_norms.tolist(),'gpu_norms':gpu_norms.tolist(),
           'accuracy_pass':bool(max(hankel_error,score_error,norm_error)<=0.05)}
    (folder/'accuracy-check.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps({k:check[k] for k in ('score_relative_l2_error','norm_relative_l2_error','accuracy_pass')}),flush=True)
    if not check['accuracy_pass']:
        raise RuntimeError('CPU/GPU numerical gate failed')
    image,norms,info=reconstruct_gpu(config,fc,bg,rf,sigma)
    np.savez_compressed(folder/'reconstruction.npz',score=image.reshape(128,128),norm=norms.reshape(128,128))
    sampled=image.reshape(128,128)[ij[:,0],ij[:,1]]
    full_sample_error=float(np.linalg.norm(sampled-cpu_scores)/np.linalg.norm(cpu_scores))
    info.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                device=torch.cuda.get_device_name(0),full_image_accuracy_relative_l2=full_sample_error,
                cpu_total_active_seconds=time.process_time()-cpu_started,
                all_step_wall_seconds=time.perf_counter()-started,forward=forward_info,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                prior_cpu_accounted_seconds=freeze['cpu_prior_accounted_seconds'],
                dtype_accuracy_gate=check['accuracy_pass'],peak_memory_target_bytes=8*1024**3,
                full_dataset_reconstruction_calls=960,
                full_dataset_gpu_wall_seconds_linear_estimate=960*info['wall_seconds'],
                independent_ai_review='NOT_RUN_NO_AUTHORIZATION')
    (folder/'benchmark.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',action='store_true',required=True)
    parser.parse_args(); benchmark()
