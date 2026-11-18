"""Algebraically equivalent recurrence evaluation of the cluster gate.

Hankel recurrence H_(n+1)=2n/z H_n-H_(n-1) reuses H0 and H1.
No change to frequencies, modes, geometry, or scattering equations.
"""
import cluster_physics as cp
import numpy as np
from scipy.special import hankel1


def hankel_orders(z, modes):
    z = np.asarray(z)
    values = [hankel1(0, z)]
    if modes:
        values.append(hankel1(1, z))
    for n in range(1, modes):
        values.append(2*n/z*values[-1]-values[-2])
    return np.stack(values, axis=-1)


def signed_hankel(z, modes):
    positive = hankel_orders(z, modes)
    orders = np.arange(-modes, modes+1)
    return positive[..., np.abs(orders)] * np.where(orders < 0, (-1.0)**np.abs(orders), 1)


def arc_propagators(frequencies, centers, background, config, modes, quadrature=3):
    n = np.arange(-modes, modes+1)
    f = np.atleast_1d(frequencies)
    k = 2*np.pi*f/background["c"]
    alpha = background["alpha0"]*(f/1e6)**config["attenuation_power"]
    angles = 2*np.pi*np.arange(config["full_elements"])/config["full_elements"]
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    out = np.zeros((len(f), len(centers), len(n), len(angles)), complex)
    inc = np.zeros_like(out)
    for node, weight in zip(nodes, weights/2):
        theta = angles + node*config["element_width_m"]/(2*config["radius_m"])
        sensors = config["radius_m"]*np.column_stack((np.cos(theta), np.sin(theta)))
        delta = sensors[None]-centers[:,None]
        distance = np.linalg.norm(delta,axis=-1)
        phase = np.arctan2(delta[...,1],delta[...,0])
        radial = signed_hankel(k[:,None,None]*distance[None],modes).transpose(0,1,3,2)
        radial *= np.exp(-alpha[:,None,None,None]*distance[None,:,None,:])
        angular = np.exp(1j*n[None,:,None]*phase[:,None,:])
        out += weight*radial*angular[None]
        inc += weight*radial*angular.conj()[None]
    return inc.reshape(len(f),-1,len(angles)),out.reshape(len(f),-1,len(angles))


def translation(frequencies, centers, background, modes):
    n = np.arange(-modes,modes+1)
    width = len(n)
    k = 2*np.pi*np.atleast_1d(frequencies)/background["c"]
    result = np.zeros((len(k),len(centers)*width,len(centers)*width),complex)
    orders = n[None,:]-n[:,None]
    for j in range(len(centers)):
        for ell in range(len(centers)):
            if j == ell:
                continue
            delta = centers[j]-centers[ell]
            d = np.linalg.norm(delta)
            phase = np.arctan2(delta[1],delta[0])
            h = signed_hankel(k*d,2*modes)
            block = h[:,orders+2*modes]*np.exp(1j*orders*phase)
            result[:,j*width:(j+1)*width,ell*width:(ell+1)*width] = block
    return result


def install():
    cp.arc_propagators = arc_propagators
    cp.translation = translation
