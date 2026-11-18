"""Elastic-cylinder cluster gate using Graf translation and a T-matrix solve.

Uses the existing lossless single-cylinder model. As in pilot-v1, external
paths carry approximate real attenuation factors, without causal dispersion.
"""
import pilot  # sets BLAS thread limits before importing numpy
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import hankel1

ROOT = Path(__file__).resolve().parent


def positions(pattern, shift=(0.0, 0.0), angle=0.0):
    if pattern == "line":
        points = np.column_stack((np.arange(-2, 3)*0.0005, np.zeros(5)))
    elif pattern == "ring":
        angles = 2*np.pi*np.arange(5)/5
        points = np.column_stack((np.cos(angles), np.sin(angles))) * np.sqrt(0.5)*0.001
    elif pattern == "single":
        points = np.zeros((1, 2))
    else:
        raise ValueError(pattern)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return points @ rotation.T + np.array(shift)


def arc_propagators(frequencies, centers, background, config, modes, quadrature=3):
    """Outgoing at receivers and regular-wave coefficients of incident sources."""
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
        delta = sensors[None, :, :] - centers[:, None, :]
        distance = np.linalg.norm(delta, axis=-1)
        phase = np.arctan2(delta[..., 1], delta[..., 0])
        radial = hankel1(n[None, None, :, None], k[:, None, None, None]*distance[None, :, None, :])
        radial *= np.exp(-alpha[:, None, None, None]*distance[None, :, None, :])
        angular = np.exp(1j*n[None, :, None]*phase[:, None, :])
        out += weight*radial*angular[None]
        inc += weight*radial*angular.conj()[None]
    return inc.reshape(len(f), -1, len(angles)), out.reshape(len(f), -1, len(angles))


def translation(frequencies, centers, background, modes):
    n = np.arange(-modes, modes+1)
    width = len(n)
    f = np.atleast_1d(frequencies)
    k = 2*np.pi*f/background["c"]
    matrix = np.zeros((len(f), len(centers)*width, len(centers)*width), complex)
    # row n at center j, outgoing column m from center l.
    order = n[None, :] - n[:, None]
    for j in range(len(centers)):
        for ell in range(len(centers)):
            if j == ell:
                continue
            delta = centers[j]-centers[ell]
            distance = np.linalg.norm(delta)
            phase = np.arctan2(delta[1], delta[0])
            block = hankel1(order[None], k[:, None, None]*distance)*np.exp(1j*order*phase)
            matrix[:, j*width:(j+1)*width, ell*width:(ell+1)*width] = block
    return matrix


def transfer(frequencies, centers, background, config, modes=16, quadrature=3, both=True):
    """Return superposition and coupled responses for every source/receiver pair.

    b = T(a + P b). With s=sqrt(abs(T)), solve for z=b/s to
    balance high-order multipoles; never delete high-order terms to hide issues.
    Inter-cylinder P is lossless, retaining pilot-v1's local scattering model;
    loss is applied only on source-to-cluster and cluster-to-receiver paths.
    """
    f = np.atleast_1d(frequencies)
    positive, boundary_residual = pilot.elastic_modes(f, 50e-6, background, config["materials"]["hap"], modes)
    coeff = np.tile(positive[:, np.abs(np.arange(-modes, modes+1))], (1, len(centers)))
    inc, out = arc_propagators(f, centers, background, config, modes, quadrature)
    incident = 0.25j*inc
    first = coeff[:, :, None]*incident
    single_response = np.einsum("far,fat->frt", out, first, optimize=True)
    if not both or len(centers) == 1:
        return single_response, single_response, {"single_boundary_residual": boundary_residual, "coupled_residual": 0.0}
    p = translation(f, centers, background, modes)
    scale = np.sqrt(np.maximum(np.abs(coeff), 1e-300))
    matrix = -(coeff/scale)[:, :, None]*p*scale[:, None, :]
    indices = np.arange(matrix.shape[1])
    matrix[:, indices, indices] += 1
    rhs = (coeff/scale)[:, :, None]*incident
    z = np.linalg.solve(matrix, rhs)
    residual = float(np.linalg.norm(np.einsum("fab,fbt->fat", matrix, z)-rhs)/np.linalg.norm(rhs))
    scattered = z*scale[:, :, None]
    coupled_response = np.einsum("far,fat->frt", out, scattered, optimize=True)
    if not np.all(np.isfinite(coupled_response)):
        raise FloatingPointError("Nonfinite coupled response")
    return single_response, coupled_response, {"single_boundary_residual": boundary_residual, "coupled_residual": residual}


def cluster_rf(config, fc, centers, modes=16, quadrature=3, sample_rate=None, samples=None,
               background=None, both=True, check_budget=None):
    f, pulse, mask = pilot.pulse_spectrum(config, fc, sample_rate, samples)
    indices = np.flatnonzero(mask)
    count = samples or config["samples"]
    channels = config["full_elements"]
    first_spectrum = np.zeros((len(f), channels, channels), complex)
    coupled_spectrum = np.zeros_like(first_spectrum)
    worst = 0.0
    bg = background or config["backgrounds"]["glandular"]
    started = time.perf_counter()
    for start in range(0, len(indices), 16):
        if check_budget:
            check_budget()
        subset = indices[start:start+16]
        first, coupled, info = transfer(f[subset], centers, bg, config, modes, quadrature, both)
        first_spectrum[subset] = pulse[subset, None, None]*first.conj()
        coupled_spectrum[subset] = pulse[subset, None, None]*coupled.conj()
        worst = max(worst, info["coupled_residual"])
    return (np.fft.irfft(first_spectrum, n=count, axis=0),
            np.fft.irfft(coupled_spectrum, n=count, axis=0),
            {"wall_seconds": time.perf_counter()-started, "coupled_residual": worst})


def channel_mask(config, limited=False):
    count = config["full_elements"]
    mask = ~np.eye(count, dtype=bool)
    if limited:
        keep = np.arange(count) < count//2
        mask &= keep[:, None] & keep[None, :]
    return mask


def relative_error(candidate, reference, mask):
    return float(np.linalg.norm((candidate-reference)[:, mask])/np.linalg.norm(reference[:, mask]))


def gate():
    folder = ROOT / "results/pilot-v2"
    freeze = json.loads((folder/"freeze.json").read_text())
    for p, expected in freeze["files"].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen specification changed: {p}")
    config = pilot.load_config()
    started = time.perf_counter()
    records = []

    def budget():
        if time.perf_counter()-started > 3500:
            raise TimeoutError("Physics CPU ceiling reached; preserving time for report")
        if datetime.now(timezone.utc) > datetime.fromisoformat(freeze["deadline_jst"]):
            raise TimeoutError("Calendar deadline reached")

    status = "RUNNING"
    try:
        for pattern in ("line", "ring"):
            for fc in (1000000, 3000000):
                for shift in ((0.0, 0.0), (0.002, 0.0)):
                    budget()
                    centers = positions(pattern, shift)
                    first, coupled, info = cluster_rf(config, fc, centers, check_budget=budget)
                    row = {"pattern": pattern, "fc_hz": fc, "shift_x_m": shift[0], **info}
                    for limited in (False, True):
                        row["limited_error" if limited else "full_error"] = relative_error(first, coupled, channel_mask(config, limited))
                    records.append(row)
                    (folder/"physics-gate-progress.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                    print(json.dumps(row), flush=True)
        status = "COMPLETE"
    except (TimeoutError, MemoryError, np.linalg.LinAlgError) as error:
        status = "TECHNICAL_HOLD"
        (folder/"physics-error.txt").write_text(str(error), encoding="utf-8")
    summary = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
        "wall_seconds": time.perf_counter()-started, "gpu_seconds": 0,
        "records": records, "tolerance": 0.05,
        "all_superposition_pass": len(records) == 8 and all(max(r["full_error"],r["limited_error"]) <= 0.05 for r in records),
        "expected_datasets": {"phantoms": 480, "positive_phantoms": 320, "frequencies_each": 2},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (folder/"physics-gate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true", required=True)
    parser.parse_args()
    gate()
