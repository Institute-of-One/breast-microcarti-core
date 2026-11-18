"""Conditional SKE/BKE pilot; not a validated breast imaging simulator.

No external services. Elastic-cylinder equations are derived from boundary
conditions, not translated from third-party code. See docs/SPECIFICATION.md.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.special import hankel1, h1vp, jv, jvp, ndtr, ndtri
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent


def load_config():
    return json.loads((ROOT / "configs/pilot-v1.json").read_text(encoding="utf-8"))


def elastic_speeds(material):
    rho, young, nu = (material[k] for k in ("rho", "young_pa", "poisson"))
    if rho <= 0 or young <= 0 or not -1 < nu < 0.5:
        raise ValueError("Invalid isotropic elastic material")
    mu = young / (2 * (1 + nu))
    lam = young * nu / ((1 + nu) * (1 - 2 * nu))
    return math.sqrt((lam + 2 * mu) / rho), math.sqrt(mu / rho), lam, mu


def elastic_modes(frequencies, radius, background, material, modes=16):
    """Scattered A_n per incident J_n, using exp(-i omega t).

    Solid potentials are (a^2/mu)*[B J_n(kL r) cos(n theta),
    C J_n(kT r) sin(n theta)], u=grad(phi)+curl(psi e_z).
    Matrix rows enforce normal displacement, normal stress, shear stress.
    Column/row scaling avoids ill-conditioned powers at small ka.
    """
    freq = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if np.any(freq <= 0) or radius <= 0:
        raise ValueError("Positive frequencies and radius required")
    cl, ct, lam, mu = elastic_speeds(material)
    omega = 2 * np.pi * freq
    x0, xl, xt = omega * radius / background["c"], omega * radius / cl, omega * radius / ct
    mass = background["rho"] * omega**2 * radius**2 / mu
    ans = np.empty((len(freq), modes + 1), dtype=complex)
    max_residual = 0.0
    for n in range(modes + 1):
        jl, jt = jv(n, xl), jv(n, xt)
        ul, ut = xl * jvp(n, xl), n * jt
        rr_l = -(lam / mu) * xl**2 * jl + 2 * xl**2 * jvp(n, xl, 2)
        rr_t = 2 * n * (xt * jvp(n, xt) - jt)
        rt_l = -2 * n * (xl * jvp(n, xl) - jl)
        rt_t = -n**2 * jt - xt**2 * jvp(n, xt, 2) + xt * jvp(n, xt)
        count = 2 if n == 0 else 3
        mat = np.zeros((len(freq), count, count), complex)
        rhs = np.zeros((len(freq), count), complex)
        mat[:, 0, 0] = x0 * h1vp(n, x0)
        mat[:, 0, 1] = -mass * ul
        mat[:, 1, 0] = hankel1(n, x0)
        mat[:, 1, 1] = rr_l
        rhs[:, 0] = -x0 * jvp(n, x0)
        rhs[:, 1] = -jv(n, x0)
        if n:
            mat[:, 0, 2] = -mass * ut
            mat[:, 1, 2] = rr_t
            mat[:, 2, 1] = rt_l
            mat[:, 2, 2] = rt_t
        col = np.max(np.abs(mat), axis=1)
        scaled = mat / col[:, None, :]
        row = np.maximum(np.max(np.abs(scaled), axis=2), np.abs(rhs))
        scaled = scaled / row[:, :, None]
        target = rhs / row
        solution = np.linalg.solve(scaled, target[..., None])[..., 0]
        residual = np.max(np.abs(np.einsum("fij,fj->fi", scaled, solution) - target))
        max_residual = max(max_residual, float(residual))
        ans[:, n] = solution[:, 0] / col[:, 0]
    if not np.all(np.isfinite(ans)):
        raise FloatingPointError("Nonfinite cylinder coefficients")
    return ans, max_residual


def fluid_modes(frequencies, radius, background, density, speed, modes=16):
    """Independent closed form for a penetrable fluid cylinder."""
    freq = np.atleast_1d(frequencies)
    k0, k1 = 2 * np.pi * freq / background["c"], 2 * np.pi * freq / speed
    result = []
    for n in range(modes + 1):
        x, y = k0 * radius, k1 * radius
        num = k1 / density * jvp(n, y) * jv(n, x) - k0 / background["rho"] * jvp(n, x) * jv(n, y)
        den = k0 / background["rho"] * h1vp(n, x) * jv(n, y) - k1 / density * jvp(n, y) * hankel1(n, x)
        result.append(num / den)
    return np.stack(result, axis=1)


def channel_weights(elements, limited=False):
    """Rotationally equivalent channel differences and their multiplicities."""
    angles = 2 * np.pi * np.arange(1, elements) / elements
    if limited:
        m = elements // 2
        offset = np.minimum(np.arange(1, elements), elements - np.arange(1, elements))
        weights = np.maximum(m - offset, 0)
    else:
        weights = np.full(elements - 1, elements)
    return angles, weights


def pulse_spectrum(config, center_frequency, sample_rate=None, samples=None):
    fs = sample_rate or config["sample_rate_hz"]
    count = samples or config["samples"]
    t = np.arange(count) / fs
    sigma_f = config["fractional_bandwidth"] * center_frequency / (2 * math.sqrt(2 * math.log(2)))
    sigma_t = 1 / (2 * np.pi * sigma_f)
    tau = t - config["pulse_t0_s"]
    pulse = np.exp(-0.5 * (tau / sigma_t)**2) * np.cos(2 * np.pi * center_frequency * tau)
    pulse /= np.max(np.abs(pulse))
    frequencies = np.fft.rfftfreq(count, 1 / fs)
    mask = (frequencies >= config["min_frequency_hz"]) & (frequencies <= config["max_frequency_hz"])
    mask[0] = mask[-1] = False
    return frequencies, np.fft.rfft(pulse), mask


def scatter_rf(config, center_frequency, diameter_um, background_name="fat", material_name="hap",
               modes=None, sample_rate=None, samples=None, alpha_scale=1.0, fluid=False):
    bg = config["backgrounds"][background_name]
    material = config["materials"][material_name]
    count = samples or config["samples"]
    f, pulse, mask = pulse_spectrum(config, center_frequency, sample_rate, samples)
    freq = f[mask]
    max_mode = modes or config["modes"]
    radius = diameter_um * 0.5e-6
    if fluid:
        cl, _, _, _ = elastic_speeds(material)
        coeff = fluid_modes(freq, radius, bg, material["rho"], cl, max_mode)
        residual = None
    else:
        coeff, residual = elastic_modes(freq, radius, bg, material, max_mode)
    n = np.arange(max_mode + 1)
    k = 2 * np.pi * freq / bg["c"]
    ring = config["radius_m"]
    # Uniform arcs are integrated analytically for each angular harmonic.
    aperture = np.sinc(n * config["element_width_m"] / (2 * ring * np.pi))**2
    neumann = np.where(n == 0, 1.0, 2.0)
    angles, _ = channel_weights(config["full_elements"])
    radial = hankel1(n[None, :], k[:, None] * ring)**2
    pressure = 0.25j * ((coeff * radial * aperture * neumann) @ np.cos(n[:, None] * angles))
    attenuation = bg["alpha0"] * alpha_scale * (freq / 1e6)**config["attenuation_power"]
    pressure *= np.exp(-attenuation[:, None] * 2 * ring)
    spectrum = np.zeros((len(f), len(angles)), complex)
    # numpy uses exp(+i omega t) synthesis; conjugate the e^-iwt response.
    spectrum[mask] = pulse[mask, None] * pressure.conj()
    rf = np.fft.irfft(spectrum, n=count, axis=0)
    passive_error = None if fluid else float(np.max(np.abs(np.abs(1 + 2 * coeff) - 1)))
    return rf, {"boundary_residual": residual, "passivity_error": passive_error}


def reference_rf(config):
    f, pulse, mask = pulse_spectrum(config, config["center_frequencies_hz"][0])
    bg = config["backgrounds"]["fat"]
    distance = 2 * config["radius_m"]
    response = 0.25j * hankel1(0, 2 * np.pi * f[mask] / bg["c"] * distance)
    response *= np.exp(-bg["alpha0"] * (f[mask] / 1e6)**config["attenuation_power"] * distance)
    spectrum = np.zeros_like(pulse)
    spectrum[mask] = pulse[mask] * response.conj()
    return np.fft.irfft(spectrum, n=config["samples"])


def rf_norm(rf, weights):
    return float(np.sqrt(np.sum(rf**2 * weights[None, :])))


def wilson(successes, total):
    z = float(ndtri(0.975))
    p = successes / total
    den = 1 + z*z / total
    center = (p + z*z / (2*total)) / den
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return [max(0.0, center-half), min(1.0, center+half)]


def empirical_auc(negative, positive):
    combined = np.r_[negative, positive]
    n0, n1 = len(negative), len(positive)
    ranks = rankdata(combined)
    return float((np.sum(ranks[n0:]) - n1*(n1+1)/2) / (n0*n1))


def observer_metrics(dprime, config, seed):
    n = config["replicates_per_class"]
    rng = np.random.default_rng(seed)
    # Separate noise draws for calibration, absent evaluation, present evaluation.
    calibration = rng.normal(size=config["threshold_replicates"])
    negative = rng.normal(size=n)
    positive = rng.normal(size=n) + dprime
    threshold = float(np.quantile(calibration, 0.95, method="higher"))
    fp, tp = int(np.sum(negative > threshold)), int(np.sum(positive > threshold))
    boot = [empirical_auc(rng.choice(negative, n), rng.choice(positive, n))
            for _ in range(config["bootstrap_replicates"])]
    return {
        "dprime": dprime, "auc_analytic": float(ndtr(dprime / math.sqrt(2))),
        "sensitivity_fpr05_analytic": float(ndtr(dprime - ndtri(0.95))),
        "auc_mc": empirical_auc(negative, positive),
        "auc_mc_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "threshold_mc": threshold, "fpr_mc": fp/n, "sensitivity_mc": tp/n,
        "fpr_mc_ci95": wilson(fp, n), "sensitivity_mc_ci95": wilson(tp, n),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    started = time.perf_counter()
    config = load_config()
    freeze = json.loads((ROOT / "results/freeze.json").read_text(encoding="utf-8"))
    for name, expected in freeze["files"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    deadline = datetime.fromisoformat(config["deadline_jst"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Use a new output directory; results are not overwritten")
    prior_seconds = 0.0
    for p in (ROOT / "results").glob("*/summary.json"):
        previous = json.loads(p.read_text(encoding="utf-8"))
        prior_seconds += previous["resource"]["wall_seconds"]

    def budget():
        elapsed = time.perf_counter() - started
        if datetime.now(timezone.utc) > deadline:
            raise TimeoutError("Calendar deadline reached")
        if elapsed > config["max_run_seconds"] or prior_seconds + elapsed > 3600 * config["total_device_hours_cap"]:
            raise TimeoutError("Frozen compute resource ceiling reached")

    ref = reference_rf(config)
    ref_rms = float(np.sqrt(np.mean(ref**2)))
    np.savez_compressed(output / "reference.npz", rf=ref)
    primary, sensitivities, convergence = [], [], []
    quality = {"max_boundary_residual": 0.0, "max_passivity_error": 0.0}
    waveforms = {}

    def update_quality(info):
        for a, b in (("max_boundary_residual", "boundary_residual"), ("max_passivity_error", "passivity_error")):
            if info[b] is not None:
                quality[a] = max(quality[a], info[b])

    def rows(rf, metadata, mc=False):
        result = []
        for limited in (False, True):
            _, weights = channel_weights(config["full_elements"], limited)
            signal_norm = rf_norm(rf, weights)
            for snr in config["snr_db"]:
                sigma = ref_rms * 10**(-snr/20)
                dp = signal_norm / sigma
                row = dict(metadata, view="limited" if limited else "full", snr_db=snr,
                           channels=int(weights.sum()), sigma_pa=sigma, signal_l2=signal_norm)
                if mc:
                    key = json.dumps(row, sort_keys=True).encode()
                    seed = config["seed"] + int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
                    row.update(observer_metrics(dp, config, seed))
                else:
                    row.update(dprime=dp, auc_analytic=float(ndtr(dp / math.sqrt(2))),
                               sensitivity_fpr05_analytic=float(ndtr(dp - ndtri(0.95))))
                result.append(row)
        return result

    for bg in config["backgrounds"]:
        for fc in config["center_frequencies_hz"]:
            for diameter in config["diameters_um"]:
                budget()
                rf, info = scatter_rf(config, fc, diameter, bg)
                update_quality(info)
                key = f"{bg}_{fc}_{diameter}"
                waveforms[key] = rf
                primary.extend(rows(rf, {"background": bg, "fc_hz": fc, "diameter_um": diameter, "material": "hap"}, mc=True))
                if diameter in (50, 500):
                    for label, kwargs in (
                        ("modes24", {"modes": 24}),
                        ("fs40MHz", {"sample_rate": 2*config["sample_rate_hz"], "samples": 2*config["samples"]}),
                        ("window409us", {"samples": 2*config["samples"]}),
                    ):
                        budget()
                        refined, extra = scatter_rf(config, fc, diameter, bg, **kwargs)
                        update_quality(extra)
                        _, w = channel_weights(config["full_elements"])
                        baseline_norm = rf_norm(rf, w)
                        if label == "fs40MHz":
                            compare = refined[::2]
                            refined_norm = rf_norm(refined, w) / math.sqrt(2)
                        elif label == "window409us":
                            compare = refined[:config["samples"]]
                            refined_norm = rf_norm(compare, w)
                        else:
                            compare, refined_norm = refined, rf_norm(refined, w)
                        change = abs(refined_norm / baseline_norm - 1)
                        waveform_change = rf_norm(compare-rf, w) / baseline_norm
                        convergence.append({"background": bg, "fc_hz": fc, "diameter_um": diameter,
                                            "refinement": label, "relative_norm_change": change,
                                            "relative_waveform_change": waveform_change,
                                            "pass": bool(max(change, waveform_change) <= config["convergence_relative_tolerance"])})
                if diameter in (100, 300):
                    for label, kwargs in (
                        ("calcium_oxalate", {"material_name": "calcium_oxalate"}),
                        ("fluid_hap", {"fluid": True}),
                        ("attenuation0.9", {"alpha_scale": 0.9}),
                        ("attenuation1.1", {"alpha_scale": 1.1}),
                    ):
                        budget()
                        alternative, extra = scatter_rf(config, fc, diameter, bg, **kwargs)
                        update_quality(extra)
                        sensitivities.extend(rows(alternative, {"background": bg, "fc_hz": fc,
                                                  "diameter_um": diameter, "variant": label}))
            print(f"Completed {bg} {fc/1e6:g} MHz", flush=True)
    assert len(primary) == 96
    assert len(convergence) == 24
    assert len(sensitivities) == 128
    np.savez_compressed(output / "signals.npz", **waveforms)
    for name, data in (("primary", primary), ("sensitivities", sensitivities), ("convergence", convergence)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        flat = [{k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in data]
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    elapsed = time.perf_counter() - started
    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "specification_hashes": freeze["files"],
        "code_sha256": sha256(Path(__file__)), "python": platform.python_version(),
        "package_versions": {x: importlib.metadata.version(x) for x in ("numpy", "scipy")},
        "reference_rms_pa": ref_rms, "counts": {"primary": len(primary), "sensitivities": len(sensitivities), "convergence": len(convergence)},
        "material_speeds_m_s": {k: elastic_speeds(v)[:2] for k, v in config["materials"].items()},
        "quality": quality, "all_convergence_pass": all(r["pass"] for r in convergence),
        "resource": {"wall_seconds": elapsed, "device_hours": elapsed/3600,
                     "prior_recorded_seconds": prior_seconds, "gpu_hours": 0, "blas_threads": 1},
        "independent_ai_review": "NOT_RUN_NO_TRANSMISSION_AUTHORIZATION",
        "independent_physics_validation": "NOT_RUN",
        "image_reconstruction": "NOT_RUN_PHYSICS_AND_NOVELTY_GATE_FIRST",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New run directory")
    args = parser.parse_args()
    run(args.output)
