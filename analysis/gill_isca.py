#!/usr/bin/env python3
"""Gill response in PyLBM-Isca: process validation + CPU-vs-GPU parity.

Mirrors LBM_GPU_GMD/analysis/a8_gill_decay.py but on the Isca/GFDL core and on
BOTH backends.  A resting, isothermal basic state is forced by a steady
equatorial Gaussian heating; we measure the Kelvin (east) and Rossby (west)
e-folding lengths of the low-level equatorial zonal wind (analytic ratio 3),
and verify that the CPU and GPU integrations agree to float64 round-off.

Writes analysis/gill_isca.json.
"""
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ISCA_ROOT = os.path.dirname(HERE)
sys.path.insert(0, ISCA_ROOT)

from pylbm_isca import Config, LBM, Spectral, VerticalGrid          # noqa: E402
from pylbm_isca.forcing import SteadyForcing                        # noqa: E402
from pylbm_isca.backend import asnumpy                              # noqa: E402

A_EARTH = 6.371e6


def resting_state(sp, vg, t0=250.0):
    shape = (sp.nlat, sp.nlon, vg.kmax)
    return types.SimpleNamespace(
        u=np.zeros(shape), v=np.zeros(shape),
        t=np.full(shape, t0), q=np.zeros(shape),
        ps=np.full((sp.nlat, sp.nlon), 1000.0))


def equatorial_heating(sp, vg, lon0=180.0, amp_k_per_day=1.0):
    frc = SteadyForcing.zeros(sp.nlat, sp.nlon, vg.kmax)
    lon = asnumpy(sp.lon_deg)
    lat = asnumpy(sp.lat_deg)
    sig = asnumpy(vg.sig)
    dl = (lon[None, :] - lon0 + 180.0) % 360.0 - 180.0
    hxy = np.exp(-(dl / 20.0) ** 2 - (lat[:, None] / 10.0) ** 2)
    vz = np.sin(np.pi * (1.0 - sig)) * (sig > 0.1)
    frc.t = hxy[:, :, None] * vz[None, None, :] * (amp_k_per_day / 86400.0)
    return frc


def efold_length(dist_km, amp, lo, hi):
    m = (dist_km >= lo) & (dist_km <= hi) & (amp > 0)
    if m.sum() < 4:
        return np.nan, np.nan
    x = dist_km[m]
    y = np.log(amp[m])
    slope, intercept = np.polyfit(x, y, 1)
    L = -1.0 / slope if slope < 0 else np.nan
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return L, r2


def run(device, lon0=180.0):
    sp, vg = Spectral(device=device), VerticalGrid(device=device)
    m = LBM(resting_state(sp, vg), sp, vg, Config(ndays=6.0))
    m.run(steady_forcing=equatorial_heating(sp, vg, lon0=lon0), verbose=False)
    out = {k: asnumpy(v) for k, v in m.diagnostics().items()}
    return out, asnumpy(sp.lat_deg), asnumpy(sp.lon_deg), asnumpy(vg.sig)


def gill_scales(hist, lat, lon, sig, lon0=180.0):
    klo = int(np.argmin(np.abs(sig - 0.85)))
    eq = int(np.argmin(np.abs(lat)))
    u = hist["u"][eq, :, klo]
    dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
    order = np.argsort(dlon)
    dlon, u = dlon[order], u[order]
    dist_km = np.radians(dlon) * A_EARTH / 1000.0
    east, west = dlon > 0, dlon < 0
    iE = np.nanargmax(np.abs(np.where(east, u, np.nan)))
    iW = np.nanargmax(np.abs(np.where(west, u, np.nan)))
    L_K, r2_K = efold_length(dist_km, np.where(east, np.abs(u), 0.0),
                             lo=dist_km[iE], hi=np.radians(150) * A_EARTH / 1e3)
    L_R, r2_R = efold_length(-dist_km, np.where(west, np.abs(u), 0.0),
                             lo=-dist_km[iW], hi=np.radians(85) * A_EARTH / 1e3)
    return {
        "kelvin_peak_lon_offset_deg": float(dlon[iE]),
        "kelvin_peak_u_ms": float(u[iE]),
        "rossby_peak_lon_offset_deg": float(dlon[iW]),
        "rossby_peak_u_ms": float(u[iW]),
        "L_K_kelvin_efold_km": float(L_K),
        "L_K_fit_r2": float(r2_K),
        "L_R_rossby_efold_km": float(L_R),
        "L_R_fit_r2": float(r2_R),
        "ratio_LK_over_LR": float(L_K / L_R),
    }


def parity(hg, hc):
    """CPU-vs-GPU relative RMS for each history field."""
    out = {}
    for k in ("psi", "chi", "u", "v", "t", "z", "w"):
        a = np.asarray(hc[k], dtype=np.float64)
        b = np.asarray(hg[k], dtype=np.float64)
        scale = np.sqrt(np.mean(a ** 2))
        rms = np.sqrt(np.mean((a - b) ** 2))
        out[k] = {"rel_rms": float(rms / scale) if scale else np.nan,
                  "max_abs_diff": float(np.max(np.abs(a - b)))}
    return out


def main():
    from pylbm_isca import gpu_available
    hc, lat, lon, sig = run("cpu")
    scales_cpu = gill_scales(hc, lat, lon, sig)
    out = {
        "experiment": "Gill, equatorial Gaussian heating 1 K/day, resting "
                      "isothermal 250 K, Isca core T42L20, day 6, sigma~0.85",
        "analytic_ratio_LK_over_LR": 3.0,
        "cpu": scales_cpu,
    }
    if gpu_available():
        hg, *_ = run("gpu")
        out["gpu"] = gill_scales(hg, lat, lon, sig)
        out["cpu_gpu_parity"] = parity(hg, hc)
    else:
        out["gpu"] = "unavailable"

    path = os.path.join(HERE, "gill_isca.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
