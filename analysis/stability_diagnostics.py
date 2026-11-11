#!/usr/bin/env python3
# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""Instability and averaging-window diagnostics (GMD revision item 4).

Two experiments on the realistic DJF 1979 basic state:

1. Unforced growth.  A small random perturbation is integrated with no forcing
   (strictly linear) at three time steps (dt = 1200, 600, 300 s) and with
   stronger horizontal diffusion.  The global wind norm is tracked and an
   e-folding growth rate is fit over a quasi-exponential window.  A growth rate
   that is independent of dt is a physical property of the basic state
   (baroclinic instability); a rate that changes with dt would indicate a
   numerical instability.  The stronger-diffusion run separates the expected
   sensitivity to resolution-dependent damping from any numerical blow-up.

2. Averaging window.  The forced MJO phase-1 response is averaged over days
   1-15, 15-30 and 30-45; pattern correlations and amplitudes between windows,
   together with the unforced growth curve, show whether the days 15-30 window
   used in the paper is materially contaminated by the growing mode.

Outputs: analysis/stability_growth.{pdf,png}, analysis/stability_windows.{pdf,png},
analysis/stability_diagnostics.json, and a printed summary.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pylbm_isca import Spectral, VerticalGrid, LBM, Config, gpu_available  # noqa: E402
from pylbm_isca.backend import asnumpy                                     # noqa: E402
from pylbm_isca.basicstate import load_basic_state                        # noqa: E402
from pylbm_isca.forcing import MJOForcing                                 # noqa: E402


class Collector:
    """A writer that records a daily global wind norm and the 200 hPa psi."""
    def __init__(self, k200):
        self.k200 = k200
        self.days, self.norm, self.psi200 = [], [], []

    def add(self, day, fields):
        u = asnumpy(fields["u"]); v = asnumpy(fields["v"])
        self.norm.append(float(np.sqrt(np.mean(u ** 2) + np.mean(v ** 2))))
        self.psi200.append(asnumpy(fields["psi"])[..., self.k200].copy())
        self.days.append(float(day))

    def close(self):
        pass


def efold_rate(days, norm, t0, t1):
    """Fit ln(norm) ~ slope*day over [t0,t1]; return (slope/day, e-folding days)."""
    d = np.asarray(days); n = np.asarray(norm)
    m = (d >= t0) & (d <= t1) & (n > 0)
    slope = np.polyfit(d[m], np.log(n[m]), 1)[0]
    return slope, (1.0 / slope if slope > 0 else np.inf)


def run_unforced(bs, sp, vg, ndays, delt, seed=1, **over):
    cfg = Config(ndays=ndays, delt=delt, out_interval_days=1.0,
                 time_average=True, strictly_linear=True, **over)
    m = LBM(bs, sp, vg, cfg)
    k200 = int(np.argmin(np.abs(asnumpy(vg.sig) - 0.175)))
    rng = np.random.default_rng(seed)
    t0 = asnumpy(sp.wave_to_grid(sp.grid_to_wave(
        rng.standard_normal((sp.nlat, sp.nlon, vg.kmax)))))
    m.set_initial_perturbation(t=0.1 * t0)             # ~0.1 K seed
    col = Collector(k200)
    m.run(writer=col, verbose=False)
    return col


def run_forced(bs, sp, vg, ndays, delt, mjo_path):
    cfg = Config(ndays=ndays, delt=delt, out_interval_days=1.0,
                 time_average=True, strictly_linear=False)
    m = LBM(bs, sp, vg, cfg)
    k200 = int(np.argmin(np.abs(asnumpy(vg.sig) - 0.175)))
    nt = int(round(86400.0 / delt))
    mjo = MJOForcing(mjo_path, nlev=vg.kmax, ndays=73, nt=nt, spinup=435)
    col = Collector(k200)
    m.run(mjo_forcing=mjo, writer=col, verbose=False)
    return col


def wmean(col, lo, hi):
    """Mean psi200 over days (lo, hi]."""
    d = np.asarray(col.days)
    idx = np.where((d > lo) & (d <= hi))[0]
    return np.mean(np.stack([col.psi200[i] for i in idx]), axis=0), idx


def pattern_corr(a, b, coslat):
    w = np.broadcast_to(coslat[:, None], a.shape)
    aw = a - np.average(a, weights=w); bw = b - np.average(b, weights=w)
    return float(np.sum(w * aw * bw) /
                 np.sqrt(np.sum(w * aw ** 2) * np.sum(w * bw ** 2)))


def rms(a, coslat):
    w = np.broadcast_to(coslat[:, None], a.shape)
    return float(np.sqrt(np.average(a ** 2, weights=w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic-state", default=os.path.join(ROOT, "data", "bs.1979.bin"))
    ap.add_argument("--mjo-forcing", default=os.path.join(ROOT, "data", "frc.phase1.bin"))
    ap.add_argument("--ngrow", type=int, default=48)
    ap.add_argument("--nforce", type=int, default=45)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = args.device
    if dev == "auto":
        dev = "gpu" if gpu_available() else "cpu"
    sp = Spectral(device=dev); vg = VerticalGrid(device=dev)
    bs = load_basic_state(args.basic_state, sp, vg, kmo=1, navg=3)
    coslat = np.sqrt(1.0 - asnumpy(sp.mu) ** 2)
    print(f"Stability diagnostics on realistic DJF 1979, backend={dev}")

    # --- 1. unforced growth, dt sweep + stronger diffusion ---------------
    configs = [("dt=1200 s", dict(delt=1200.0)),
               ("dt=600 s",  dict(delt=600.0)),
               ("dt=300 s",  dict(delt=300.0)),
               ("dt=1200 s, 2x hyperdiff", dict(delt=1200.0, hdif_tefold=1.0))]
    growth = {}
    tfit0, tfit1 = 12, args.ngrow
    print(f"\nUnforced growth (seed=1), e-folding fit over days {tfit0}-{tfit1}:")
    for name, over in configs:
        col = run_unforced(bs, sp, vg, args.ngrow, **over)
        slope, ef = efold_rate(col.days, col.norm, tfit0, tfit1)
        growth[name] = {"days": col.days, "norm": col.norm,
                        "slope_per_day": slope, "efold_days": ef}
        print(f"  {name:28s}  growth {slope:+.3f}/day  e-folding {ef:5.1f} d")

    base = growth["dt=1200 s"]["efold_days"]
    efs = [growth[n]["efold_days"] for n in ("dt=1200 s", "dt=600 s", "dt=300 s")]
    spread = (max(efs) - min(efs)) / base * 100
    print(f"\n  e-folding spread across dt (1200/600/300): {spread:.1f}%  "
          "(small -> physical, dt-converged)")

    # --- 2. forced run + averaging windows -------------------------------
    print(f"\nForced MJO phase-1 run ({args.nforce} days):")
    fcol = run_forced(bs, sp, vg, args.nforce, 1200.0, args.mjo_forcing)
    windows = [(0, 15), (15, 30), (30, 45)]
    means, amps = {}, {}
    for lo, hi in windows:
        if hi > args.nforce:
            continue
        mn, _ = wmean(fcol, lo, hi)
        means[(lo, hi)] = mn
        amps[(lo, hi)] = rms(mn, coslat)
    wk = list(means.keys())
    ref = means[(15, 30)] if (15, 30) in means else means[wk[0]]
    corrs = {k: pattern_corr(means[k], ref, coslat) for k in wk}   # each vs 15-30
    print("  window (days)    amplitude[m2/s]   pattern corr vs 15-30")
    for (lo, hi) in wk:
        print(f"   {lo:2d}-{hi:<2d}         {amps[(lo,hi)]:12.3e}        {corrs[(lo,hi)]:+.3f}")
    # contamination: growth of the unforced mode between mid-points of the windows
    g = growth["dt=1200 s"]
    print(f"  unforced e-folding is {base:.0f} d; over the 15 d from the 15-30 to "
          f"the 30-45 window the growing mode amplifies ~{np.exp(15*g['slope_per_day']):.2f}x")

    out = {"growth": {k: {kk: vv for kk, vv in v.items()} for k, v in growth.items()},
           "forced": {"days": fcol.days, "norm": fcol.norm,
                      "window_amp": {f"{lo}-{hi}": amps[(lo, hi)] for lo, hi in wk},
                      "window_corr_vs_1530": {f"{lo}-{hi}": corrs.get((lo, hi), 1.0)
                                              for lo, hi in wk}},
           "efold_spread_pct": spread}
    with open(os.path.join(HERE, "stability_diagnostics.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    _plots(sp, growth, fcol, means, amps, corrs, coslat, tfit0, tfit1)
    return 0


def _plots(sp, growth, fcol, means, amps, corrs, coslat, tfit0, tfit1):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                # pragma: no cover
        print(f"(skipping plots: {e})"); return

    # Fig A: growth curves
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, g in growth.items():
        ax.semilogy(g["days"], g["norm"], marker=".", ms=3, label=name)
    ax.axvspan(tfit0, tfit1, color="0.9", zorder=0)
    ax.set_xlabel("day"); ax.set_ylabel(r"wind norm $\sqrt{\langle u'^2+v'^2\rangle}$ [m/s]")
    ax.set_title("Unforced perturbation growth (realistic DJF 1979)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"stability_growth.{ext}"), dpi=150)

    # Fig B: window maps + norm vs time
    lon = np.arange(sp.nlon) * 360.0 / sp.nlon
    lat = np.degrees(np.arcsin(asnumpy(sp.mu)))
    wk = list(means.keys())
    fig = plt.figure(figsize=(10, 6))
    vmax = max(np.abs(means[k]).max() for k in wk) or 1.0
    for i, (lo, hi) in enumerate(wk):
        axm = fig.add_subplot(2, len(wk), i + 1)
        cf = axm.contourf(lon, lat, means[(lo, hi)],
                          levels=np.linspace(-vmax, vmax, 21), cmap="RdBu_r", extend="both")
        axm.set_ylim(-80, 80); axm.set_title(
            f"days {lo}-{hi}\nRMS={amps[(lo,hi)]:.2e}, r(15-30)={corrs.get((lo,hi),1.0):+.2f}",
            fontsize=9)
        axm.set_xlabel("lon"); axm.set_ylabel("lat")
    axn = fig.add_subplot(2, 1, 2)
    axn.semilogy(fcol.days, fcol.norm, "k-", label="forced response norm")
    axn.semilogy(growth["dt=1200 s"]["days"], growth["dt=1200 s"]["norm"],
                 "r--", label="unforced growing mode")
    for lo, hi in wk:
        axn.axvspan(lo, hi, alpha=0.08, color="C0")
    axn.set_xlabel("day"); axn.set_ylabel("wind norm [m/s]")
    axn.set_title("Forced response vs unforced growth; shaded = averaging windows")
    axn.legend(fontsize=8); axn.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"stability_windows.{ext}"), dpi=150)
    print(f"\nwrote {os.path.join(HERE, 'stability_growth.pdf')} and stability_windows.pdf")


if __name__ == "__main__":
    raise SystemExit(main())
