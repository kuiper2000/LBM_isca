#!/usr/bin/env python3
# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""Broader realistic validation: summer (JJA) basic state (GMD revision item 5).

The realistic MJO-phase-1 experiment of the paper is repeated on a summer (JJA)
climatological basic state, with the *same* MJO phase-1 heating and identical
numerical settings, and compared with the winter (DJF 1979) case. This shows the
model produces a physically distinct, sensible teleconnection on a second
climatology -- useful behaviour beyond a single winter basic state.

Comparison conventions (stated explicitly):
  * field     : 200 hPa (sigma ~ 0.175) streamfunction response, psi200.
  * window    : days 15-30 mean (the established window, Sect. 5 / item 4).
  * weights   : cos(latitude) area weights.
  * masks     : global, and the Northern-Hemisphere extratropics (20-80 N).
  * metrics   : area-weighted pattern correlation r(JJA, DJF), relative RMS
                difference ||JJA-DJF|| / ||DJF||, and amplitude ratio
                RMS(JJA)/RMS(DJF).  Full fields are used (zonal mean retained);
                the response is a heating-driven anomaly about the basic state,
                so no further anomaly subtraction is applied.

Outputs: analysis/summer_validation.{pdf,png,json} and a printed table.
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


SIG_LOW = 0.175      # diagnostic level ~200 hPa (upper-tropospheric teleconnection)
PTOP_DATA = 100.0    # hPa; input reliable only up to here (see cap_above_100)


class Collector:
    def __init__(self, klow):
        self.klow = klow
        self.days, self.psilow = [], []

    def add(self, day, fields):
        self.psilow.append(asnumpy(fields["psi"])[..., self.klow].copy())
        self.days.append(float(day))

    def close(self):
        pass


def cap_top(bs, sp, vg):
    """Hold the basic state constant (isothermally) above PTOP_DATA.

    The JJA climatology was downloaded only up to 100 hPa, so the basic state
    above that is an extrapolation; here each column's fields are held at their
    highest reliable (>=100 hPa) value upward.  Those levels lie in the model's
    top sponge and are not analysed (the diagnostic level is ~850 hPa); the cap
    simply keeps the semi-implicit solver well posed.  Applied to the summer
    state only; the DJF state has reliable data to 10 hPa and is left untouched.
    """
    sig = asnumpy(vg.sig)
    ps = asnumpy(bs.ps)                                  # hPa
    p = sig[None, None, :] * ps[..., None]               # (nlat,nlon,kmax)
    kcap = np.maximum((p >= PTOP_DATA).sum(axis=-1) - 1, 0)   # last reliable level
    for name in ("t", "u", "v", "q"):
        a = asnumpy(getattr(bs, name))
        acap = np.take_along_axis(a, kcap[..., None], axis=-1)
        setattr(bs, name, np.where(p >= PTOP_DATA, a, acap))   # keep host (numpy)
    return bs


def run_case(bs_path, sp, vg, mjo_path, ndays, delt, cap_above_100=False):
    bs = load_basic_state(bs_path, sp, vg, kmo=1, navg=3)
    if cap_above_100:
        tmin0 = float(asnumpy(bs.t.min()))
        bs = cap_top(bs, sp, vg)
        print(f"  [{os.path.basename(bs_path)}] basic state capped isothermally "
              f"above {PTOP_DATA:.0f} hPa (extrapolated top; was Tmin {tmin0:.0f} K)")
    cfg = Config(ndays=ndays, delt=delt, out_interval_days=1.0, time_average=True)
    m = LBM(bs, sp, vg, cfg)
    klow = int(np.argmin(np.abs(asnumpy(vg.sig) - SIG_LOW)))
    nt = int(round(86400.0 / delt))
    mjo = MJOForcing(mjo_path, nlev=vg.kmax, ndays=73, nt=nt, spinup=435)
    col = Collector(klow)
    m.run(mjo_forcing=mjo, writer=col, verbose=False)
    d = np.asarray(col.days)
    idx = np.where((d > 15) & (d <= 30))[0]
    psi = np.mean(np.stack([col.psilow[i] for i in idx]), axis=0)
    jet = asnumpy(bs.u)[..., klow]            # 200 hPa zonal wind of the basic state
    return psi, jet


def wstats(a, b, w2d):
    """Area-weighted amplitude(a,b), pattern corr, rel-RMS diff, amp ratio."""
    wa = w2d
    rms = lambda x: np.sqrt(np.average(x ** 2, weights=wa))
    am = lambda x: x - np.average(x, weights=wa)
    corr = np.sum(wa * am(a) * am(b)) / np.sqrt(np.sum(wa * am(a) ** 2) * np.sum(wa * am(b) ** 2))
    relrms = rms(a - b) / (rms(b) or 1.0)
    return dict(rms_a=float(rms(a)), rms_b=float(rms(b)), corr=float(corr),
                rel_rms=float(relrms), amp_ratio=float(rms(a) / (rms(b) or 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--djf", default=os.path.join(ROOT, "data", "bs.1979.bin"))
    ap.add_argument("--jja", default=os.path.join(ROOT, "data", "bs.jja.bin"))
    ap.add_argument("--mjo-forcing", default=os.path.join(ROOT, "data", "frc.phase1.bin"))
    ap.add_argument("--ndays", type=int, default=35)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = args.device
    if dev == "auto":
        dev = "gpu" if gpu_available() else "cpu"
    sp = Spectral(device=dev); vg = VerticalGrid(device=dev)
    lat = np.degrees(np.arcsin(asnumpy(sp.mu)))
    coslat = np.sqrt(1.0 - asnumpy(sp.mu) ** 2)
    w_glob = np.broadcast_to(coslat[:, None], (sp.nlat, sp.nlon))
    nh = (lat >= 20) & (lat <= 80)
    sh = (lat <= -20) & (lat >= -80)
    w_nh = w_glob * nh[:, None]
    w_sh = w_glob * sh[:, None]
    print(f"Summer (JJA) vs winter (DJF) MJO phase-1 validation, backend={dev}")

    psi_djf, jet_djf = run_case(args.djf, sp, vg, args.mjo_forcing, args.ndays, 1200.0,
                                cap_above_100=False)   # DJF has reliable data to 10 hPa
    psi_jja, jet_jja = run_case(args.jja, sp, vg, args.mjo_forcing, args.ndays, 1200.0,
                                cap_above_100=True)    # JJA downloaded only to 100 hPa

    g = wstats(psi_jja, psi_djf, w_glob)
    n = wstats(psi_jja, psi_djf, w_nh)
    s_ = wstats(psi_jja, psi_djf, w_sh)
    print("\n200 hPa streamfunction response (days 15-30 mean), JJA vs DJF:")
    print(f"  {'mask':16s}{'RMS(DJF)':>12}{'RMS(JJA)':>12}{'amp ratio':>11}"
          f"{'pattern r':>11}{'rel RMS':>10}")
    for name, s in (("global", g), ("NH extratrop.", n), ("SH extratrop.", s_)):
        print(f"  {name:16s}{s['rms_b']:12.3e}{s['rms_a']:12.3e}"
              f"{s['amp_ratio']:11.2f}{s['corr']:+11.2f}{s['rel_rms']:10.2f}")
    # basic-state jet contrast (physical context)
    umax_djf = float(np.max(np.average(jet_djf, weights=w_glob, axis=1)))
    umax_jja = float(np.max(np.average(jet_jja, weights=w_glob, axis=1)))
    print(f"\n  200 hPa zonal-mean jet max: DJF {umax_djf:.1f} m/s, JJA {umax_jja:.1f} m/s"
          "  (weaker, poleward-shifted summer jet)")

    out = {"window": "days 15-30", "level": "sigma~0.175 (200 hPa)",
           "global": g, "nh_extratropics": n, "sh_extratropics": s_,
           "jet_max_djf": umax_djf, "jet_max_jja": umax_jja}
    with open(os.path.join(HERE, "summer_validation.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    _plot(sp, lat, psi_djf, psi_jja, g)
    return 0


def _plot(sp, lat, psi_djf, psi_jja, g):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                # pragma: no cover
        print(f"(skipping plot: {e})"); return
    lon = np.arange(sp.nlon) * 360.0 / sp.nlon
    diff = psi_jja - psi_djf
    vmax = max(np.abs(psi_djf).max(), np.abs(psi_jja).max()) or 1.0
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 9))
    for ax, fld, ttl, vm in ((axs[0], psi_djf, "DJF 1979 (winter)", vmax),
                             (axs[1], psi_jja, "JJA (summer)", vmax),
                             (axs[2], diff, f"JJA - DJF  (r={g['corr']:+.2f}, "
                              f"amp ratio {g['amp_ratio']:.2f})", vmax)):
        cf = ax.contourf(lon, lat, fld, levels=np.linspace(-vm, vm, 21),
                         cmap="RdBu_r", extend="both")
        ax.contour(lon, lat, fld, levels=np.linspace(-vm, vm, 21),
                   colors="k", linewidths=0.2)
        ax.set_ylim(-80, 80); ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("lat"); fig.colorbar(cf, ax=ax, shrink=0.85)
    axs[-1].set_xlabel("lon")
    fig.suptitle("MJO phase-1 200 hPa streamfunction response (days 15-30)", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"summer_validation.{ext}"), dpi=150)
    print(f"\nwrote {os.path.join(HERE, 'summer_validation.pdf')}")


if __name__ == "__main__":
    raise SystemExit(main())
