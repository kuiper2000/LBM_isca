#!/usr/bin/env python3
"""A9 on PyLBM-Isca: ensemble forcing-uncertainty forecast (roadmap B2, Fig 8).

Isca-core counterpart of LBM_GPU_GMD/analysis/a9_ensemble_uncertainty.py.

All members share one basic state (JFM 1979) and a *similar* MJO phase-1 heating
that differs slightly between members (small random jitter in convective
amplitude and longitude).  Running the whole ensemble in ONE batched GPU
integration, we ask how robust the extratropical Rossby-wave teleconnection is to
that uncertainty.

Outputs:
  <manuscript>/figures/ensemble_forecast_uncertainty.pdf (+ .png)
  <manuscript>/analysis/a9_ensemble_uncertainty.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ISCA_ROOT = os.path.dirname(HERE)                       # .../LBM_isca
REPO_ROOT = os.path.dirname(ISCA_ROOT)
DATA_ROOT = os.path.join(ISCA_ROOT, "data")            # realistic-case inputs
MANU = os.path.join(REPO_ROOT, "LBM_GPU_GMD")           # manuscript tree
sys.path.insert(0, ISCA_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from pylbm_isca import Config, Spectral, VerticalGrid, IMAX, JMAX, NMAX  # noqa: E402
from pylbm_isca.backend import asnumpy                               # noqa: E402
from pylbm_isca.basicstate import load_basic_state                   # noqa: E402
from pylbm_isca.forcing import MJOForcing                            # noqa: E402
from pylbm_isca.ensemble import EnsembleForcing, make_ensemble_model # noqa: E402
from pylbm_isca.regrid import detect_grid, regrid_basic_state        # noqa: E402

FIGDIR = os.path.join(MANU, "figures")
ANADIR = os.path.join(MANU, "analysis")
os.makedirs(FIGDIR, exist_ok=True)

BS_FILE = os.path.join(DATA_ROOT, "bs.1979.bin")
FRC_FILE = os.path.join(DATA_ROOT, "frc.phase1.bin")


def load_bs(device):
    sp_plain = Spectral(device=device)
    vg = VerticalGrid(device=device)
    bs_nlon, bs_nlat = detect_grid(BS_FILE)
    regrid = (bs_nlon, bs_nlat) != (sp_plain.nlon, sp_plain.nlat)
    bs_sp = (Spectral(nlon=bs_nlon, nlat=bs_nlat, nmax=(bs_nlon - 1) // 3,
                      device=device) if regrid else sp_plain)
    bs = load_basic_state(BS_FILE, bs_sp, vg, kmo=1, navg=3)
    if regrid:
        bs = regrid_basic_state(bs, bs_sp, sp_plain)
    return bs


def representative_heating(device):
    f_nlon, f_nlat = detect_grid(FRC_FILE)
    mjo = MJOForcing(FRC_FILE, nlon=f_nlon, nlat=f_nlat,
                     ndays=_ndays(FRC_FILE, f_nlon, f_nlat))
    gtf = np.asarray(mjo.gtf)                     # (ndays, nlat, nlon, kmax)
    rms = np.sqrt((gtf ** 2).mean(axis=(1, 2, 3)))
    active = np.argsort(rms)[::-1][:20]           # 20 most active days
    base = gtf[active].mean(axis=0)               # (nlat, nlon, kmax)
    return base, f_nlon, f_nlat


def _ndays(path, nlon, nlat):
    from pylbm_isca.forcing import read_fortran_records
    from pylbm_isca.constants import KMAX
    rec = read_fortran_records(path, nlon, nlat)
    return rec.shape[0] // KMAX


def build_members(base, nmem, dlon_deg, amp_frac, lon_jit_deg, seed=12345):
    rng = np.random.default_rng(seed)
    nlat, nlon, kmax = base.shape
    out = np.empty((nmem, nlat, nlon, kmax))
    amps = np.ones(nmem)
    shifts = np.zeros(nmem, dtype=int)
    for e in range(nmem):
        if e == 0:
            out[e] = base
            continue
        a = 1.0 + amp_frac * rng.standard_normal()
        s_deg = lon_jit_deg * rng.standard_normal()
        s = int(round(s_deg / dlon_deg))
        amps[e] = a
        shifts[e] = s
        out[e] = a * np.roll(base, s, axis=1)
    return out, amps, shifts


class PsiAccumulator:
    def __init__(self, k, lo, hi):
        self.k, self.lo, self.hi = k, lo, hi
        self.sum = None
        self.n = 0

    def add(self, day, fields):
        if self.lo <= day <= self.hi:
            psi = np.asarray(fields["psi"])[..., self.k]     # (E, nlat, nlon)
            self.sum = psi.copy() if self.sum is None else self.sum + psi
            self.n += 1

    def close(self):
        pass

    @property
    def mean(self):
        return self.sum / self.n


def main():
    device = "gpu"
    nmem = int(os.environ.get("A9_NMEM", "50"))
    days = float(os.environ.get("A9_DAYS", "30"))
    avg_lo, avg_hi = 15, 30

    print(f"A9 forecast-uncertainty ensemble (PyLBM-Isca): {nmem} members, "
          f"{days:.0f} days, device={device}")
    bs = load_bs(device)
    base, f_nlon, f_nlat = representative_heating(device)
    peak = np.abs(base).max() * 86400
    print(f"base heating peak {peak:.2f} K/day on {f_nlon}x{f_nlat} grid")

    if (f_nlon, f_nlat) != (IMAX, JMAX):
        raise SystemExit(f"forcing native {f_nlon}x{f_nlat} != model "
                         f"{IMAX}x{JMAX}; add spectral regrid if this triggers")

    dlon_deg = 360.0 / IMAX
    members, amps, shifts = build_members(
        base, nmem, dlon_deg, amp_frac=0.10, lon_jit_deg=5.0)
    print(f"amplitude jitter sd=10%, longitude jitter sd=5 deg "
          f"(amps {amps.min():.2f}-{amps.max():.2f})")

    cfg = Config(ndays=days, out_interval_days=1.0, time_average=True)
    model, sp_batch, vg = make_ensemble_model(bs, nmem, cfg, device=device)

    k200 = int(np.argmin(np.abs(asnumpy(vg.sig) - 0.175)))
    acc = PsiAccumulator(k200, avg_lo, avg_hi)
    forcing = EnsembleForcing(members, sp_batch.xp)

    t0 = time.time()
    model.run(mjo_forcing=forcing, writer=acc, verbose=True)
    wall = time.time() - t0
    print(f"batched run wall {wall:.1f}s for {nmem} members "
          f"({wall/nmem:.2f}s/member-equivalent)")

    psi = acc.mean                                 # (E, nlat, nlon), days 15-30
    psi_eddy = psi - psi.mean(axis=2, keepdims=True)
    ens_mean = psi_eddy.mean(axis=0)
    ens_std = psi_eddy.std(axis=0)
    control = psi_eddy[0]

    lat = asnumpy(sp_batch.lat_deg)
    lon = asnumpy(sp_batch.lon_deg)

    _plot(lon, lat, ens_mean, ens_std, control, nmem)

    def box_stat(field, lon_lo, lon_hi, lat_lo, lat_hi):
        jm = (lat >= lat_lo) & (lat <= lat_hi)
        im = (lon >= lon_lo) & (lon <= lon_hi)
        sub = field[np.ix_(jm, im)]
        return float(np.mean(sub))

    gm = np.sqrt((ens_mean ** 2).mean())
    gs = np.sqrt((ens_std ** 2).mean())
    out = {
        "experiment": "A9 forcing-uncertainty ensemble (PyLBM-Isca), JFM 1979 "
                      "basic state, MJO phase-1 representative heating, 200 hPa "
                      "psi days 15-30",
        "model": "PyLBM-Isca",
        "nmem": nmem,
        "days": days,
        "wall_seconds_batched": wall,
        "seconds_per_member_equiv": wall / nmem,
        "amplitude_jitter_sd": 0.10,
        "longitude_jitter_sd_deg": 5.0,
        "rms_ensemble_mean_psi": gm,
        "rms_intermember_std_psi": gs,
        "global_signal_to_noise": gm / gs,
        "wpac_high_meanpsi": box_stat(ens_mean, 120, 160, 10, 35),
        "wpac_high_std": box_stat(ens_std, 120, 160, 10, 35),
        "namer_meanpsi": box_stat(ens_mean, 220, 260, 40, 65),
        "namer_std": box_stat(ens_std, 220, 260, 40, 65),
    }
    path = os.path.join(ANADIR, "a9_ensemble_uncertainty.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


def _plot(lon, lat, ens_mean, ens_std, control, nmem):
    order = np.argsort(lat)[::-1]
    latp = lat[order]
    M = ens_mean[order] / 1e6
    S = ens_std[order] / 1e6
    vmax = np.nanpercentile(np.abs(M), 99)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    cf = axes[0].pcolormesh(lon, latp, M, cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax, shading="auto")
    axes[0].set_title(f"(a) Ensemble-mean 200 hPa streamfunction "
                      f"(N={nmem}), days 15-30")
    fig.colorbar(cf, ax=axes[0], label=r"$\psi'$ ($10^6$ m$^2$ s$^{-1}$)")
    smax = np.nanpercentile(S, 99)
    cf2 = axes[1].pcolormesh(lon, latp, S, cmap="viridis",
                             vmin=0, vmax=smax, shading="auto")
    axes[1].set_title("(b) Inter-member spread (std) from heating uncertainty")
    fig.colorbar(cf2, ax=axes[1], label=r"std $\psi'$ ($10^6$ m$^2$ s$^{-1}$)")
    for ax in axes:
        for (x0, x1, y0, y1, c) in [(120, 160, 10, 35, "k"),
                                    (220, 260, 40, 65, "magenta")]:
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                    color=c, lw=1.2, ls="--")
        ax.set_ylabel("latitude")
        ax.set_ylim(-80, 80)
    axes[1].set_xlabel("longitude")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"ensemble_forecast_uncertainty.{ext}"),
                    dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
