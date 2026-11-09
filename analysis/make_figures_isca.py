#!/usr/bin/env python3
"""Regenerate the three validation figures with PyLBM-Isca (roadmap B1, Figs 1-3).

Isca-core counterpart of LBM_GPU_GMD/analysis/make_figures.py (which used the
old Watanabe-Kimoto GPU port, ``lbm``).  Every figure is now produced by running
the linearised Isca core, ``lbm_isca``:

  1. gill_response.pdf          - Gill response to equatorial heating on a
                                  resting isothermal state (Sect. 4.1)
  2. solidbody_teleconnection.pdf - stationary Rossby-wave train from idealised
                                  heating on a solid-body-rotation state (Sect. 4.2)
  3. teleconnection_d15_30.pdf  - 200 hPa psi response to realistic MJO phase-1
                                  heating on the JFM 1979 NCEP basic state,
                                  days 15-30 (Sect. 4.3)

PDFs are written straight into the manuscript's figures/ directory so the paper
picks them up.  Embedded suptitles are dropped (the captions carry the
description) and fonts are embedded as TrueType (roadmap front-matter item).
"""

import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ISCA_ROOT = os.path.dirname(HERE)                       # .../LBM_isca
sys.path.insert(0, ISCA_ROOT)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42               # embed TrueType, not Type-3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from lbm_isca import Config, LBM, Spectral, VerticalGrid, asnumpy, gpu_available
from lbm_isca.basicstate import load_basic_state
from lbm_isca.forcing import SteadyForcing, MJOForcing
from lbm_isca.output import NetCDFWriter

DEVICE = "gpu" if gpu_available() else "cpu"

# write into the manuscript figures/ directory
MANU_FIGDIR = os.path.normpath(
    os.path.join(ISCA_ROOT, "..", "LBM_GPU_GMD", "figures"))
FIGDIR = MANU_FIGDIR if os.path.isdir(MANU_FIGDIR) else os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# inputs shared with the sibling port
DATA = os.path.normpath(os.path.join(ISCA_ROOT, "..", "LBM_python_GPU"))
BS_1979 = os.path.join(DATA, "bs.1979.bin")
FRC_PHASE1 = os.path.join(DATA, "frc.phase1.bin")

print(f"PyLBM-Isca figures  backend={DEVICE}  -> {FIGDIR}")


# ---------------------------------------------------------------------------
# idealised basic states / forcing (identical construction to the sibling)
# ---------------------------------------------------------------------------
def resting_state(sp, vg, t0=250.0):
    shape = (sp.nlat, sp.nlon, vg.kmax)
    return types.SimpleNamespace(
        u=np.zeros(shape), v=np.zeros(shape),
        t=np.full(shape, t0), q=np.zeros(shape),
        ps=np.full((sp.nlat, sp.nlon), 1000.0))


def solidbody_state(sp, vg, u0=25.0, t0=250.0):
    shape = (sp.nlat, sp.nlon, vg.kmax)
    coslat = np.sqrt(1.0 - sp.mu ** 2)
    u = np.broadcast_to(u0 * coslat[:, None, None], shape).copy()
    return types.SimpleNamespace(
        u=u, v=np.zeros(shape),
        t=np.full(shape, t0), q=np.zeros(shape),
        ps=np.full((sp.nlat, sp.nlon), 1000.0))


def gaussian_heating(sp, vg, lon0=180.0, lat0=0.0, dlon=20.0, dlat=10.0,
                     amp_k_per_day=1.0):
    frc = SteadyForcing.zeros(sp.nlat, sp.nlon, vg.kmax)
    dl = (sp.lon_deg[None, :] - lon0 + 180.0) % 360.0 - 180.0
    hxy = np.exp(-(dl / dlon) ** 2 - ((sp.lat_deg[:, None] - lat0) / dlat) ** 2)
    sig = asnumpy(vg.sig)
    vz = np.sin(np.pi * (1.0 - sig)) * (sig > 0.1)
    frc.t = hxy[:, :, None] * vz[None, None, :] * (amp_k_per_day / 86400.0)
    return frc


def kidx(vg, target):
    return int(np.argmin(np.abs(asnumpy(vg.sig) - target)))


def _save(fig, name):
    out = os.path.join(FIGDIR, name)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out[:-4] + ".png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"      wrote {out}")


# ---------------------------------------------------------------------------
# Figure 1 - Gill response
# ---------------------------------------------------------------------------
def fig_gill():
    print("\n[1/3] Gill response (equatorial heating, resting state)...")
    sp = Spectral(device=DEVICE)
    vg = VerticalGrid(device=DEVICE)
    frc = gaussian_heating(sp, vg, lon0=180.0, lat0=0.0, amp_k_per_day=1.0)
    m = LBM(resting_state(sp, vg), sp, vg, Config(ndays=6.0, device=DEVICE))
    m.run(steady_forcing=frc, verbose=False)

    klo = kidx(vg, 0.85)
    lon, lat = sp.lon_deg, sp.lat_deg
    u = asnumpy(m.ga.u[:, :, klo])
    v = asnumpy(m.ga.v[:, :, klo])
    heat = frc.t[:, :, kidx(vg, 0.5)] * 86400.0           # K/day

    fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    vmax = np.abs(u).max()
    lev = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(lon, lat, u, levels=lev, cmap="RdBu_r", extend="both")
    ax.contour(lon, lat, heat, levels=[0.1, 0.3, 0.5, 0.7],
               colors="green", linewidths=1.0)
    sk = 4
    ax.quiver(lon[::sk], lat[::sk * 2], u[::sk * 2, ::sk], v[::sk * 2, ::sk],
              scale=60, width=0.0018, color="0.25")
    ax.axhline(0, color="0.3", lw=0.6, ls=":")
    ax.set_ylim(-45, 45)
    ax.set_ylabel("latitude")
    ax.set_title(r"(a) Low-level ($\sigma\approx0.85$) zonal wind $u'$ "
                 r"[m s$^{-1}$]; green = heating", loc="left", fontsize=10)
    fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.025)

    ax = axes[1]
    eq = int(np.argmin(np.abs(lat)))
    ueq = u[eq]
    ax.plot(lon, ueq, "b-", lw=1.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(180.0, color="green", lw=1.0, ls="--")
    ax.fill_between(lon, 0, ueq, where=ueq > 0, color="tab:red", alpha=0.25)
    ax.fill_between(lon, 0, ueq, where=ueq < 0, color="tab:blue", alpha=0.25)
    ax.set_xlim(0, 360)
    ax.set_xlabel("longitude")
    ax.set_ylabel(r"$u'$ [m s$^{-1}$]")
    ax.set_title("(b) Equatorial profile", loc="left", fontsize=10)

    fig.tight_layout()
    _save(fig, "gill_response.pdf")


# ---------------------------------------------------------------------------
# Figure 2 - solid-body-rotation teleconnection
# ---------------------------------------------------------------------------
def fig_solidbody():
    print("\n[2/3] Rossby-wave train (idealised heating, solid-body state)...")
    sp = Spectral(device=DEVICE)
    vg = VerticalGrid(device=DEVICE)
    frc = gaussian_heating(sp, vg, lon0=180.0, lat0=15.0, dlon=15.0, dlat=8.0,
                           amp_k_per_day=2.0)
    m = LBM(solidbody_state(sp, vg, u0=25.0), sp, vg,
            Config(ndays=25.0, device=DEVICE))
    tmp = os.path.join(ISCA_ROOT, "out", "solidbody_fig.nc")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    m.run(steady_forcing=frc, writer=NetCDFWriter(tmp, sp, vg), verbose=False)

    import xarray as xr
    ds = xr.open_dataset(tmp)
    kup = kidx(vg, 0.2)
    sel = ds.sel(time=slice(12.0, 24.0)).mean("time")
    psi = sel.psi.isel(sigma=kup).values
    lon, lat = ds.lon.values, ds.lat.values
    heat = asnumpy(frc.t[:, :, kidx(vg, 0.5)]) * 86400.0

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = np.abs(psi).max()
    lev = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(lon, lat, psi, levels=lev, cmap="RdBu_r", extend="both")
    ax.contour(lon, lat, psi, levels=lev[::4], colors="k",
               linewidths=0.4, alpha=0.5)
    ax.contour(lon, lat, heat, levels=[0.2, 0.6, 1.0],
               colors="green", linewidths=1.2)
    ax.axhline(0, color="0.3", lw=0.6, ls=":")
    ax.set_ylim(-80, 80)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(r"(a) Upper-level ($\sigma\approx0.2$) streamfunction anomaly "
                 r"[m$^2$ s$^{-1}$], days 12-24", loc="left", fontsize=10)
    fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.03)
    fig.tight_layout()
    _save(fig, "solidbody_teleconnection.pdf")
    ds.close()


# ---------------------------------------------------------------------------
# Figure 3 - realistic MJO teleconnection (35-day run on the NCEP basic state)
# ---------------------------------------------------------------------------
def fig_teleconnection():
    print("\n[3/3] Realistic MJO phase-1 teleconnection (days 15-30)...")
    sp = Spectral(device=DEVICE)
    vg = VerticalGrid(device=DEVICE)
    bs = load_basic_state(BS_1979, sp, vg)
    m = LBM(bs, sp, vg, Config(ndays=35.0, device=DEVICE))
    nc = os.path.join(ISCA_ROOT, "out", "isca.1979.phase1.d35.nc")
    os.makedirs(os.path.dirname(nc), exist_ok=True)
    m.run(mjo_forcing=MJOForcing(FRC_PHASE1), writer=NetCDFWriter(nc, sp, vg),
          verbose=False)

    import xarray as xr
    ds = xr.open_dataset(nc)
    sel = ds.sel(time=slice(15.0, 30.0)).mean("time")
    k = int(np.argmin(np.abs(ds.sigma.values - 0.2)))
    sig = float(ds.sigma.values[k])
    psi = sel.psi.isel(sigma=k).values
    lon, lat = ds.lon.values, ds.lat.values

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = np.abs(psi).max()
    lev = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(lon, lat, psi, levels=lev, cmap="RdBu_r", extend="both")
    ax.contour(lon, lat, psi, levels=lev[::4], colors="k",
               linewidths=0.4, alpha=0.5)
    ax.axhline(0, color="0.3", lw=0.6, ls=":")
    ax.set_ylim(-80, 80)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(rf"(a) 200 hPa ($\sigma\approx{sig:.3f}$) streamfunction "
                 r"anomaly [m$^2$ s$^{-1}$]", loc="left", fontsize=10)
    fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.03)
    fig.tight_layout()
    _save(fig, "teleconnection_d15_30.pdf")
    ds.close()


if __name__ == "__main__":
    fig_gill()
    fig_solidbody()
    fig_teleconnection()
    print("\nAll PyLBM-Isca validation figures written to", FIGDIR)
