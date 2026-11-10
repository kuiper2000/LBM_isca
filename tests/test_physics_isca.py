# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""Physical validation of the linearised Isca-core dynamics.

The key tests are (1) a resting isothermal basic state with no perturbation and
no forcing must give *exactly* zero tendencies -- i.e. the linearisation has no
spurious source terms -- and (2) the Gill response: equatorial heating must give
a Kelvin wave east of the heating and a Rossby gyre pair to the west.  Getting
the Gill response right exercises the transforms, the Simmons-Burridge vertical
operators, the semi-implicit solve and the damping together.
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylbm_isca import Config, LBM, Spectral, VerticalGrid      # noqa: E402
from pylbm_isca.forcing import SteadyForcing                    # noqa: E402

_CACHE = {}


def resting_state(sp, vg, t0=250.0):
    shape = (sp.nlat, sp.nlon, vg.kmax)
    return types.SimpleNamespace(
        u=np.zeros(shape), v=np.zeros(shape),
        t=np.full(shape, t0), q=np.zeros(shape),
        ps=np.full((sp.nlat, sp.nlon), 1000.0))


def equatorial_heating(sp, vg, lon0=120.0, amp_k_per_day=1.0):
    frc = SteadyForcing.zeros(sp.nlat, sp.nlon, vg.kmax)
    dl = (sp.lon_deg[None, :] - lon0 + 180.0) % 360.0 - 180.0
    hxy = np.exp(-(dl / 20.0) ** 2 - (sp.lat_deg[:, None] / 10.0) ** 2)
    vz = np.sin(np.pi * (1.0 - vg.sig)) * (vg.sig > 0.1)
    frc.t = hxy[:, :, None] * vz[None, None, :] * (amp_k_per_day / 86400.0)
    return frc


def gill_run(days=6.0, scale=1.0, strictly_linear=False):
    key = (days, scale, strictly_linear)
    if key not in _CACHE:
        sp, vg = Spectral(device="cpu"), VerticalGrid(device="cpu")
        m = LBM(resting_state(sp, vg), sp, vg,
                Config(ndays=days, strictly_linear=strictly_linear))
        m.run(steady_forcing=equatorial_heating(sp, vg, amp_k_per_day=scale),
              verbose=False)
        _CACHE[key] = m
    return _CACHE[key]


# ---------------------------------------------------------------------------
def test_zero_state_zero_tendency():
    sp, vg = Spectral(device="cpu"), VerticalGrid(device="cpu")
    m = LBM(resting_state(sp, vg), sp, vg, Config(ndays=0.1))
    tend = m.dyn.tendencies(m.ga)
    for name, x in zip(("vor", "div", "t", "pi", "q"), tend):
        assert np.abs(x).max() == 0.0, f"{name} tendency is not zero"


def test_semi_implicit_matrix_well_conditioned():
    sp, vg = Spectral(device="cpu"), VerticalGrid(device="cpu")
    m = LBM(resting_state(sp, vg), sp, vg, Config())
    m.semi.prepare(m.cfg.delt)
    assert m.semi.cond.max() < 100.0, f"cond = {m.semi.cond.max():.3g}"
    resid = np.einsum("nij,njk->nik", m.semi.m, m.semi.minv)
    err = np.abs(resid - np.eye(vg.kmax)).max()
    assert err < 1e-10, f"inverse residual {err:.2e}"


def test_gill_response_structure():
    m = gill_run()
    sp, vg = m.sp, m.vg
    klo = int(np.argmin(np.abs(vg.sig - 0.85)))
    eq = int(np.argmin(np.abs(sp.lat_deg)))
    u = m.ga.u[eq, :, klo]
    ilon = lambda d: int(round(d / (360.0 / sp.nlon)))        # noqa: E731
    # West of the heating (Rossby): low-level westerlies feeding the heat
    assert u[ilon(100)] > 0 and u[ilon(80)] > 0
    # East of the heating (Kelvin): low-level easterlies
    assert u[ilon(160)] < 0 and u[ilon(200)] < 0
    # Kelvin signal reaches much further east than the Rossby does west
    assert abs(u[ilon(280)]) > abs(u[ilon(40)])


def test_gill_rossby_gyres_are_antisymmetric():
    m = gill_run()
    sp, vg = m.sp, m.vg
    klo = int(np.argmin(np.abs(vg.sig - 0.85)))
    vor = sp.wave_to_grid(sp.uv_to_vordiv(m.ga.u, m.ga.v)[0])[:, :, klo]
    iw = int(round(90.0 / (360.0 / sp.nlon)))
    jn = int(np.argmin(np.abs(sp.lat_deg - 12.0)))
    js = int(np.argmin(np.abs(sp.lat_deg + 12.0)))
    north, south = vor[jn, iw], vor[js, iw]
    assert north > 0 > south, "expected a cyclonic pair straddling the equator"
    assert abs(north + south) < 1e-3 * abs(north), "pair is not antisymmetric"


def test_response_stays_bounded():
    m = gill_run()
    assert m.ga.isfinite()
    mx = m.ga.max_abs()
    assert mx["u"] < 20.0 and mx["t"] < 20.0, mx


def test_free_decay():
    sp, vg = Spectral(device="cpu"), VerticalGrid(device="cpu")
    m = LBM(resting_state(sp, vg), sp, vg, Config(ndays=4.0))
    t0 = 0.5 * np.exp(-(sp.lat_deg[:, None] / 20.0) ** 2) * np.cos(
        3.0 * np.radians(sp.lon_deg)[None, :])
    m.set_initial_perturbation(t=np.repeat(t0[:, :, None], vg.kmax, axis=2))
    start = np.abs(m.ga.t).max()
    m.run(verbose=False)
    assert m.ga.isfinite()
    assert np.abs(m.ga.t).max() < start, "perturbation grew without forcing"


def test_mass_is_not_spuriously_created():
    m = gill_run()
    gw = m.sp.gw[:, None]
    mean_ps = (m.ga.ps * gw).sum() / (gw.sum() * m.sp.nlon)
    assert abs(mean_ps) < 1e-3, f"global mean ps' = {mean_ps:.3e} hPa"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL  {name}: {exc}")
    print("all good" if not fails else f"{fails} failure(s)")
    sys.exit(1 if fails else 0)
