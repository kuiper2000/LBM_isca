#!/usr/bin/env python3
# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""Tangent-linear verification of the PyLBM-Isca operator (GMD revision item 3).

We test that the implemented linear operator L(xbar) is the exact tangent-linear
of the model's own tendency F about a nontrivial (realistic) basic state, and
that the single retained nonlinearity is the documented kinetic-energy term.

Method (gradient / Taylor test).  For a fixed perturbation direction dx in the
prognostic variables (u', v', T', q', pi'), with pi' set exactly through
``State.set_from_lnps`` so the log-pressure perturbation scales linearly, we form

    F_a  = F(a*dx)             nonlinear tendency, KE term retained
    L_a  = L(a*dx)             strictly-linear tendency (should equal a*L_dir)
    R(a) = || F_a - F(0) - a*L_dir ||          (Taylor remainder)

for a geometric sweep of amplitudes a and for each prognostic variable.
Because the only nonlinearity is 0.5*(u'^2+v'^2) in the divergence equation:

  * the divergence remainder R_D(a) decreases as O(a^2) until round-off
    (cancellation) dominates -- the classic tangent-linear signature;
  * the vorticity, temperature, humidity and log-pressure remainders sit at the
    round-off floor for *all* a, i.e. those tendencies are linear to machine
    precision (no unintended nonlinearity).

Isca's full nonlinear core is not run here; this verifies the linearization of
the *implemented* tendency (the derivation's correspondence to Isca is
established by the equation-to-code audit, Appendix A / docs/equation_code_map.md).

Usage:  python analysis/tl_verification.py [--basic-state data/bs.1979.bin]
Outputs: analysis/tl_verification.{pdf,png,json} and a printed table.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pylbm_isca import Spectral, VerticalGrid               # noqa: E402
from pylbm_isca.backend import asnumpy                      # noqa: E402
from pylbm_isca.dynamics import BasicStateTerms, Dynamics   # noqa: E402
from pylbm_isca.state import State                          # noqa: E402

VARS = ["vor", "div", "T", "q", "pi"]


class _BasicState:
    """Minimal duck-typed basic state (fields on the model grid)."""
    def __init__(self, u, v, t, q, ps):
        self.u, self.v, self.t, self.q, self.ps = u, v, t, q, ps


def load_basic(sp, vg, path):
    """Realistic DJF basic state if the file exists, else a synthetic jet."""
    if path and os.path.exists(path):
        from pylbm_isca.basicstate import load_basic_state
        bs = load_basic_state(path, sp, vg, kmo=1, navg=3)
        return bs, f"realistic DJF 1979 ({os.path.basename(path)})"

    # Synthetic baroclinic fallback: a mid-latitude zonal jet with vertical
    # shear, so the vertical-advection and baroclinic terms are all exercised.
    nlat, nlon, k = sp.nlat, sp.nlon, vg.kmax
    lat = np.arcsin(sp.mu)[:, None, None]
    sig = asnumpy(vg.sig)[None, None, :]
    jet = 30.0 * np.exp(-((lat - np.deg2rad(40)) / np.deg2rad(15)) ** 2)
    u = (jet * (sig ** 0.5)) * np.ones((nlat, nlon, k))
    v = np.zeros((nlat, nlon, k))
    t = 250.0 - 40.0 * np.sin(lat) ** 2 + 5.0 * sig        # crude baroclinicity
    t = t * np.ones((nlat, nlon, k))
    q = 1e-3 * np.exp(-((lat) / np.deg2rad(30)) ** 2) * np.ones((nlat, nlon, k))
    ps = 1.0e5 * np.ones((nlat, nlon))
    return _BasicState(u, v, t, q, ps), "synthetic baroclinic jet"


def make_direction(sp, vg, seed=0):
    """A fixed, band-limited unit-ish perturbation direction (grid space)."""
    rng = np.random.default_rng(seed)
    nlat, nlon, k = sp.nlat, sp.nlon, vg.kmax

    def smooth(scale, shape):
        g = rng.standard_normal(shape)
        # band-limit by a spectral round trip
        g = asnumpy(sp.wave_to_grid(sp.grid_to_wave(g)))
        g /= np.max(np.abs(g)) or 1.0
        return scale * g

    # physical O(1) magnitudes for the direction; amplitude a scales all of them
    u0 = smooth(1.0, (nlat, nlon, k))       # ~1 m/s
    v0 = smooth(1.0, (nlat, nlon, k))
    t0 = smooth(1.0, (nlat, nlon, k))       # ~1 K
    q0 = smooth(1.0e-3, (nlat, nlon, k))    # ~1e-3 kg/kg
    pi0 = smooth(1.0e-3, (nlat, nlon))      # ~1e-3 in log-pressure (2-D)
    return u0, v0, t0, q0, pi0


def state_at(sp, vg, bs, direction, a):
    u0, v0, t0, q0, pi0 = direction
    st = State.zeros(np.asarray(bs.ps), vg.kmax, xp=sp.xp)
    st.u = sp.xp.asarray(a * u0)
    st.v = sp.xp.asarray(a * v0)
    st.t = sp.xp.asarray(a * t0)
    st.q = sp.xp.asarray(a * q0)
    st.set_from_lnps(sp.xp.asarray(a * pi0))     # pi' = a*pi0 exactly
    return st


def tend_vec(dyn, st):
    """Return the 5 spectral tendencies as flat numpy complex arrays."""
    out = dyn.tendencies(st)
    return [np.asarray(asnumpy(x)).ravel() for x in out]


def norm(x):
    return float(np.sqrt(np.sum(np.abs(x) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic-state",
                    default=os.path.join(ROOT, "data", "bs.1979.bin"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    sp = Spectral(device=args.device)
    vg = VerticalGrid(device=args.device)
    bs, label = load_basic(sp, vg, args.basic_state)
    bst = BasicStateTerms(sp, vg, bs)
    dyn_lin = Dynamics(sp, vg, bst, strictly_linear=True)
    dyn_full = Dynamics(sp, vg, bst, strictly_linear=False)

    direction = make_direction(sp, vg)

    # F(0) and the linear direction L_dir = L(base)/1 (base amplitude a0).
    f0 = tend_vec(dyn_full, state_at(sp, vg, bs, direction, 0.0))
    a0 = 1.0
    l_dir = [x / a0 for x in tend_vec(dyn_lin, state_at(sp, vg, bs, direction, a0))]

    amps = np.logspace(-1, -8, 15)
    remainder = {v: [] for v in VARS}      # ABS  ||F_a - F0 - a*L_dir||
    rel_rem = {v: [] for v in VARS}        # REL  ||F_a - F0 - a*L_dir|| / ||a*L_dir||
    lin_check = {v: [] for v in VARS}      # REL  ||L_a - a*L_dir|| / ||a*L_dir||
    for a in amps:
        st = state_at(sp, vg, bs, direction, a)
        fa = tend_vec(dyn_full, st)
        la = tend_vec(dyn_lin, st)
        for i, v in enumerate(VARS):
            scale = norm(a * l_dir[i]) or 1.0
            rem = norm(fa[i] - f0[i] - a * l_dir[i])
            remainder[v].append(rem)
            rel_rem[v].append(rem / scale)
            lin_check[v].append(norm(la[i] - a * l_dir[i]) / scale)

    # ---- report ----------------------------------------------------------
    print(f"Tangent-linear verification on the {label} basic state")
    print(f"backend={sp.xp.__name__}; direction seed=0; amplitudes {amps[0]:.0e}..{amps[-1]:.0e}")
    print("\nAbsolute Taylor remainder  ||F(a dx) - F(0) - a L dx||")
    print("   a      " + "".join(f"{v:>12}" for v in VARS))
    for j, a in enumerate(amps):
        print(f"{a:8.1e}  " + "".join(f"{remainder[v][j]:12.2e}" for v in VARS))

    # Convergence slope of the divergence remainder in its O(a^2) regime
    # (before round-off cancellation, a > 1e-4).
    la_ = np.log(amps)
    reg = amps > 1e-4
    slope = {v: float(np.polyfit(la_[reg], np.log(np.array(remainder[v])[reg]), 1)[0])
             for v in VARS}
    # linearity of L, read at a moderate amplitude (free of 1/a round-off blow-up)
    jmod = int(np.argmin(np.abs(amps - 1e-2)))
    print("\nDivergence remainder slope (a>1e-4):  "
          f"{slope['div']:.2f}   (expect ~2 -> exact tangent-linear of the KE term)")
    print("Remainder slope of vor/T/q/pi:        "
          + ", ".join(f"{v}={slope[v]:.2f}" for v in ("vor", "T", "q", "pi"))
          + "   (flat ~0 -> linear to round-off)")
    print(f"Linearity residual of L at a=1e-2:    "
          f"{max(lin_check[v][jmod] for v in VARS):.2e}   (expect round-off)")

    out_json = os.path.join(HERE, "tl_verification.json")
    with open(out_json, "w") as fh:
        json.dump({"label": label, "amps": amps.tolist(),
                   "remainder_abs": remainder, "remainder_rel": rel_rem,
                   "lin_check": lin_check, "slope": slope}, fh, indent=2)
    print(f"\nwrote {out_json}")

    # ---- figure ----------------------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                       # pragma: no cover
            print(f"(skipping plot: {e})")
            return 0
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        markers = {"vor": "o", "div": "s", "T": "^", "q": "v", "pi": "D"}
        labels = {"vor": r"$\zeta'$", "div": r"$D'$", "T": r"$T'$",
                  "q": r"$q'$", "pi": r"$\pi'$"}
        for v in VARS:
            ax.loglog(amps, remainder[v], markers[v] + "-", ms=4, label=labels[v])
        # reference O(a^2) slope anchored on the divergence curve
        aref = amps[amps > 1e-4]
        rref = np.array(remainder["div"])[amps > 1e-4]
        c = rref[0] / aref[0] ** 2
        ax.loglog(aref, c * aref ** 2, "k--", lw=1, label=r"$\mathcal{O}(a^2)$")
        ax.set_xlabel("perturbation amplitude $a$")
        ax.set_ylabel(r"Taylor remainder $\|F(a\,\delta x)-F(0)-a\,\mathbf{L}\,\delta x\|$")
        ax.set_title("Tangent-linear verification (PyLBM-Isca)")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(HERE, f"tl_verification.{ext}"), dpi=150)
        print(f"wrote {os.path.join(HERE, 'tl_verification.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
