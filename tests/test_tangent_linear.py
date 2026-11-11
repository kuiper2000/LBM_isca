#!/usr/bin/env python3
# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""Tangent-linear check (GMD revision item 3), self-contained for CI.

Asserts that the strictly-linear operator L is the exact tangent-linear of the
model's own (KE-retained) tendency F about a nontrivial basic state:

* the divergence Taylor remainder decreases as O(a^2) (the retained
  kinetic-energy term is exactly linearized), and
* the vorticity / temperature / humidity / log-pressure remainders stay at the
  round-off floor for all amplitudes (those tendencies are linear to machine
  precision).

Runs on a synthetic baroclinic jet so it needs no external data; the full
figure on the realistic DJF 1979 state is produced by
``analysis/tl_verification.py``.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

from pylbm_isca import Spectral, VerticalGrid               # noqa: E402
from pylbm_isca.dynamics import BasicStateTerms, Dynamics   # noqa: E402
import tl_verification as tl                                 # noqa: E402


def test_tangent_linear():
    sp = Spectral(device="cpu")
    vg = VerticalGrid(device="cpu")
    bs, label = tl.load_basic(sp, vg, path=None)       # synthetic jet
    bst = BasicStateTerms(sp, vg, bs)
    dyn_lin = Dynamics(sp, vg, bst, strictly_linear=True)
    dyn_full = Dynamics(sp, vg, bst, strictly_linear=False)
    direction = tl.make_direction(sp, vg)

    f0 = tl.tend_vec(dyn_full, tl.state_at(sp, vg, bs, direction, 0.0))
    l_dir = tl.tend_vec(dyn_lin, tl.state_at(sp, vg, bs, direction, 1.0))

    amps = np.logspace(-1, -4, 7)             # O(a^2) regime, before round-off
    rem = {v: [] for v in tl.VARS}            # absolute remainder
    rel = {v: [] for v in tl.VARS}            # remainder / ||a*L_dir||
    for a in amps:
        fa = tl.tend_vec(dyn_full, tl.state_at(sp, vg, bs, direction, a))
        for i, v in enumerate(tl.VARS):
            r = tl.norm(fa[i] - f0[i] - a * l_dir[i])
            rem[v].append(r)
            rel[v].append(r / (tl.norm(a * l_dir[i]) or 1.0))

    la = np.log(amps)
    slope = {v: float(np.polyfit(la, np.log(np.array(rem[v]) + 1e-300), 1)[0])
             for v in tl.VARS}

    # Divergence carries the quadratic KE term -> exact O(a^2) convergence.
    assert 1.8 <= slope["div"] <= 2.2, f"div remainder slope {slope['div']:.2f} != 2"

    # Every other prognostic tendency is linear to (near) round-off: its
    # remainder relative to ||a*L_dx|| stays negligible at every amplitude.
    for v in ("vor", "T", "q", "pi"):
        assert max(rel[v]) < 1e-6, f"{v} relative remainder too large ({max(rel[v]):.1e})"

    print(f"OK: tangent-linear verified on {label}; div slope={slope['div']:.2f}; "
          "linear-var relative remainder < "
          f"{max(max(rel[v]) for v in ('vor', 'T', 'q', 'pi')):.1e}")


if __name__ == "__main__":
    test_tangent_linear()
