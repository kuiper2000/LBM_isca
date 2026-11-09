#!/usr/bin/env python3
"""Regression test for the sign of the vertical-advection operator.

A sign error in ``_vert_advect`` (feeding the Watanabe--Kimoto-convention
``sdot`` into the Isca advective-form stencil) left every isothermal / solid-body
test passing -- those states have ``d(basic)/d(sigma) = 0`` and ``sdot_bar = 0``,
so all vertical-advection terms vanish -- while collapsing the response on a
realistic baroclinic basic state by ~5x (the divergent circuulation by ~20x).

The clean invariant that pins the sign: advecting the coordinate field
``f = sigma`` must give the physical ``-sigma_dot`` (interpolated to full
levels), since ``d(sigma)/d(sigma) = 1``.  The magnitude is only approximate on
the stretched L20 grid, but the *sign* is exact -- which is exactly what the bug
got wrong.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from lbm_isca import VerticalGrid                          # noqa: E402
from lbm_isca.dynamics import _vert_advect                 # noqa: E402


def test_vert_advect_sign():
    vg = VerticalGrid(device="cpu")
    sig = np.asarray(vg.sig)
    dsig = np.asarray(vg.dsig)
    k = len(sig)

    # smooth interior sigma-dot, zero at both boundaries
    sdot = np.zeros(k + 1)
    sdot[1:-1] = 1e-3 * np.sin(np.pi * np.arange(1, k) / k)

    adv = _vert_advect(sig, sdot, dsig)        # advection of the coordinate
    ref = -0.5 * (sdot[:-1] + sdot[1:])        # -sigma_dot at full levels

    # sign must match everywhere in the interior (this is what the bug flipped)
    interior = slice(1, -1)
    assert np.all(np.sign(adv[interior]) == np.sign(ref[interior])), (
        "vertical advection has the wrong sign -- advecting f=sigma must give "
        "-sigma_dot, not +sigma_dot (the _sigma_dot/_vert_advect convention bug)")

    # and be strongly positively correlated with the physical reference
    corr = np.corrcoef(adv[interior], ref[interior])[0, 1]
    assert corr > 0.95, f"advection operator poorly correlated with -sigma_dot: {corr}"
    print(f"OK: vert-advection sign correct (corr with -sigma_dot = {corr:.4f})")


if __name__ == "__main__":
    test_vert_advect_sign()
