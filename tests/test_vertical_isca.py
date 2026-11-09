# Copyright (C) 2026  LBM_isca authors.  GPLv3 (or later).
"""Checks on the Simmons-Burridge vertical operators (lbm_isca.vertical).

These verify the ps-independent geometry (alpha, log-pressure weights, the
pressure-gradient weight) and the reference-state matrices against independent
hand computations, so a sign or index error shows up here rather than silently
downstream.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbm_isca.constants import KAPPA, RAIR, SIGM            # noqa: E402
from lbm_isca.vertical import VerticalGrid, compute_tbar    # noqa: E402
from lbm_isca.backend import asnumpy                        # noqa: E402


def _vg():
    return VerticalGrid(device="cpu")


def test_dsig_sums_to_one():
    vg = _vg()
    assert abs(float(asnumpy(vg.dsig).sum()) - 1.0) < 1e-12


def test_alpha_is_positive_and_top_is_one():
    vg = _vg()
    alpha = asnumpy(vg.alpha)
    assert np.all(alpha > 0.0), alpha
    # massless top layer: alpha == 1 (GFDL ln_top_level_factor = -1)
    assert abs(alpha[-1] - 1.0) < 1e-12


def test_full_level_between_half_levels():
    vg = _vg()
    sig = asnumpy(vg.sig)
    sm, sp = SIGM[:-1], SIGM[1:]
    # full level strictly inside its layer: sig_top < sig_full < sig_bot
    assert np.all(sig < sm + 1e-12)
    assert np.all(sig[:-1] > sp[:-1] - 1e-12)   # interior layers


def test_alpha_matches_definition():
    vg = _vg()
    sm, sp = SIGM[:-1], SIGM[1:]
    expect = np.ones_like(sm)
    interior = sp > 0
    expect[interior] = 1.0 - sp[interior] * (
        np.log(sm[interior]) - np.log(sp[interior])) / (sm[interior] - sp[interior])
    assert np.abs(asnumpy(vg.alpha) - expect).max() < 1e-12


def test_hydrostatic_matches_recurrence():
    """phi = Tv @ WW.T must equal the layer-by-layer upward integral."""
    vg = _vg()
    rng = np.random.default_rng(0)
    tv = 200.0 + 50.0 * rng.standard_normal(vg.kmax)
    phi_mat = asnumpy(vg.hydrostatic(tv))

    alpha = asnumpy(vg.alpha)
    dlog3 = asnumpy(vg.dlog3)
    # upward half-level integral, phi_half[0] = surface = 0
    phi_ref = np.zeros(vg.kmax)
    phi_half = 0.0
    for k in range(vg.kmax):                      # k = 0 bottom .. top
        phi_ref[k] = phi_half + RAIR * alpha[k] * tv[k]
        phi_half = phi_half + RAIR * dlog3[k] * tv[k]
    assert np.abs(phi_mat - phi_ref).max() < 1e-9


def test_pgf_weight_finite_and_reasonable():
    vg = _vg()
    cpg = asnumpy(vg.cpg)
    assert np.all(np.isfinite(cpg))
    assert np.all(cpg > 0.0)
    # dimensionless weight, O(1) per layer
    assert cpg.max() < 5.0, cpg


def test_reference_matrices_shape_and_finiteness():
    vg = _vg()
    tbar = 250.0 * np.ones(vg.kmax)
    vg.set_reference_temperature(tbar)
    for name in ("gg", "tmtx", "whmtx", "gcmtx"):
        a = asnumpy(getattr(vg, name))
        assert np.all(np.isfinite(a)), name
    assert asnumpy(vg.tmtx).shape == (vg.kmax, vg.kmax)
    # tmtx is upper-triangular (energy conversion depends on same+above levels)
    tm = asnumpy(vg.tmtx)
    assert np.abs(np.tril(tm, -1)).max() == 0.0


def test_tmtx_energy_conversion_sign():
    """tmtx is stored with the integrator sign convention (positive diagonal
    +kappa*Tbar*alpha); the physical energy conversion is -tmtx @ div, applied
    implicitly."""
    vg = _vg()
    tbar = 250.0 * np.ones(vg.kmax)
    vg.set_reference_temperature(tbar)
    tm = asnumpy(vg.tmtx)
    alpha = asnumpy(vg.alpha)
    diag = np.diag(tm)
    assert np.all(diag > 0.0)
    assert np.allclose(diag, KAPPA * tbar * alpha, rtol=1e-12)
    # whmtx = WW @ tmtx must be positive-definite-ish (stable gravity waves)
    assert np.linalg.eigvals(asnumpy(vg.whmtx)).real.min() > -1.0


def test_compute_tbar_is_plain_mean():
    gft = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    tb = asnumpy(compute_tbar(gft))
    assert np.allclose(tb, gft.mean(axis=(0, 1)))


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
