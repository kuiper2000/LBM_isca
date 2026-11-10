"""Validation of the spherical-harmonic transforms.

These are the tests that matter most: if the transforms or the metric factors
are wrong, every tendency in the model is wrong in a way that is very hard to
diagnose later.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylbm_isca.spectral import Spectral  # noqa: E402


def _sp():
    return Spectral(device="cpu")


def test_gaussian_weights():
    sp = _sp()
    assert abs(sp.gw.sum() - 1.0) < 1e-13
    # north first
    assert sp.lat_deg[0] > 87.0 and sp.lat_deg[-1] < -87.0


def test_legendre_orthonormality():
    """sum_j gw_j P_n^m P_n'^m = delta_{nn'} for every m."""
    sp = _sp()
    worst = 0.0
    for m in [0, 1, 5, 20, 42]:
        p = sp.pnm[m, m:]                      # (n, j)
        gram = (p * sp.gw) @ p.T
        err = np.abs(gram - np.eye(gram.shape[0])).max()
        worst = max(worst, err)
    assert worst < 1e-10, f"orthonormality error {worst:.2e}"


def test_roundtrip_bandlimited():
    """A field built from harmonics must survive grid->wave->grid exactly."""
    sp = _sp()
    rng = np.random.default_rng(0)
    xi = np.zeros((sp.nmax + 1, sp.nmax + 1), dtype=complex)
    xi[sp.mask] = rng.normal(size=sp.mask.sum()) + 1j * rng.normal(size=sp.mask.sum())
    xi[0] = xi[0].real                        # m=0 must be real
    g = sp.wave_to_grid(xi)
    xi2 = sp.grid_to_wave(g)
    err = np.abs(xi2 - xi)[sp.mask].max() / np.abs(xi)[sp.mask].max()
    assert err < 1e-11, f"round-trip error {err:.2e}"


def test_laplacian_eigenvalue():
    """grad^2 Y_n^m = -n(n+1)/a^2 Y_n^m, verified in grid space."""
    sp = _sp()
    for (m, n) in [(0, 1), (1, 1), (3, 7), (10, 20)]:
        xi = np.zeros((sp.nmax + 1, sp.nmax + 1), dtype=complex)
        xi[m, n] = 1.0 + 0.5j
        g = sp.wave_to_grid(xi)
        lap = sp.wave_to_grid(sp.laplacian(xi))
        expect = -n * (n + 1) / sp.a ** 2 * g
        err = np.abs(lap - expect).max() / np.abs(expect).max()
        assert err < 1e-10, f"(m,n)=({m},{n}) laplacian error {err:.2e}"


def test_gradient_against_analytic():
    """Gradient of f = sin(lat) (= mu) has d/dy = cos(lat)/a, d/dx = 0."""
    sp = _sp()
    f = np.repeat(sp.mu[:, None], sp.nlon, axis=1)
    xi = sp.grid_to_wave(f)
    dfdx, dfdy = sp.hgrad(xi)
    assert np.abs(dfdx).max() < 1e-16
    expect = sp.coslat[:, None] / sp.a
    err = np.abs(dfdy - expect).max() / expect.max()
    assert err < 1e-10, f"dy error {err:.2e}"


def test_gradient_zonal():
    """f = cos(lam)*cos(lat): check both derivative components analytically."""
    sp = _sp()
    lam = sp.lon[None, :]
    cl = sp.coslat[:, None]
    f = np.cos(lam) * cl
    dfdx, dfdy = sp.hgrad(sp.grid_to_wave(f))
    # df/dx = 1/(a cos) * d/dlam = -sin(lam)/a
    ex = -np.sin(lam) / sp.a * np.ones_like(cl)
    # df/dy = 1/a d/dphi (cos(lam) cos(phi)) = -cos(lam) sin(phi)/a
    ey = -np.cos(lam) * sp.mu[:, None] / sp.a
    assert np.abs(dfdx - ex).max() / np.abs(ex).max() < 1e-10
    assert np.abs(dfdy - ey).max() / np.abs(ey).max() < 1e-10


def test_vordiv_roundtrip():
    """u,v -> vor,div -> u,v must be the identity on band-limited winds."""
    sp = _sp()
    rng = np.random.default_rng(1)
    xi = np.zeros((sp.nmax + 1, sp.nmax + 1), dtype=complex)
    sel = sp.mask.copy()
    sel[:, 0] = False                        # n=0 has no rotational/divergent part
    xi[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
    xi[0] = xi[0].real
    psi = xi * 1e6
    chi = xi.conj() * 5e5
    chi[0] = chi[0].real
    u, v = sp.psichi_to_uv(psi, chi)
    vor, div = sp.uv_to_vordiv(u, v)
    psi2 = sp.inverse_laplacian(vor)
    chi2 = sp.inverse_laplacian(div)
    e1 = np.abs(psi2 - psi)[sel].max() / np.abs(psi)[sel].max()
    e2 = np.abs(chi2 - chi)[sel].max() / np.abs(chi)[sel].max()
    assert e1 < 1e-10 and e2 < 1e-10, f"psi err {e1:.2e}, chi err {e2:.2e}"


def test_solid_body_rotation():
    """u = U0 cos(lat), v = 0 is pure vorticity with zeta = 2 U0 sin(lat)/a."""
    sp = _sp()
    u0 = 20.0
    u = u0 * sp.coslat[:, None] * np.ones((1, sp.nlon))
    v = np.zeros_like(u)
    vor, div = sp.uv_to_vordiv(u, v)
    zg = sp.wave_to_grid(vor)
    dg = sp.wave_to_grid(div)
    expect = 2.0 * u0 * sp.mu[:, None] / sp.a
    assert np.abs(dg).max() < 1e-18, "solid rotation must be non-divergent"
    err = np.abs(zg - expect).max() / np.abs(expect).max()
    assert err < 1e-10, f"solid-body vorticity error {err:.2e}"


def test_divergence_of_gradient_is_laplacian():
    """div(grad f) == laplacian(f), a strong check on the metric factors."""
    sp = _sp()
    rng = np.random.default_rng(2)
    xi = np.zeros((sp.nmax + 1, sp.nmax + 1), dtype=complex)
    sel = sp.mask.copy()
    sel[:, 41:] = False                      # keep away from the truncation edge
    xi[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
    xi[0] = xi[0].real
    gx, gy = sp.hgrad(xi)
    divg = -sp.neg_divergence(gx, gy)
    lap = sp.laplacian(xi)
    err = np.abs(divg - lap)[sel].max() / np.abs(lap)[sel].max()
    assert err < 1e-9, f"div(grad) vs laplacian error {err:.2e}"


def test_multilevel_consistency():
    """The 3-D path must agree with the 2-D path level by level."""
    sp = _sp()
    rng = np.random.default_rng(3)
    f = rng.normal(size=(sp.nlat, sp.nlon, 4))
    xi3 = sp.grid_to_wave(f)
    for k in range(4):
        xi2 = sp.grid_to_wave(f[:, :, k])
        assert np.abs(xi3[:, :, k] - xi2).max() < 1e-12


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
