"""Semi-implicit leapfrog time integration, damping and vertical diffusion.

The hyperdiffusion/sponge, the per-wavenumber gravity-wave matrix and the
implicit vertical diffusion are numerical scaffolding, independent of which
dynamical core supplies the vertical operators; they follow the standard
semi-implicit spectral scheme (Bourke 1974; Hoskins & Simmons 1975).  Here they
consume the **Simmons-Burridge** operators (``whmtx = WW @ tmtx``, ``gcmtx``,
``gg``) that :mod:`pylbm_isca.vertical` builds, so the same per-wavenumber matrix
``M(n) = diag(fd*ft) - dt^2*lap*(whmtx + gcmtx*ft)`` becomes Isca's gravity-wave
operator (``implicit.F90:build_matrix``).  The Robert-Asselin time filter lives
in :mod:`pylbm_isca.state` / the time loop, not here.

The gravity-wave matrix depends only on the *total* wavenumber n, so there is
one KMAX x KMAX matrix per n (43 of them at T42), rebuilt whenever dt changes.

GPU note
--------
The coefficient vectors and the factorised operators live on the device.  The
43 small gravity-wave matrices (and the vertical-diffusion operators) are
assembled on the host -- a one-off cost that only recurs when ``dt`` changes --
then inverted in a single batched :func:`linalg.inv` on the GPU.  The two
contractions that run every step, :meth:`SemiImplicit.solve` and
:meth:`VerticalDiffusion.apply`, are batched ``einsum`` calls over all (m, n)
spectral coefficients, so the whole implicit solve stays on the device.
"""

import numpy as np

from .backend import asnumpy, to_device
from .constants import S2D


class Damping:
    """Hyperdiffusion, top sponge and Rayleigh/Newtonian damping.

    All of these are applied *implicitly*, as 1/(1 + 2*dt*K) factors inside
    the leapfrog update and inside the semi-implicit matrix -- never as an
    explicit tendency.  The resulting coefficients are (n, k) arrays in s^-1.
    """

    def __init__(self, sp, vgrid, order=4, tefold=2.0, tunit_hdif="HOUR",
                 ddragv=None, ddragd=None, ddragt=None, ddragq=None,
                 tunit_damp="DAY", ray0=30.0, zsclh=8.0e3, rzsclh=7.0e3,
                 sponge=True):
        self.sp, self.vg = sp, vgrid
        self.xp = sp.xp
        nmax, kmax = sp.nmax, vgrid.kmax
        n = np.arange(nmax + 1, dtype=np.float64)
        nn1 = n * (n + 1.0)
        p = order // 2

        tau = _to_seconds(tefold, tunit_hdif)

        # Hyperdiffusion coefficients: the /NMAX inside both diffusion scalings
        # cancels, leaving [n(n+1)/(N(N+1))]^p.
        difmax = (nmax * (nmax + 1.0) / nmax) ** p
        dift_n = (nn1 / nmax) ** p
        difv_n = dift_n - (2.0 / nmax) ** p

        kt = (1.0 / tau) * dift_n / difmax
        kv = (1.0 / tau) * difv_n / difmax
        kt[0] = 0.0
        kv[0] = 0.0
        kv[1] = max(kv[1], 0.0)          # exactly zero by construction

        self.kt_h = kt
        self.kv_h = kv

        # Top sponge.  Note RAY0 is interpreted in the
        # &nmhdif TUNIT, which the run script sets to HOUR.
        self.sponge_k = np.zeros(kmax)
        if sponge:
            ray0_s = _to_seconds(ray0, tunit_hdif)
            z = -zsclh * np.log(asnumpy(vgrid.sig))
            rzofs = -zsclh * np.log(float(asnumpy(vgrid.sig[-1])))
            fac = np.maximum(1.0 + np.tanh((z - rzofs) / rzsclh), 1e-20)
            self.sponge_k = fac / ray0_s

        # Rayleigh / Newtonian damping (&nmdamp)
        self.drag_v = _drag(ddragv, kmax, tunit_damp)
        self.drag_d = _drag(ddragd, kmax, tunit_damp)
        self.drag_t = _drag(ddragt, kmax, tunit_damp)
        self.drag_q = _drag(ddragq, kmax, tunit_damp)

        # Assemble the (n, k) coefficient arrays.  Order matters: the
        # hyperdiffusion is zero at n=0 (and, for vor/div, at n=1 as well --
        # that falls out automatically), but the sponge and the
        # &nmdamp drag are added at *every* n, n=0 included.
        kv2 = kv[:, None] + self.sponge_k[None, :] + self.drag_v[None, :]
        kd2 = kv[:, None] + self.sponge_k[None, :] + self.drag_d[None, :]
        ktt2 = kt[:, None] + self.drag_t[None, :]
        kq2 = kt[:, None] + self.drag_q[None, :]

        # Hot-loop coefficients live on the device.
        self.kv = to_device(kv2, self.xp)
        self.kd = to_device(kd2, self.xp)
        self.ktt = to_device(ktt2, self.xp)
        self.kq = to_device(kq2, self.xp)


def _drag(d, kmax, tunit):
    """Convert a per-level damping timescale list into s^-1 (<=0 disables)."""
    out = np.zeros(kmax)
    if d is None:
        return out
    d = np.asarray(d, dtype=np.float64)
    if d.size != kmax:
        raise ValueError(f"damping profile must have {kmax} entries")
    sec = np.array([_to_seconds(x, tunit) for x in d])
    good = sec > 0
    out[good] = 1.0 / sec[good]
    return out


def _to_seconds(value, unit):
    u = unit.strip().upper()[:3]
    return float(value) * {"SEC": 1.0, "MIN": 60.0, "HOU": 3600.0,
                           "DAY": 86400.0, "MON": 30 * 86400.0}[u]


class SemiImplicit:
    """The gravity-wave matrix and its factorisation, one per wavenumber n."""

    def __init__(self, sp, vgrid, damping):
        self.sp, self.vg, self.dmp = sp, vgrid, damping
        self.xp = sp.xp
        self._dt = None
        self.minv = None

    def prepare(self, dt):
        """Build and invert the matrices for this dt.

        The 43 (nmax+1) matrices are assembled once on the host -- cheap, and
        only when ``dt`` changes -- then inverted in one batched GPU call.
        """
        if self._dt == dt:
            return
        sp, vg, dmp = self.sp, self.vg, self.dmp
        nmax, kmax = sp.nmax, vg.kmax
        dt2 = 2.0 * dt

        # Host copies of the operators needed to assemble the matrices.
        laplace = asnumpy(sp.laplace)
        ktt = asnumpy(dmp.ktt)
        kd = asnumpy(dmp.kd)
        whmtx = asnumpy(vg.whmtx)
        gcmtx = asnumpy(vg.gcmtx)

        idx = np.arange(kmax)
        m = np.empty((nmax + 1, kmax, kmax))
        for n in range(nmax + 1):
            lap = laplace[n]
            ft = 1.0 + dt2 * ktt[n]                 # (kmax,)
            fd = 1.0 + dt2 * kd[n]
            core = whmtx + gcmtx * ft[:, None]
            mat = -dt * dt * lap * core
            mat[idx, idx] += fd * ft
            m[n] = mat

        self.m = to_device(m, self.xp)
        self.minv = self.xp.linalg.inv(self.m)
        self._dt = dt
        self.cond = np.array([np.linalg.cond(x) for x in m])

    def solve(self, rhs):
        """Apply M(n)^-1 to a spectral field of shape (m, n, k).

        The ellipsis batches any leading dimensions (e.g. an ensemble member
        axis, giving (E, m, n, k)); for a plain (m, n, k) field it is identical
        to ``einsum('nij,mnj->mni', ...)``.
        """
        return self.xp.einsum("nij,...nj->...ni", self.minv, rhs, optimize=True)


class VerticalDiffusion:
    """Implicit sigma-space vertical diffusion (DSETVDF / SOLVDF)."""

    def __init__(self, vgrid, vdifv=None, vdifd=None, vdift=None, vdifq=None):
        self.vg = vgrid
        self.xp = vgrid.xp
        kmax = vgrid.kmax
        dsig, dsigm = asnumpy(vgrid.dsig), asnumpy(vgrid.dsigm)

        # Tridiagonal second difference in sigma
        a = np.zeros(kmax)
        c = np.zeros(kmax)
        a[1:] = 1.0 / (dsig[1:] * dsigm[1:kmax])
        c[:-1] = 1.0 / (dsig[:-1] * dsigm[1:kmax])
        b = -(a + c)
        # zero-flux ends
        c[0] = (dsigm[0] + dsigm[1]) / (dsig[0] * dsigm[0] * dsigm[1])
        a[0] = 0.0
        b[0] = -c[0]
        a[-1] = (dsigm[kmax - 1] + dsigm[kmax]) / (
            dsig[-1] * dsigm[kmax - 1] * dsigm[kmax])
        c[-1] = 0.0
        b[-1] = -a[-1]

        lop = np.zeros((kmax, kmax))
        for k in range(kmax):
            lop[k, k] = b[k]
            if k > 0:
                lop[k, k - 1] = a[k]
            if k < kmax - 1:
                lop[k, k + 1] = c[k]
        self.lop = lop                       # host copy for assembly

        self.coef = {}
        for name, v in (("vor", vdifv), ("div", vdifd),
                        ("t", vdift), ("q", vdifq)):
            self.coef[name] = _vdif_coef(v, kmax)

        self._dt = None
        self.minv = {}

    def prepare(self, dt):
        if self._dt == dt:
            return
        kmax = self.vg.kmax
        eye = np.eye(kmax)
        self.minv = {}
        for name, nu in self.coef.items():
            if nu is None:
                self.minv[name] = None
            else:
                minv_h = np.linalg.inv(eye - dt * nu[:, None] * self.lop)
                self.minv[name] = to_device(minv_h, self.xp)
        self._dt = dt

    def apply(self, field, name, skip_low_n=False):
        """Solve (I - dt*nu*L) X_new = X for a spectral field (m, n, k)."""
        minv = self.minv.get(name)
        if minv is None:
            return field
        out = self.xp.einsum("ij,...j->...i", minv, field, optimize=True)
        if skip_low_n:
            # n is the second-to-last axis for both (m, n, k) and (E, m, n, k).
            out[..., 0, :] = field[..., 0, :]
            out[..., 1, :] = field[..., 1, :]
        return out


def _vdif_coef(v, kmax):
    """VDF(k) = 1/(86400*VDIF(k)); a non-positive entry disables the variable
    entirely."""
    if v is None:
        return None
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    if v.size == 1:
        v = np.full(kmax, v[0])
    if np.any(v <= 0):
        return None
    return 1.0 / (S2D * v)


class Integrator:
    """Semi-implicit leapfrog time integration."""

    def __init__(self, sp, vgrid, damping, semi, vdiff):
        self.sp, self.vg = sp, vgrid
        self.xp = sp.xp
        self.dmp, self.semi, self.vdiff = damping, semi, vdiff

    def step(self, dt, prev, tend):
        """Advance one leapfrog step.

        Parameters
        ----------
        dt : float
            Half the leapfrog interval, i.e. the time the model clock
            advances per step.
        prev : dict
            Spectral fields at ``t-dt``: keys vor, div, t, pi, q.
        tend : tuple
            Explicit tendencies (wt_vor, wt_div, wt_t, wt_pi, wt_q).
        """
        xp, vg, dmp = self.xp, self.vg, self.dmp
        self.semi.prepare(dt)
        self.vdiff.prepare(dt)
        dt2 = 2.0 * dt

        wt_vor, wt_div, wt_t, wt_pi, wt_q = tend
        ft = 1.0 + dt2 * dmp.ktt            # (n, k)
        fd = 1.0 + dt2 * dmp.kd

        # --- non-gravity-wave temperature estimate and its geopotential ---
        wdx = prev["t"] * (1.0 + dt * dmp.ktt) + wt_t * dt
        wdphix = vg.hydrostatic(wdx)

        wdpix = prev["pi"] + wt_pi * dt
        # WPHIS == 0 (no orography)
        wdphiy = wdphix + ft * (wdpix[..., None] * vg.gg)

        lap = self.sp.laplace[None, :, None]
        rhs = (prev["div"] * (1.0 + dt * dmp.kd) * ft
               + dt * (wt_div * ft - lap * wdphiy))
        dbar = self.semi.solve(rhs)

        # --- restore the implicit gravity-wave coupling -------------------
        wt_t = wt_t - xp.einsum("kl,...l->...k", vg.tmtx, dbar, optimize=True)
        wt_pi = wt_pi - (dbar * vg.dsig).sum(axis=-1)

        new = {}
        new["vor"] = (prev["vor"] + dt2 * wt_vor) / (1.0 + dt2 * dmp.kv)
        new["div"] = 2.0 * dbar - prev["div"]
        new["t"] = (prev["t"] + dt2 * wt_t) / ft
        new["q"] = (prev["q"] + dt2 * wt_q) / (1.0 + dt2 * dmp.kq)
        new["pi"] = prev["pi"] + dt2 * wt_pi

        # --- implicit vertical diffusion (applied after the leapfrog) -----
        new["vor"] = self.vdiff.apply(new["vor"], "vor", skip_low_n=True)
        new["div"] = self.vdiff.apply(new["div"], "div", skip_low_n=True)
        new["t"] = self.vdiff.apply(new["t"], "t")
        new["q"] = self.vdiff.apply(new["q"], "q")
        return new
