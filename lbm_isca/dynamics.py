# Copyright (C) 2026  LBM_isca authors
#
# This file is part of LBM_isca and is distributed under the GNU General Public
# License v3 (or later).  It linearises the Isca / GFDL ``atmos_spectral``
# primitive equations (github.com/ExeClim/Isca, GPLv3) about a basic state.
"""Linearised Isca/GFDL primitive-equation tendencies (sigma coordinate).

A term-by-term linearisation of the moist, dry-temperature Isca dynamical core:

  spectral_dynamics.F90  tendency assembly (:856-934) and ``four_in_one``
                         (:1041-1115)
  press_and_geopot.F90   ``compute_geopotential``
  vert_advection.F90     ``vert_advection_3d`` (SECOND_CENTERED, ADVECTIVE_FORM)

Everything is a (basic x perturbation) product, with the single documented
exception of the kinetic-energy term ``0.5*(u'^2 + v'^2)``, which Isca retains
exactly; set ``strictly_linear=True`` to drop it.

Physics choices reproduced from Isca (confirmed against the source):

* ``four_in_one`` is called with **virtual** temperature, so both the
  pressure-gradient term ``-R*Tv*cpg*grad(ln ps)`` and the adiabatic term
  ``-kappa*Tv*x5`` use ``Tv = T(1 + 0.608 q)`` (spectral_dynamics.F90:860-867).
* ``compute_geopotential`` virtualises internally, so the geopotential uses Tv
  as well.
* Horizontal and vertical advection of *temperature* use the **dry** T
  (spectral_dynamics.F90:890 passes ``tg``), as does tracer advection for q.

Semi-implicit split (see :mod:`lbm_isca.vertical`): the reference
energy-conversion from divergence is carried as ``+ div @ tmtx.T`` and the
reference pressure-gradient / continuity coupling through ``gg``; the integrator
restores the implicit gravity-wave coupling.  The dry geopotential of the
perturbation temperature is handled implicitly (``hydrostatic`` of the T
estimate inside the integrator), so the explicit divergence term keeps only the
*virtual* geopotential correction ``hydrostatic(Tv' - T')``.

Sign / index conventions
------------------------
* Grid arrays are ``(nlat, nlon, kmax)`` (optionally a leading member axis),
  latitudes north -> south, ``k=0`` the bottom sigma layer.
* ``sdot`` (= sigma-dot = Isca ``wg/ps``) is on half levels, ``kmax+1`` entries,
  zero at both boundaries.
"""

import numpy as np

from .backend import get_array_module, to_device
from .constants import EPSVT, KAPPA, OMEGA, RAIR


class BasicStateTerms:
    """Everything derived from the time-constant basic state."""

    def __init__(self, sp, vgrid, bs):
        self.sp = sp
        self.vg = vgrid
        self.xp = sp.xp
        xp = self.xp
        self.u = to_device(np.ascontiguousarray(bs.u, dtype=np.float64), xp)
        self.v = to_device(np.ascontiguousarray(bs.v, dtype=np.float64), xp)
        self.t = to_device(np.ascontiguousarray(bs.t, dtype=np.float64), xp)
        self.q = to_device(np.ascontiguousarray(bs.q, dtype=np.float64), xp)
        self.ps = to_device(np.ascontiguousarray(bs.ps, dtype=np.float64), xp)

        # Coriolis parameter f = 2*Omega*sin(lat)
        self.coriol = to_device((2.0 * OMEGA * sp.mu)[:, None], xp)

        # Basic-state vorticity and divergence
        self.vor, self.div_s = sp.uv_to_vordiv(self.u, self.v)
        self.vor_g = sp.wave_to_grid(self.vor)
        self.div = sp.wave_to_grid(self.div_s)

        # log surface pressure and its gradient
        self.lnps = xp.log(self.ps)
        self.pix, self.piy = sp.hgrad(sp.grid_to_wave(self.lnps))

        # Virtual temperature (basic)
        self.tv = self.t * (1.0 + EPSVT * self.q)

        # A_bar = v . grad(ln ps); column sum; sigma-dot
        self.padv = (self.u * self.pix[..., None]
                     + self.v * self.piy[..., None])
        self.ssum = _column_sum(self.div + self.padv, vgrid.dsig)
        self.sdot = _sigma_dot(self.ssum, vgrid.sigm)

        # Full basic-state Simmons-Burridge omega factor x5_bar
        self.x5 = (vgrid.w3 * self.ssum[..., 1:]
                   + vgrid.alpha * self.div
                   + vgrid.wP * self.padv)

        # Reference *virtual*-temperature profile -> semi-implicit operators.
        from .vertical import compute_tbar
        vgrid.set_reference_temperature(compute_tbar(self.tv))


def _column_sum(x, dsig):
    """S[k] = sum_{l>=k} dsig_l * x_l, with a trailing zero so that S[kmax]=0."""
    xp = get_array_module(x)
    w = x * dsig
    out = xp.zeros(x.shape[:-1] + (x.shape[-1] + 1,), dtype=x.dtype)
    out[..., :-1] = xp.flip(xp.cumsum(xp.flip(w, axis=-1), axis=-1), axis=-1)
    return out


def _sigma_dot(ssum, sigm):
    """sdot_k = sigm_k*S(0) - S(k); zero at both boundaries.

    In pure sigma this is Isca's half-level mass flux divided by ps
    (``wg = ps * sdot``); the two forms are algebraically identical.
    """
    sdot = sigm * ssum[..., :1] - ssum
    sdot[..., 0] = 0.0
    sdot[..., -1] = 0.0
    return sdot


def _vert_advect(field, sdot, dsig):
    """Isca SECOND_CENTERED, ADVECTIVE_FORM vertical advection tendency.

    ``rdt(k) = +[flux(k+1) - flux(k) - field(k)*(sdot(k+1)-sdot(k))] / dsig(k)``
    with ``flux(i) = sdot(i) * 0.5*(field(i)+field(i-1))`` and zero flux at the
    top/surface (``sdot`` is zero there).  Bilinear in ``(field, sdot)`` so the
    linearisation ``adv(f',s_bar) + adv(f_bar,s')`` is exact.
    (vert_advection.F90:184-193, 460-471; four_in_one wg boundary = 0.)

    Sign note: the shared ``_sigma_dot`` returns sigma-dot in the
    Watanabe--Kimoto convention (``sdot_k = sigm_k*S(0) - S(k)``), which is the
    opposite sign to Isca's half-level mass flux ``wg`` that the advective-form
    stencil above was written for.  Feeding the WK-convention ``sdot`` into the
    Isca stencil therefore requires the overall ``+`` sign here so that the
    tendency is the physical ``-sigma_dot * d(field)/d(sigma)``.  This term
    vanishes identically for isothermal / solid-body basic states
    (``d(basic)/d(sigma)=0`` and ``sdot_bar=0``), so it is only exercised by a
    realistic baroclinic basic state.
    """
    xp = get_array_module(field)
    kmax = field.shape[-1]
    lead = np.broadcast_shapes(field.shape[:-1], sdot.shape[:-1])
    flux = xp.zeros(lead + (kmax + 1,), dtype=field.dtype)
    flux[..., 1:kmax] = (sdot[..., 1:kmax] * 0.5
                         * (field[..., 1:kmax] + field[..., :kmax - 1]))
    dsdot = sdot[..., 1:] - sdot[..., :-1]
    return (flux[..., 1:] - flux[..., :-1] - field * dsdot) / dsig


class Dynamics:
    """Explicit (non-gravity-wave) tendencies of the linear Isca-core model."""

    def __init__(self, sp, vgrid, bst, strictly_linear=False):
        self.sp = sp
        self.vg = vgrid
        self.bs = bst
        self.strictly_linear = bool(strictly_linear)

    # ------------------------------------------------------------------
    def tendencies(self, st, frc_t=None, frc_vor=None, frc_div=None,
                   frc_ps=None, frc_q=None):
        """Explicit tendencies of (vor, div, T, pi, q) in spectral space."""
        sp, vg, bs = self.sp, self.vg, self.bs
        dsig, sigm = vg.dsig, vg.sigm
        alpha, w3, wP = vg.alpha, vg.w3, vg.wP
        tref = vg.tbar                               # reference Tv profile (1D)

        up, vp, tp, qp = st.u, st.v, st.t, st.q

        # --- perturbation vorticity / divergence ------------------------
        vor_s, div_s = sp.uv_to_vordiv(up, vp)
        vor = sp.wave_to_grid(vor_s)
        div = sp.wave_to_grid(div_s)

        # --- perturbation ln ps and its gradient ------------------------
        pi = st.lnps_pert()
        pix, piy = sp.hgrad(sp.grid_to_wave(pi))

        # --- A' = v.grad(ln ps); column sums; sigma-dot -----------------
        padv = (up * bs.pix[..., None] + bs.u * pix[..., None]
                + vp * bs.piy[..., None] + bs.v * piy[..., None])
        ssum = _column_sum(div + padv, dsig)
        sdot = _sigma_dot(ssum, sigm)
        gpid = -ssum[..., 0]

        # --- virtual temperature perturbation ---------------------------
        tv = tp + EPSVT * (tp * bs.q + bs.t * qp)

        # ================================================================
        # 1. surface pressure (continuity; divergence part is implicit)
        # ================================================================
        gt_pi = gpid + (div * dsig).sum(axis=-1)      # = -sum dsig * padv'
        if frc_ps is not None:
            gt_pi = gt_pi + frc_ps
        wt_pi = sp.grid_to_wave(gt_pi)

        # ================================================================
        # 2. temperature  (Isca four_in_one adiabatic + advection)
        # ================================================================
        # Full Simmons-Burridge omega factor of the perturbation.
        x5p = alpha * div + wP * padv + w3 * ssum[..., 1:]
        # Full linearised adiabatic term -kappa*Tv*x5 (uses full basic Tv).
        adiab = -KAPPA * (tv * bs.x5 + bs.tv * x5p)

        # horizontal advection of dry T (flux form; the reference constant
        # cancels exactly, so any per-level constant works -- we use tref)
        tdev = bs.t - tref
        xu = up * tdev + bs.u * tp
        xv = vp * tdev + bs.v * tp
        hadv = tp * bs.div + tdev * div

        # vertical advection of dry T (fully explicit)
        vadv = (_vert_advect(tp, bs.sdot, dsig)
                + _vert_advect(bs.t, sdot, dsig))

        # ``div @ tmtx.T`` removes the reference energy-conversion-from-div part
        # from the explicit tendency; the integrator restores it implicitly as
        # ``- tmtx @ dbar`` (tmtx carries the integrator's positive-diagonal
        # sign, so this subtraction is the physical ``-kappa*Tref*(...)``).
        gt_t = adiab + hadv + vadv + div @ vg.tmtx.T
        if frc_t is not None:
            gt_t = gt_t + frc_t
        wt_t = sp.neg_divergence(xu, xv) + sp.grid_to_wave(gt_t)

        # ================================================================
        # 3. momentum  (vorticity flux + PGF + vertical advection)
        # ================================================================
        absvor = bs.vor_g + bs.coriol[..., None]
        fu = vp * absvor + bs.v * vor
        fv = -up * absvor - bs.u * vor

        # pressure gradient: -R*cpg*Tv*grad(ln ps); reference part of the
        # perturbation-lnps gradient is implicit, so use (Tv_bar - Tref) there.
        rcpg = RAIR * vg.cpg
        tvdev = bs.tv - tref                           # Tv_bar - Tref (3D)
        fu -= rcpg * (tv * bs.pix[..., None] + tvdev * pix[..., None])
        fv -= rcpg * (tv * bs.piy[..., None] + tvdev * piy[..., None])

        # vertical advection of momentum (fully explicit)
        fu += _vert_advect(up, bs.sdot, dsig) + _vert_advect(bs.u, sdot, dsig)
        fv += _vert_advect(vp, bs.sdot, dsig) + _vert_advect(bs.v, sdot, dsig)

        # kinetic energy + virtual geopotential correction (dry geopotential
        # is implicit in the integrator)
        phiv = vg.hydrostatic(tv - tp)
        ke = phiv + up * bs.u + vp * bs.v
        if not self.strictly_linear:
            ke = ke + 0.5 * (up * up + vp * vp)

        wt_vor, wt_div = sp.uv_to_vordiv(fu, fv)
        wt_div = wt_div - sp.laplacian(sp.grid_to_wave(ke))
        if frc_vor is not None:
            wt_vor = wt_vor + sp.grid_to_wave(frc_vor)
        if frc_div is not None:
            wt_div = wt_div + sp.grid_to_wave(frc_div)

        # ================================================================
        # 4. humidity  (tracer advection; dynamics only)
        # ================================================================
        yu = up * bs.q + bs.u * qp
        yv = vp * bs.q + bs.v * qp
        g = qp * bs.div + bs.q * div
        g = g + _vert_advect(qp, bs.sdot, dsig) + _vert_advect(bs.q, sdot, dsig)
        if frc_q is not None:
            g = g + frc_q
        wt_q = sp.neg_divergence(yu, yv) + sp.grid_to_wave(g)

        # ================================================================
        # 5. omega diagnostic  wg_full = -x5 * p_full, p_full = sig*ps  [hPa/s]
        # ================================================================
        omega = -vg.sig * (x5p * bs.ps[..., None] + bs.x5 * st.ps[..., None])

        self.diag = {"vor": vor, "div": div, "vor_s": vor_s, "div_s": div_s,
                     "omega": omega, "sdot": sdot, "pi": pi}
        return wt_vor, wt_div, wt_t, wt_pi, wt_q
