# Copyright (C) 2026  PyLBM-Isca authors
#
# This file is part of PyLBM-Isca and is distributed under the GNU General Public
# License v3 (or later).  The discretisation re-derives the Simmons & Burridge
# (1981) energy-conserving vertical scheme as implemented in the GPLv3 Isca /
# GFDL ``atmos_spectral`` core (github.com/ExeClim/Isca).
"""Sigma-coordinate discretisation -- Simmons & Burridge (Isca/GFDL) scheme.

This replaces the Arakawa-Suarez vertical operators of :mod:`lbm.vertical`
with the **Simmons & Burridge (1981)** energy-conserving differencing that the
GFDL spectral core (and hence Isca) uses, ported from

  src/atmos_spectral/model/press_and_geopot.F90   pressure_variables_3d,
                                                   compute_geopotential
  src/atmos_spectral/model/spectral_dynamics.F90  four_in_one
  src/atmos_spectral/model/implicit.F90           linear_* operators

The vertical coordinate is **pure sigma** (``pk = 0``), so ``p_half = SIGM*ps``
and *every* Simmons-Burridge coefficient (alpha, the log-pressure weights, the
pressure-gradient weight and the hydrostatic matrix) is a ps-independent
geometric constant -- exactly like the Arakawa alpha/beta were -- so they drop
straight into the same per-wavenumber semi-implicit machinery.

Index convention (0-based, bottom-up): ``k = 0`` is the *bottom* layer
(sigma ~ 0.995) and ``k = KMAX-1`` the top.  Half level ``k`` sits *below* full
level ``k``: ``sigm[k] > sig[k] > sigm[k+1]``.  For layer ``k`` the bottom
(larger sigma) interface is ``sigm[k] = SIGM[k]`` and the top (smaller sigma)
interface is ``sigm[k+1] = SIGM[k+1]``.

GFDL ordering has ``k=1`` at the top; the translation is ``bk_gfdl(j) <->
SIGM[KMAX+1-j]``.  The massless-top convention (``p_half=0`` at the model top)
is reproduced term for term: ``alpha`` of the top layer is 1 (GFDL
``ln_top_level_factor=-1``), and every place the ``ln(p_half=0)`` sentinel
could leak in it is multiplied by the top half-sigma ``SIGM[KMAX]=0`` or by a
column sum that is zero at the top, so the limits are taken analytically here
(the offending coefficients are stored as 0 and only ever multiply 0).

GPU note
--------
The 20x20 operators are assembled once on the host and moved onto the device
(``self.xp``); :meth:`hydrostatic` is the single GEMM ``phi = Tv @ WW.T``.
"""

import numpy as np

from .backend import asnumpy, get_array_module, resolve, to_device
from .constants import KAPPA, KMAX, RAIR, SIGM


class VerticalGrid:
    """Simmons-Burridge sigma coordinate and its vertical operator matrices."""

    # resolution-only (ps-independent) operators moved to the device
    _DEVICE_ATTRS = ("dsig", "sig", "sigm", "dsigm", "alpha", "dlog3",
                     "cpg", "w3", "wP", "ww")
    # TBAR-dependent operators (need the reference profile)
    _DEVICE_ATTRS_TBAR = ("tbar", "gg", "tmtx", "whmtx", "gcmtx")

    def __init__(self, sigm=SIGM, kappa=KAPPA, device="auto"):
        self.xp = resolve(device)
        self.sigm = np.asarray(sigm, dtype=np.float64)
        self.kmax = len(self.sigm) - 1
        self.kappa = kappa

        sm, sp = self.sigm[:-1], self.sigm[1:]    # bottom (large) / top (small)
        self.dsig = sm - sp                        # layer thickness in sigma > 0

        # --- Simmons-Burridge alpha (press_and_geopot.F90:176,189) ----------
        # alpha(k) = 1 - sig_top*(ln sig_bot - ln sig_top)/(sig_bot - sig_top),
        # taking the sig_top->0 limit (sig_top*ln sig_top -> 0) => alpha=1 at
        # the massless top layer (GFDL ln_top_level_factor = -1).
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sp > 0.0,
                             sp * (np.log(sm) - np.log(np.where(sp > 0, sp, 1.0)))
                             / (sm - sp), 0.0)
        self.alpha = 1.0 - ratio                   # alpha[top] = 1

        # full-level sigma: ln p_full = ln p_half(bottom) - alpha
        self.sig = sm * np.exp(-self.alpha)

        # --- log-pressure weights (four_in_one:1071-1073) -------------------
        # dlog_1 = ln p_half(bottom) - ln p_full = alpha
        # dlog_2 = ln p_full - ln p_half(top)            (top: *0, store 0)
        # dlog_3 = ln p_half(bottom) - ln p_half(top)    (top: *0, store 0)
        self.dlog1 = self.alpha.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            dlog2 = np.where(sp > 0.0, np.log(self.sig) - np.log(
                np.where(sp > 0, sp, 1.0)), 0.0)
            dlog3 = np.where(sp > 0.0, np.log(sm) - np.log(
                np.where(sp > 0, sp, 1.0)), 0.0)
        self.dlog2 = dlog2
        self.dlog3 = dlog3

        # --- pressure-gradient weight (four_in_one x1 -> per-lnps form) -----
        # momentum PGF: -R*Tv*cpg*grad(ln ps),  cpg = (sig_bot*dlog_1 +
        # sig_top*dlog_2)/dsig  (top: sig_top=0 => cpg = sig_bot*alpha/dsig).
        self.cpg = (sm * self.dlog1 + sp * self.dlog2) / self.dsig

        # weights entering the Simmons-Burridge omega factor x5:
        #   x5 = w3*S_above + alpha*div + wP*padv,   w3 = dlog_3/dsig,
        #   wP = alpha - cpg   (coefficient of padv = v.grad ln ps)
        self.w3 = self.dlog3 / self.dsig           # w3[top] = 0 (dlog3[top]=0)
        self.wP = self.alpha - self.cpg

        # DSIGM (half-to-half spacing) for the implicit vertical diffusion.
        self.dsigm = np.empty(self.kmax + 1)
        self.dsigm[0] = self.sigm[0] - self.sig[0]
        self.dsigm[1:self.kmax] = self.sig[:-1] - self.sig[1:]
        self.dsigm[self.kmax] = self.sig[-1] - self.sigm[-1]

        # --- hydrostatic matrix WW (compute_geopotential:350-356) ----------
        # phi_full[k] = sum_{l<k} R*dlog_3[l]*Tv[l] + R*alpha[k]*Tv[k]
        ww = np.zeros((self.kmax, self.kmax))
        for k in range(self.kmax):
            for l in range(self.kmax):
                if l < k:
                    ww[k, l] = RAIR * self.dlog3[l]
                elif l == k:
                    ww[k, l] = RAIR * self.alpha[k]
        self.ww = ww

        # TBAR-dependent operators are built later (need the basic state).
        self.tbar = None
        self.gg = None
        self.tmtx = None
        self.whmtx = None
        self.gcmtx = None

        self._push(self._DEVICE_ATTRS)

    # ------------------------------------------------------------------
    def _push(self, names):
        for name in names:
            val = getattr(self, name)
            if val is not None:
                setattr(self, name, to_device(np.asarray(val), self.xp))

    # ------------------------------------------------------------------
    def set_reference_temperature(self, tbar):
        """Build the reference (TBAR-dependent) semi-implicit operators.

        Parameters
        ----------
        tbar : (KMAX,) array
            Reference *virtual*-temperature profile (the horizontal mean of the
            basic-state Tv).  The semi-implicit gravity-wave matrix is built
            about this profile -- the proper LBM choice -- rather than Isca's
            isothermal 300 K ``ref_temperature_implicit``.

        Operators (names match what :mod:`pylbm_isca.integrator` consumes):

        * ``gg``   = R * Tbar * cpg -- the reference pressure-gradient weight
          ``h1`` (implicit.F90:pres_grad_funct).  In pure sigma the second part
          ``h2`` of Isca's ``h = h1 + h2`` vanishes (a uniform shift of ln p
          leaves the hydrostatic integral unchanged), so ``gg = h1``.
        * ``tmtx`` = **minus** the reference energy-conversion operator:
          ``tmtx @ div = +kappa*Tbar*(alpha*div + w3*S_above(div))``.  The sign
          follows the Watanabe-Kimoto integrator convention (its ``tmtx`` has a
          positive diagonal): the dynamics carries the *full* explicit energy
          conversion and adds ``div @ tmtx.T`` to *remove* the reference part,
          which the integrator then restores implicitly as ``- tmtx @ dbar``
          (i.e. ``-kappa*Tbar*(...)`` evaluated at the implicit divergence
          ``dbar``).  The vertical advection of the reference profile is kept
          *explicit* here -- a valid and stable semi-implicit split; the fast
          gravity waves (divergence <-> geopotential <-> compression) stay
          implicit.
        * ``whmtx`` = WW @ tmtx ;  ``gcmtx`` = outer(gg, dsig).
        """
        kmax = self.kmax
        tbar = np.asarray(asnumpy(tbar), dtype=np.float64).copy()
        alpha = asnumpy(self.alpha)
        w3 = asnumpy(self.w3)
        cpg = asnumpy(self.cpg)
        dsig = asnumpy(self.dsig)
        ww = asnumpy(self.ww)

        self.tbar = tbar
        self.gg = RAIR * tbar * cpg

        # Energy-conversion operator (integrator sign convention, positive
        # diagonal): (tmtx @ div)[k] =
        #   +kappa*Tbar[k]*( alpha[k]*div[k] + w3[k]*sum_{l>k} dsig[l]*div[l] )
        tmtx = np.zeros((kmax, kmax))
        for k in range(kmax):
            tmtx[k, k] = KAPPA * tbar[k] * alpha[k]
            for l in range(k + 1, kmax):
                tmtx[k, l] = KAPPA * tbar[k] * w3[k] * dsig[l]
        self.tmtx = tmtx

        self.whmtx = ww @ tmtx
        self.gcmtx = np.outer(self.gg, dsig)

        self._push(self._DEVICE_ATTRS_TBAR)

    # ------------------------------------------------------------------
    def hydrostatic(self, tv):
        """Geopotential from (virtual) temperature: ``phi = Tv @ WW.T``.

        Works on any array whose last axis is the vertical one (grid or
        spectral).  Surface geopotential is zero (flat lower boundary).
        """
        tv = self.xp.asarray(tv)
        return tv @ self.ww.T


def compute_tbar(gft):
    """Reference profile: the plain arithmetic mean over all grid points.

    Backend-agnostic; returns an array on the same device as ``gft``.  (Unlike
    the CCSR port we do not reproduce the duplicated wrap-around column: the
    reference profile is only a linearisation anchor, so a clean grid mean is
    the natural choice.)
    """
    xp = get_array_module(gft)
    nlat, nlon, kmax = gft.shape
    return gft.sum(axis=(0, 1)) / (nlat * nlon)
