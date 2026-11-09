# Copyright (C) 2026  LBM_isca authors
#
# This file is part of LBM_isca, a Linear Baroclinic Model whose dynamical core
# is a linearisation of the Isca / GFDL ``atmos_spectral`` primitive equations.
#
# LBM_isca is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.  The physical constants and the Simmons-Burridge discretisation it
# re-derives come from the GPLv3 Isca sources (github.com/ExeClim/Isca).
"""Physical constants and resolution parameters for the Isca-core LBM.

Unlike :mod:`lbm.constants` (which carried the CCSR/NIES AGCM numbers), the
values here are GFDL/Isca's, taken from

  src/shared/constants/constants.F90          (ExeClim/Isca)

so that the linearised core reproduces Isca's primitive-equation dynamics.
In Isca these are *namelist-overridable variables*, not Fortran parameters
(``constants_nml``), which is how Isca runs non-Earth planets; we take the
Earth defaults.

The horizontal/vertical resolution (T42, 20 pure-sigma levels) and the sigma
half levels are kept identical to :mod:`lbm.constants` on purpose: this makes
the Isca-core model directly comparable to the Watanabe-Kimoto port and lets it
reuse the same NCEP basic-state pipeline (:mod:`lbm_isca.basicstate`) and the
same GrADS output records.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants  (constants.F90; Earth defaults)
# ---------------------------------------------------------------------------
# Earth radius.  Isca's constants.F90:258 default is 6376.0e3, but the FMS doc
# header and most of the literature use 6371 km, and the current LBM grid uses
# 6371 km.  We default to 6371 km so the spherical-harmonic grid matches the
# Watanabe-Kimoto port term for term; pass ``Spectral(radius=6.376e6)`` to
# reproduce Isca's own value exactly.
ER = 6.371e6          # earth radius [m]            (constants.F90:258, see note)
GRAV = 9.80           # gravity [m s-2]             (constants.F90:83,259)
RDGAS = 287.04        # gas constant, dry air       (constants.F90:84,269)
RVGAS = 461.50        # gas constant, water vapour  (constants.F90:124,270)
KAPPA = 2.0 / 7.0     # R/Cp                        (constants.F90:85,278)
CP = RDGAS / KAPPA    # = 1004.64 J kg-1 K-1        (constants.F90:86,279)

# Isca's rotation rate is the sidereal value EARTH_OMEGA (constants.F90:254,260),
# which differs from the Watanabe-Kimoto port's 7.2722e-5 (derived from a 24 h
# solar day).  If &constants_nml sets rotation_period>0, Isca recomputes
# OMEGA = 2*pi/rotation_period (constants.F90:317-318); we take the default.
OMEGA = 7.2921150e-5  # rotation rate [rad s-1]     (constants.F90:254,260)

# Aliases / derived quantities kept for API parity with lbm.constants.
RAIR = RDGAS                     # the dry gas constant, R
EPSV = RDGAS / RVGAS             # 0.62197...
# Virtual-temperature factor: Tv = T (1 + EPSVT q), EPSVT = Rv/Rd - 1 ~ 0.608
# (spectral_dynamics.F90:440, virtual_factor = rvgas/rdgas - 1).
EPSVT = RVGAS / RDGAS - 1.0      # 0.60781...

DAYLEN_SEC = 86400.0
S2D = 86400.0         # seconds per day, used by the forcing / damping scaling

# gt2gr-compatible height-anomaly conversion (output.py); these are output
# constants, not dynamics, so they stay as in the reference writer.
RHO00 = 1.225
GRAV_GT2GR = 9.80

# ---------------------------------------------------------------------------
# Resolution: T42 L20  (kept identical to lbm.constants for comparability)
# ---------------------------------------------------------------------------
IMAX = 128            # longitudes
JMAX = 64             # Gaussian latitudes
KMAX = 20             # sigma levels
NMAX = 42             # triangular truncation
MMAX = NMAX
LMAX = NMAX
MINT = 1

# IJDIM is the unweighted point count used by the reference-temperature mean
# (see lbm_isca.vertical.compute_tbar); the duplicated wrap-around column means
# IDIM = IMAX+1.  Carried for exact parity with the reference TBAR.
IDIM = IMAX + 1
IJDIM = IDIM * JMAX

# Complex spectral coefficients for triangular truncation T(NMAX).
NSPEC = (NMAX + 1) * (NMAX + 2) // 2

# ---------------------------------------------------------------------------
# Sigma half levels, L20.  Index 0 = surface (sigma=1), KMAX = top (sigma=0).
# Pure sigma (pk=0): p_half(k) = SIGM[k] * ps, matching Isca's even/uneven_sigma
# option with bk(k) = SIGM[k] (vert_coordinate.F90:230-273).  We keep the
# Watanabe-Kimoto L20 half levels so the basic state interpolates identically.
# ---------------------------------------------------------------------------
SIGM = np.array([
    1.000, 0.990, 0.970, 0.930, 0.870,
    0.790, 0.700, 0.600, 0.500, 0.410,
    0.330, 0.260, 0.200, 0.150, 0.100,
    0.070, 0.050, 0.040, 0.030, 0.020, 0.000,
], dtype=np.float64)

# NCEP 17 pressure levels of the input basic state [hPa].
PLEV_NCEP17 = np.array([
    1000., 925., 850., 700., 600., 500., 400., 300., 250.,
    200., 150., 100., 70., 50., 30., 20., 10.,
], dtype=np.float64)

# Nominal output pressure levels (only used for an optional sigma->p output).
PLEV_OUT = np.array([
    1000., 950., 900., 850., 700., 600., 500., 400., 300., 250.,
    200., 150., 100., 70., 50., 30., 20., 10., 7., 5.,
], dtype=np.float64)
