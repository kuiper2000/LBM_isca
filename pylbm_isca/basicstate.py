"""Basic-state preprocessing: ERA5 pressure levels -> model sigma levels.

This follows the structure of the basic-state preprocessing in the
Watanabe-Kimoto linear baroclinic model, using standard published algorithms.
It reads a multi-month ERA5 pressure-level file, time-averages it, derives
surface pressure from geopotential height (and an optional topography field),
spectrally smooths, and interpolates from pressure to sigma with a natural
cubic spline (Press et al., 2007, Numerical Recipes, Sect. 3.3).

Pipeline
--------
================  =====================================  ====================
step              what it does                           here
================  =====================================  ====================
read / average    read levels, average over months       :func:`read_pressure_levels`
Ps from Z         surface pressure from Z and Zs          :func:`z2ps`
spectral smooth   grid -> wave -> grid truncation         :func:`specsm`
zonal mean/eddy   split into zonal mean and eddy          :func:`zmsw`
p -> sigma        vertical interpolation                  :func:`p2s`
cubic spline      natural cubic spline                    :func:`spline`
================  =====================================  ====================

Two notes:

* Spectral smoothing is a grid -> wave -> grid round trip truncated at the
  model's own resolution, i.e. a **T42** truncation -- ``trunc`` is exposed in
  case you want to check.
* When no topography field is supplied, ``zs`` defaults to zero (a flat lower
  boundary).  That is consistent with the model's dynamics, which set the
  topographic forcing to zero.  Pass a real surface-height field to ``zs`` to
  include orography.
"""

from dataclasses import dataclass

import numpy as np

from .backend import asnumpy
from .constants import IMAX, JMAX, KMAX, PLEV17

UNDEF_IN = -9.999e20      # input undefined-value sentinel
XMISS = -999.0            # internal missing-value marker

# Variable order inside the pressure-level file.  Relative humidity (2) and
# omega (7) are read past and discarded.
BS_VARS = ["z", "rhum", "q", "t", "u", "v", "omega"]
BS_KEEP = ["z", "q", "t", "u", "v"]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_pressure_levels(path, kmo=1, navg=3, nlon=IMAX, nlat=JMAX, nlev=17,
              nvar=7, nmonth=None, yflip=True):
    """Read and time-average the ERA5 pressure-level basic-state file.

    Layout (verified against ``bs.1979.bin``): Fortran sequential records,
    big-endian ``real*4``, ``(nlon, nlat)`` per record, ordered
    ``month -> variable -> level``.

    ``yflip=True`` converts the file's south-first rows into the model's
    north-first order.

    Returns a dict of ``(nlat, nlon, nlev)`` arrays for z, q, t, u, v.
    """
    npts = nlon * nlat
    raw = np.fromfile(path, dtype=">f4")
    if raw.size % (npts + 2):
        raise ValueError(f"{path}: not a whole number of Fortran records")
    rec = raw.reshape(-1, npts + 2)
    markers = rec[:, 0].view(">i4")
    if not np.all(markers == npts * 4):
        raise ValueError(f"{path}: bad record markers - wrong endianness "
                         f"or grid size? (got {markers[0]}, want {npts*4})")
    data = rec[:, 1:-1].reshape(-1, nlat, nlon).astype(np.float64)

    per_month = nvar * nlev
    nmo_file = data.shape[0] // per_month
    if data.shape[0] % per_month:
        raise ValueError(
            f"{path}: {data.shape[0]} records is not a multiple of "
            f"{per_month} (= {nvar} vars x {nlev} levels)")
    nmonth = nmo_file if nmonth is None else nmonth

    out = {v: np.zeros((nlat, nlon, nlev)) for v in BS_KEEP}
    miss = {v: np.zeros((nlat, nlon, nlev), dtype=bool) for v in BS_KEEP}

    issn = kmo
    for _ in range(navg):
        base = (issn - 1) * per_month
        if base + per_month > data.shape[0]:
            raise ValueError(f"{path}: month {issn} beyond end of file")
        for iv, name in enumerate(BS_VARS):
            if name not in BS_KEEP:
                continue
            blk = data[base + iv * nlev: base + (iv + 1) * nlev]   # (lev,lat,lon)
            blk = np.moveaxis(blk, 0, -1)                          # (lat,lon,lev)
            if yflip:
                blk = blk[::-1, :, :]
            miss[name] |= (blk == UNDEF_IN)
            out[name] += blk / float(navg)
        issn += 1
        if issn > nmo_file:
            issn -= nmo_file

    for name in BS_KEEP:
        if miss[name].any():
            out[name][miss[name]] = XMISS
    return out


# ---------------------------------------------------------------------------
# Surface pressure
# ---------------------------------------------------------------------------
def z2ps(z, zs, plev=PLEV17):
    """Surface pressure from geopotential height.

    Scans upward for the first level whose height exceeds the surface height,
    clamps that index to ``[2, nlev]`` and interpolates pressure linearly in
    ``ln p`` to the surface height.

    Parameters
    ----------
    z  : (nlat, nlon, nlev) geopotential height [gpm], level 0 = 1000 hPa
    zs : (nlat, nlon) surface height [m]
    plev : (nlev,) pressure levels [hPa]

    Returns ps in hPa.
    """
    z = np.asarray(z, dtype=np.float64)
    nlev = z.shape[-1]
    zs = np.broadcast_to(np.asarray(zs, dtype=np.float64), z.shape[:-1])

    above = z > zs[..., None]
    any_above = above.any(axis=-1)
    # first True along the level axis; where none, the index is left at
    # its initial value NLEV (the DO loop ran to completion).
    ka = np.where(any_above, above.argmax(axis=-1), nlev - 1)
    ka = np.clip(ka, 1, nlev - 1)                 # MIN(MAX(KA,2),NLEV), 0-based

    lp = np.log(plev)
    al = lp[ka - 1]
    bl = lp[ka] - lp[ka - 1]
    zk = np.take_along_axis(z, ka[..., None], axis=-1)[..., 0]
    zkm = np.take_along_axis(z, (ka - 1)[..., None], axis=-1)[..., 0]
    cl = zs - zkm
    dl = zk - zkm
    return np.exp(al + bl * cl / dl)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def specsm(field, sp):
    """Spectral smoothing: grid -> wave -> grid.

    ``sp.truncate`` may run on the GPU; the basic-state pipeline stays on the
    host, so the result is brought back with :func:`asnumpy` (a no-op on a CPU
    backend).  The smoothing is one-off setup, so the device round trip is
    irrelevant to runtime.
    """
    return asnumpy(sp.truncate(field))


def zmsw(field, ozm=False, osw=False):
    """Zonal mean (``ozm``) or stationary-wave (``osw``) filter (``ZMSW``)."""
    if not (ozm or osw):
        return field
    zm = field.mean(axis=1, keepdims=True)
    return np.broadcast_to(zm, field.shape).copy() if ozm else field - zm


# ---------------------------------------------------------------------------
# Vertical interpolation
# ---------------------------------------------------------------------------
def spline(x, z, xi, omisb=False, omist=False, xmiss=XMISS):
    """Natural cubic spline (Press et al., 2007, Numerical Recipes, Sect. 3.3).

    ``x`` is a shared, strictly monotonic abscissa (here ``ln p``, which is
    *descending*); ``z`` holds one data column per row; ``xi`` holds the
    target abscissae per row.  Outside the data range the scheme clamps to
    the end value unless the corresponding ``omis`` flag is set.

    Note the forward sweep of the tridiagonal solve depends only on ``x``, so
    its coefficients are scalars shared by every column; only ``u`` and the
    back substitution are per-column.

    Parameters
    ----------
    x  : (k,)       shared abscissa
    z  : (n, k)     data
    xi : (n, l)     target abscissae
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    k = x.size

    # --- tridiagonal forward sweep -------------------------------------
    y2f = np.zeros(k)                  # scalar decomposition coefficients
    u = np.zeros(z.shape)              # per-column right-hand side
    for i in range(1, k - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2f[i - 1] + 2.0
        y2f[i] = (sig - 1.0) / p
        u[:, i] = (6.0 * ((z[:, i + 1] - z[:, i]) / (x[i + 1] - x[i])
                          - (z[:, i] - z[:, i - 1]) / (x[i] - x[i - 1]))
                   / (x[i + 1] - x[i - 1]) - sig * u[:, i - 1]) / p

    # --- back substitution; QN = UN = 0 makes this a natural spline -----
    y2 = np.zeros(z.shape)
    y2[:, k - 1] = 0.0
    for i in range(k - 2, -1, -1):
        y2[:, i] = y2f[i] * y2[:, i + 1] + u[:, i]

    # --- locate the bracketing interval --------------------------------
    # find the first K with X(K) < XI(L); x is descending, so flip
    # the sign to use a standard ascending search.
    xr = -x
    ku = np.searchsorted(xr, -xi.ravel(), side="right").reshape(xi.shape)

    below = ku == 0                    # target pressure beyond the bottom level
    above = ku == k                    # ... beyond the top level
    kc = np.clip(ku, 1, k - 1)

    h = x[kc] - x[kc - 1]
    a = (x[kc] - xi) / h
    b = (xi - x[kc - 1]) / h
    zkm = np.take_along_axis(z, kc - 1, axis=1)
    zk = np.take_along_axis(z, kc, axis=1)
    y2km = np.take_along_axis(y2, kc - 1, axis=1)
    y2k = np.take_along_axis(y2, kc, axis=1)
    out = (a * zkm + b * zk
           + (a * (a * a - 1.0) * y2km + b * (b * b - 1.0) * y2k) * h * h / 6.0)

    out = np.where(below, xmiss if omisb else z[:, :1], out)
    out = np.where(above, xmiss if omist else z[:, -1:], out)
    return out


def p2s(gdi, ps, sig, plev=PLEV17, xmiss=XMISS):
    """Pressure -> sigma interpolation.

    Parameters
    ----------
    gdi  : (nlat, nlon, nlevp) field on pressure levels
    ps   : (nlat, nlon) surface pressure [hPa]
    sig  : (kmax,) sigma full levels
    """
    gdi = np.asarray(gdi, dtype=np.float64)
    nlat, nlon, nlevp = gdi.shape
    cols = gdi.reshape(-1, nlevp)
    piln = np.log(np.asarray(plev, dtype=np.float64))
    poln = np.log(ps.reshape(-1, 1) * np.asarray(sig)[None, :])
    out = spline(piln, cols, poln, xmiss=xmiss)
    return out.reshape(nlat, nlon, len(sig))


# ---------------------------------------------------------------------------
# Container and entry point
# ---------------------------------------------------------------------------
@dataclass
class BasicState:
    """Basic state on the model's sigma levels, north-first Gaussian grid."""

    u: np.ndarray            # (nlat, nlon, kmax) m/s
    v: np.ndarray            # (nlat, nlon, kmax) m/s
    t: np.ndarray            # (nlat, nlon, kmax) K
    q: np.ndarray            # (nlat, nlon, kmax) kg/kg
    ps: np.ndarray           # (nlat, nlon) hPa
    lat_deg: np.ndarray = None
    lon_deg: np.ndarray = None
    sig: np.ndarray = None
    source: str = ""

    def summary(self):
        print(f"basic state from {self.source}")
        print(f"  grid {self.t.shape[0]} lat x {self.t.shape[1]} lon "
              f"x {self.t.shape[2]} sigma levels")
        rows = [("u", self.u, "m/s"), ("v", self.v, "m/s"),
                ("t", self.t, "K"), ("q", self.q, "kg/kg"),
                ("ps", self.ps, "hPa")]
        for name, a, unit in rows:
            print(f"  {name:3s} min {a.min():11.4f}  max {a.max():11.4f}  "
                  f"mean {a.mean():11.4f}   [{unit}]")
        if not all(np.all(np.isfinite(a))
                   for a in (self.u, self.v, self.t, self.q, self.ps)):
            print("  !! non-finite values present")
        return self


def load_basic_state(path, sp, vgrid, kmo=1, navg=3, zs=None,
                     ozm=False, osw=False, smooth=True, yflip=True,
                     plev=PLEV17):
    """Full basic-state pipeline: read -> average -> Ps -> smooth -> p2s.

    Order of operations: surface pressure is smoothed *before* the vertical
    interpolation, while u, v, T and q are smoothed *after* it.
    """
    raw = read_pressure_levels(path, kmo=kmo, navg=navg, nlon=sp.nlon, nlat=sp.nlat,
                    nlev=len(plev), yflip=yflip)

    if zs is None:
        zs = np.zeros((sp.nlat, sp.nlon))
    ps = z2ps(raw["z"], zs, plev)
    if smooth:
        ps = specsm(ps, sp)
    ps = zmsw(ps, ozm, osw)

    # The whole preprocessing runs on the host; vgrid.sig may be a device
    # array, so take a host copy for the pressure->sigma interpolation.
    sig_host = asnumpy(vgrid.sig)

    fields = {}
    for name in ("u", "v", "t", "q"):
        f = p2s(raw[name], ps, sig_host, plev)
        if smooth:
            f = specsm(f, sp)
        fields[name] = zmsw(f, ozm, osw)

    return BasicState(u=fields["u"], v=fields["v"], t=fields["t"],
                      q=fields["q"], ps=ps, lat_deg=sp.lat_deg,
                      lon_deg=sp.lon_deg, sig=sig_host,
                      source=path)
