"""Spectral regridding between two Gaussian grids / truncations.

Option 2 of the horizontal-resolution work: the shipped input files (basic
state ``bs.*.bin`` and forcing ``frc.*.bin``) are stored on the native
T42 128x64 Gaussian grid, but the solver core is resolution-agnostic.  To run
on a *different* grid we move each input field through spectral space:

    native grid --grid_to_wave--> spectral (native T) --resize--> spectral
    (target T) --wave_to_grid--> target grid

Because the CCSR/NIES spherical-harmonic normalisation is independent of the
grid and the truncation, a coefficient ``xi[m, n]`` means the same thing on
both grids, so resizing is just zero-padding (upscale) or dropping the high-n
tail (downscale).  The fields are inherently band-limited to the native
truncation, so an upscale adds no new small-scale information -- it produces a
*consistent* basic state / forcing at the finer grid, nothing more.

Velocity components (u, v) are not smooth scalars at the poles, so they are
regridded through vorticity/divergence (``uv_to_vordiv`` -> resize ->
``vordiv_to_uv``) rather than as two independent scalars.  Everything else
(temperature, humidity, ln Ps, and the vorticity/divergence *forcing* fields,
which are already scalars) goes through the scalar path.

Grid detection
--------------
:func:`detect_grid` recovers a file's native grid from its Fortran record
marker, assuming the usual ``nlon = 2*nlat`` aspect, so the driver can decide
per file whether a regrid is even needed (it is not when the file already
matches the requested grid).
"""

import os

import numpy as np

from .backend import asnumpy

_REC_DTYPE = ">f4"


# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------
def detect_grid(path):
    """Return ``(nlon, nlat)`` of a Fortran-sequential field file.

    Reads the first record marker (``npts*4`` bytes) and assumes the standard
    Gaussian aspect ``nlon == 2*nlat``.  Raises if the marker does not factor
    into such a grid, in which case the caller must pass the grid explicitly.
    """
    with open(path, "rb") as fh:
        marker = int(np.frombuffer(fh.read(4), dtype=">i4")[0])
    if marker <= 0 or marker % 4:
        raise ValueError(f"{path}: unreadable record marker {marker}")
    npts = marker // 4
    nlat = int(round((npts / 2.0) ** 0.5))
    nlon = 2 * nlat
    if nlon * nlat != npts:
        raise ValueError(
            f"{path}: {npts} points per record is not a 2:1 Gaussian grid; "
            f"pass the source grid explicitly")
    return nlon, nlat


# ---------------------------------------------------------------------------
# Spectral coefficient resize
# ---------------------------------------------------------------------------
def _resize_spec(xi, dst_nmax, xp):
    """Zero-pad or truncate a spectral array ``[m, n, ...]`` to ``dst_nmax``."""
    src_nmax = xi.shape[0] - 1
    if src_nmax == dst_nmax:
        return xi
    k = min(src_nmax, dst_nmax) + 1
    out = xp.zeros((dst_nmax + 1, dst_nmax + 1) + xi.shape[2:], dtype=xi.dtype)
    out[:k, :k] = xi[:k, :k]
    return out


# ---------------------------------------------------------------------------
# Field regridding
# ---------------------------------------------------------------------------
def regrid_scalar(src_sp, dst_sp, field):
    """Spectrally interpolate a scalar grid field (2-D or 3-D) between grids.

    Returns a host (numpy) array to match the basic-state/forcing pipeline.
    """
    if (src_sp.nlon, src_sp.nlat, src_sp.nmax) == \
       (dst_sp.nlon, dst_sp.nlat, dst_sp.nmax):
        return np.asarray(field)
    xp = src_sp.xp
    xi = src_sp.grid_to_wave(field)
    xi = _resize_spec(xi, dst_sp.nmax, xp)
    return asnumpy(dst_sp.wave_to_grid(xi))


def regrid_uv(src_sp, dst_sp, u, v):
    """Regrid a horizontal wind (u, v) through vorticity/divergence."""
    if (src_sp.nlon, src_sp.nlat, src_sp.nmax) == \
       (dst_sp.nlon, dst_sp.nlat, dst_sp.nmax):
        return np.asarray(u), np.asarray(v)
    xp = src_sp.xp
    vor, div = src_sp.uv_to_vordiv(u, v)
    vor = _resize_spec(vor, dst_sp.nmax, xp)
    div = _resize_spec(div, dst_sp.nmax, xp)
    u2, v2 = dst_sp.vordiv_to_uv(vor, div)
    return asnumpy(u2), asnumpy(v2)


def regrid_basic_state(bs, src_sp, dst_sp):
    """Return a new :class:`~lbm.basicstate.BasicState` on ``dst_sp``'s grid."""
    from .basicstate import BasicState

    u, v = regrid_uv(src_sp, dst_sp, bs.u, bs.v)
    return BasicState(
        u=u, v=v,
        t=regrid_scalar(src_sp, dst_sp, bs.t),
        q=regrid_scalar(src_sp, dst_sp, bs.q),
        ps=regrid_scalar(src_sp, dst_sp, bs.ps),
        lat_deg=dst_sp.lat_deg, lon_deg=dst_sp.lon_deg,
        sig=bs.sig, source=bs.source + " (regridded)")


def regrid_mjo_forcing(mjo, src_sp, dst_sp):
    """Regrid an :class:`~lbm.forcing.MJOForcing` in place (temperature only)."""
    if (src_sp.nlon, src_sp.nlat, src_sp.nmax) == \
       (dst_sp.nlon, dst_sp.nlat, dst_sp.nmax):
        return mjo
    ndays, nlat, nlon, nlev = mjo.gtf.shape
    # (day, lat, lon, lev) -> (lat, lon, day*lev) for a single batched transform
    stack = np.transpose(mjo.gtf, (1, 2, 0, 3)).reshape(nlat, nlon, ndays * nlev)
    reg = regrid_scalar(src_sp, dst_sp, stack)
    mjo.gtf = np.transpose(
        reg.reshape(dst_sp.nlat, dst_sp.nlon, ndays, nlev), (2, 0, 1, 3))
    mjo.reset()
    return mjo


def regrid_steady_forcing(steady, src_sp, dst_sp):
    """Regrid a :class:`~lbm.forcing.SteadyForcing` in place.

    The vorticity/divergence forcing fields are grid-space scalars here (not
    wind components), so all four channels take the scalar path.
    """
    if (src_sp.nlon, src_sp.nlat, src_sp.nmax) == \
       (dst_sp.nlon, dst_sp.nlat, dst_sp.nmax):
        return steady
    steady.vor = regrid_scalar(src_sp, dst_sp, steady.vor)
    steady.div = regrid_scalar(src_sp, dst_sp, steady.div)
    steady.t = regrid_scalar(src_sp, dst_sp, steady.t)
    steady.ps = regrid_scalar(src_sp, dst_sp, steady.ps)
    return steady
