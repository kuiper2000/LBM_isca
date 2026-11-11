"""Readers for the two forcing channels of this LBM build.

1. The canonical *steady* forcing (``&nmsfrc fsfrc=``).  For the dry model the
   file holds
   ``3*KMAX+1 = 61`` records per case:
   ``vorticity(20) | divergence(20) | temperature(20) | ln(Ps)(1)``.

2. The *time-varying* MJO temperature forcing, which reads a file of
   ``ND=73`` days x ``KMAX=20`` levels and linearly interpolates it onto
   ``NT=72`` sub-steps per day.  This channel *replaces* the steady
   temperature forcing.

All files are Fortran sequential-unformatted, big-endian, ``real*4``, with
one ``(nlon, nlat)`` record per level (longitude varying fastest).

Latitude order
--------------
The model's internal order is north-first (j=0 is +87.86 deg), which is why
every GrADS ``.ctl`` in the package carries ``YREV``.  ``MJO_FORCE3`` applies
no flip, so it assumes the file is already north-first.

The two input families do **not** share a convention, so check which one you
are reading:

* ``bs.YYYY.bin`` (the basic state) comes through the usual GrADS route
  and *is* south-first; it must be flipped, which is why
  :func:`pylbm_isca.basicstate.load_basic_state` keeps ``yflip=True``.  This is
  verifiable from the data: months 1-3 of ``bs.1979.bin`` are DJF, and only
  the flipped reading puts the stronger subtropical jet in the Northern
  Hemisphere (45.3 m/s at +29.3 deg, against 30.0 m/s in the SH).
* ``frc.phaseN.bin`` (the MJO forcing) is written north-first, so it is read
  with no flip -- ``yflip=False``, the default here and for
  :meth:`SteadyForcing.from_file`.  That makes the no-flip read correct as it stands.
"""

import os

import numpy as np

from .constants import IMAX, JMAX, KMAX

_REC_DTYPE = ">f4"


def read_fortran_records(path, nlon=IMAX, nlat=JMAX, count=None, offset=0):
    """Read Fortran sequential records of ``nlon*nlat`` real*4 values.

    Returns ``(nrec, nlat, nlon)`` with the record markers verified.
    """
    npts = nlon * nlat
    raw = np.fromfile(path, dtype=_REC_DTYPE)
    if raw.size % (npts + 2):
        raise ValueError(
            f"{path}: size {raw.size*4} is not a whole number of "
            f"{npts}-value Fortran records")
    rec = raw.reshape(-1, npts + 2)
    markers = rec[:, 0].view(">i4")
    if not np.all(markers == npts * 4):
        bad = int(np.argmax(markers != npts * 4))
        raise ValueError(
            f"{path}: bad record marker at record {bad} "
            f"({markers[bad]} != {npts*4}); wrong endianness or record length?")
    rec = rec[offset: None if count is None else offset + count, 1:-1]
    return rec.reshape(-1, nlat, nlon)


def _orient(a, yflip):
    return a[:, ::-1, :] if yflip else a


class SteadyForcing:
    """vorticity / divergence / temperature / ln(Ps) forcing, time constant."""

    def __init__(self, vor, div, t, ps):
        self.vor, self.div, self.t, self.ps = vor, div, t, ps

    @classmethod
    def zeros(cls, nlat=JMAX, nlon=IMAX, nlev=KMAX):
        z3 = np.zeros((nlat, nlon, nlev))
        return cls(z3.copy(), z3.copy(), z3.copy(), np.zeros((nlat, nlon)))

    @classmethod
    def from_file(cls, path, nlev=KMAX, ncase=1, yflip=False,
                  nlon=IMAX, nlat=JMAX):
        per_case = 3 * nlev + 1
        rec = read_fortran_records(path, nlon, nlat,
                                   count=per_case, offset=(ncase - 1) * per_case)
        if rec.shape[0] != per_case:
            raise ValueError(f"{path}: expected {per_case} records, "
                             f"got {rec.shape[0]}")
        rec = _orient(rec, yflip)
        mv = np.moveaxis
        return cls(mv(rec[0:nlev], 0, -1).copy(),
                   mv(rec[nlev:2 * nlev], 0, -1).copy(),
                   mv(rec[2 * nlev:3 * nlev], 0, -1).copy(),
                   rec[3 * nlev].copy())


class MJOForcing:
    """Time-varying MJO temperature forcing.

    The call sequence:

    * The first ``spinup`` calls hold the day-1 field set during
      initialisation.
    * From ``spinup`` onward the field marches through the file, linearly
      interpolated between consecutive days over ``nt`` sub-steps:
      ``work(L) = gtf(it) + (gtf(it+1)-gtf(it))/nt * (L-1)``.
    """

    def __init__(self, path, nlev=KMAX, ndays=73, nt=72, spinup=435,
                 yflip=False, nlon=IMAX, nlat=JMAX, scale=1.0):
        rec = read_fortran_records(path, nlon, nlat)
        expect = ndays * nlev
        if rec.shape[0] != expect:
            raise ValueError(
                f"{path}: expected {expect} records ({ndays} days x {nlev} "
                f"levels), got {rec.shape[0]}.  Check ndays/nlev.")
        rec = _orient(rec, yflip) * scale
        # (day, lev, lat, lon) -> (day, lat, lon, lev)
        self.gtf = np.moveaxis(rec.reshape(ndays, nlev, nlat, nlon), 1, -1)
        self.ndays, self.nt, self.spinup = ndays, nt, spinup
        self.path = path
        self.reset()

    def reset(self):
        self.counter = 0
        self.counter2 = 0
        self.it = 1                  # 1-based day index, as in the Fortran
        self.spinning = True
        self._work = None
        self._field = self.gtf[0].copy()      # day-1 field held during spin-up

    def step(self):
        """Advance one dynamics call and return the current forcing field."""
        self.counter += 1
        if self.spinning:
            if self.counter < self.spinup:
                return self._field
            self.counter2 = 0
            self.it = 1
            self.spinning = False

        if self.counter2 == 0:
            it = min(self.it, self.ndays)
            nxt = min(self.it + 1, self.ndays)
            a = self.gtf[it - 1]
            ka = (self.gtf[nxt - 1] - a) / self.nt
            self._work = a[None, ...] + ka[None, ...] * np.arange(
                self.nt)[:, None, None, None]

        self._field = self._work[self.counter2]
        self.counter2 += 1
        if self.counter2 == self.nt:
            self.counter2 = 0
            self.it += 1
        return self._field

    @property
    def day(self):
        return self.it


def autodetect(path, nlev=KMAX, nlon=IMAX, nlat=JMAX):
    """Guess which of the two forcing layouts a file uses, from its size."""
    npts = nlon * nlat
    nrec = os.path.getsize(path) // (npts * 4 + 8)
    per_case = 3 * nlev + 1
    info = {"nrec": nrec}
    if nrec % per_case == 0:
        info["steady_cases"] = nrec // per_case
    if nrec % nlev == 0:
        info["mjo_days"] = nrec // nlev
    return info
