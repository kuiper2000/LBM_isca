#!/usr/bin/env python3
"""Shared helpers for reading GrADS (.grd) output and comparing two runs.

All model outputs compared here share the same Fortran-sequential, big-endian,
YREV GrADS record layout, so the same reader works for every run and the
comparison is apples-to-apples.
"""
import numpy as np

IMAX, JMAX, KMAX = 128, 64, 20
NPTS = IMAX * JMAX
REC_LEN = NPTS + 2                         # 4-byte marker + data + 4-byte marker
VARS = [("psi", KMAX), ("chi", KMAX), ("u", KMAX), ("v", KMAX),
        ("w", KMAX), ("t", KMAX), ("z", KMAX), ("ps", 1)]
RECS_PER_DAY = sum(n for _, n in VARS)     # 141


def read_grd(path):
    """Return (ndays, 141, JMAX, IMAX) float32 array from a .grd file."""
    raw = np.fromfile(path, dtype=">f4")
    rec = raw.reshape(-1, REC_LEN)
    data = rec[:, 1:-1].reshape(-1, JMAX, IMAX)
    ndays = data.shape[0] // RECS_PER_DAY
    return data[:ndays * RECS_PER_DAY].reshape(ndays, RECS_PER_DAY, JMAX, IMAX)


def var_slice(day_block, name):
    """All levels of one variable (shape (nlev, JMAX, IMAX)) from one day."""
    off = 0
    for vn, n in VARS:
        if vn == name:
            return day_block[off:off + n]
        off += n
    raise KeyError(name)


def field(data, name, lev, d0, d1):
    """Time-mean of one variable at one level over days [d0, d1) (0-based)."""
    acc = None
    nd = data.shape[0]
    d1 = min(d1, nd)
    for d in range(d0, d1):
        v = var_slice(data[d], name)
        sl = v[lev] if v.shape[0] > 1 else v[0]
        acc = sl.astype(np.float64) if acc is None else acc + sl
    return acc / (d1 - d0)


def metrics(ref, test):
    """Pattern correlation and RMS/max differences of test relative to ref."""
    a = np.asarray(ref, dtype=np.float64).ravel()
    b = np.asarray(test, dtype=np.float64).ravel()
    diff = a - b
    rms = float(np.sqrt(np.mean(diff ** 2)))
    maxdiff = float(np.max(np.abs(diff)))
    scale = float(np.sqrt(np.mean(a ** 2)))
    amp = float(np.max(np.abs(a)))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    return {
        "corr": corr,
        "rel_rms": rms / scale if scale else np.nan,
        "rel_max": maxdiff / amp if amp else np.nan,
        "rms": rms,
        "maxdiff": maxdiff,
        "amp_ref": amp,
        "amp_test": float(np.max(np.abs(b))),
    }


def rms_field(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
