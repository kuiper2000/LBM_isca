#!/usr/bin/env python3
"""Compare two LBM .grd runs over the standard days-15-30 window.

Usage:
    python compare_runs.py REF.grd TEST.grd --label "Fortran vs Isca" \
           --out metrics.json

Reports pattern correlation and relative RMS / max differences for the key
upper-tropospheric and mid-tropospheric fields, plus the day-by-day RMS
difference of 200 hPa streamfunction (to show whether the residual is flat or
amplifies).  Used both for CPU-vs-GPU parity (expect ~1e-9) and for
Fortran-vs-Isca (different dynamical cores -> larger, documented difference).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # PyLBM-Isca root
sys.path.insert(0, HERE)

from grdlib import read_grd, field, metrics, var_slice, rms_field  # noqa: E402
from pylbm_isca import VerticalGrid                                  # noqa: E402


def level_indices():
    sig = np.asarray(VerticalGrid(device="cpu").sig).ravel()
    return {
        "k200": int(np.argmin(np.abs(sig - 0.175))),   # ~200 hPa
        "k500": int(np.argmin(np.abs(sig - 0.500))),   # ~500 hPa
        "k850": int(np.argmin(np.abs(sig - 0.850))),   # ~850 hPa
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("test")
    p.add_argument("--label", default="")
    p.add_argument("--d0", type=int, default=15, help="window start day (1-based)")
    p.add_argument("--d1", type=int, default=30, help="window end day (inclusive)")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    lev = level_indices()
    ref = read_grd(args.ref)
    test = read_grd(args.test)
    nd = min(ref.shape[0], test.shape[0])
    d0, d1 = args.d0 - 1, args.d1          # 0-based [d0, d1)

    fields = {
        "psi_200": ("psi", lev["k200"]),
        "u_200":   ("u",   lev["k200"]),
        "v_200":   ("v",   lev["k200"]),
        "t_500":   ("t",   lev["k500"]),
        "z_200":   ("z",   lev["k200"]),
        "ps":      ("ps",  0),
    }
    table = {}
    for key, (name, k) in fields.items():
        a = field(ref, name, k, d0, d1)
        b = field(test, name, k, d0, d1)
        table[key] = metrics(a, b)

    # day-by-day 200 hPa psi RMS difference (relative to Fortran amplitude)
    ts = {"day": [], "amp_ref": [], "rms_diff": [], "rel_diff": []}
    for d in range(nd):
        a = var_slice(ref[d], "psi")[lev["k200"]].astype(np.float64)
        b = var_slice(test[d], "psi")[lev["k200"]].astype(np.float64)
        amp = rms_field(a)
        rd = rms_field(a - b)
        ts["day"].append(d + 1)
        ts["amp_ref"].append(amp)
        ts["rms_diff"].append(rd)
        ts["rel_diff"].append(rd / amp if amp else np.nan)

    out = {
        "label": args.label,
        "ref": os.path.abspath(args.ref),
        "test": os.path.abspath(args.test),
        "window_days": [args.d0, args.d1],
        "ndays_compared": nd,
        "level_indices": lev,
        "fields": table,
        "psi200_timeseries": ts,
    }

    # console summary
    print(f"\n=== {args.label} ===")
    print(f"ref  = {args.ref}")
    print(f"test = {args.test}")
    print(f"window days {args.d0}-{args.d1}, ndays available {nd}")
    print(f"{'field':10s} {'corr':>16s} {'rel_rms':>12s} {'rel_max':>12s}")
    for key, m in table.items():
        print(f"{key:10s} {m['corr']:16.12f} {m['rel_rms']:12.3e} "
              f"{m['rel_max']:12.3e}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
