#!/usr/bin/env python3
"""Fair performance benchmark for PyLBM-Isca (roadmap B2, Fig 6).

Isca-core counterpart of LBM_GPU_GMD/analysis/benchmark_perf.py.  Same honest
methodology (device sync, warm-up, peak GPU memory, setup time separated from the
loop), but times the linearised *Isca* core (moist, Simmons-Burridge vertical
operators), whose cost per step differs from the Watanabe-Kimoto port.

Writes figures/benchmark_<tag>.json into the manuscript analysis/figures/ dir so
make_perf_figure.py reads it unchanged.

Usage:
  python benchmark_perf_isca.py                          # multi-thread NumPy + GPU
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python benchmark_perf_isca.py --tag single-thread  # single-thread NumPy
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ISCA_ROOT = os.path.dirname(HERE)
sys.path.insert(0, ISCA_ROOT)

from pylbm_isca import Config, LBM, Spectral, VerticalGrid, gpu_available  # noqa: E402

try:
    import cupy
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

# write JSONs where the manuscript's make_perf_figure.py looks for them
OUTDIR = os.path.normpath(
    os.path.join(ISCA_ROOT, "..", "LBM_GPU_GMD", "analysis", "figures"))
os.makedirs(OUTDIR, exist_ok=True)

RESOLUTIONS = [("T21", 21, 64, 32), ("T42", 42, 128, 64),
               ("T63", 63, 192, 96), ("T85", 85, 256, 128),
               ("T106", 106, 320, 160)]
RUN_DAYS = 35.0
STEPS_PER_DAY = 72
TOTAL_STEPS = int(RUN_DAYS * STEPS_PER_DAY)       # 2520


def thread_count():
    for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        v = os.environ.get(var)
        if v:
            return int(v)
    try:
        from threadpoolctl import threadpool_info
        info = threadpool_info()
        if info:
            return max(i["num_threads"] for i in info)
    except Exception:
        pass
    return os.cpu_count()


def resting(sp, vg, t0=250.0):
    s = (sp.nlat, sp.nlon, vg.kmax)
    return types.SimpleNamespace(
        u=np.zeros(s), v=np.zeros(s), t=np.full(s, t0),
        q=np.zeros(s), ps=np.full((sp.nlat, sp.nlon), 1000.0))


def sync(dev):
    if dev == "gpu" and _HAS_CUPY:
        cupy.cuda.runtime.deviceSynchronize()


def time_run(nmax, nlon, nlat, dev, warm_days, timed_days):
    """Return (ms/step loop, setup_seconds, peak_gpu_MiB)."""
    if dev == "gpu" and _HAS_CUPY:
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.cuda.runtime.deviceSynchronize()

    ts = time.time()
    sp = Spectral(nlon=nlon, nlat=nlat, nmax=nmax, device=dev)
    vg = VerticalGrid(device=dev)
    m0 = LBM(resting(sp, vg), sp, vg, Config(ndays=warm_days, device=dev))
    sync(dev)
    setup_s = time.time() - ts                    # tables + first matrix build

    m0.run(verbose=False)                         # warm kernels / FFT plans
    sync(dev)

    m = LBM(resting(sp, vg), sp, vg, Config(ndays=timed_days, device=dev))
    sync(dev)
    t0 = time.time()
    m.run(verbose=False)
    sync(dev)
    dt = time.time() - t0
    nsteps = int(round(timed_days * STEPS_PER_DAY))
    ms = dt / nsteps * 1e3

    peak = np.nan
    if dev == "gpu" and _HAS_CUPY:
        peak = cupy.get_default_memory_pool().total_bytes() / 1024 / 1024
    return ms, setup_s, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None,
                    help="label for this run (default: '<nthreads>-thread')")
    args = ap.parse_args()

    nthr = thread_count()
    tag = args.tag or f"{nthr}-thread"
    have_gpu = gpu_available()
    print(f"[PyLBM-Isca]  threads={nthr}  gpu={have_gpu}  tag={tag}")

    rows = []
    for label, nmax, nlon, nlat in RESOLUTIONS:
        ndof = (nmax + 1) * (nmax + 2) // 2
        cpu_ms, cpu_setup, _ = time_run(
            nmax, nlon, nlat, "cpu", warm_days=0.05,
            timed_days=0.2 if nmax <= 63 else 0.1)
        gpu_ms = gpu_setup = gpu_mem = np.nan
        if have_gpu:
            gpu_ms, gpu_setup, gpu_mem = time_run(
                nmax, nlon, nlat, "gpu", warm_days=0.3, timed_days=1.5)
        rows.append(dict(res=label, nlon=nlon, nlat=nlat, ndof=ndof,
                         cpu_ms=cpu_ms, gpu_ms=gpu_ms,
                         cpu_setup_s=cpu_setup, gpu_setup_s=gpu_setup,
                         gpu_peak_mib=gpu_mem))
        sp = cpu_ms / gpu_ms if have_gpu else float("nan")
        print(f"  {label:4s} {nlon:4d}x{nlat:<4d} ndof={ndof:6d} | "
              f"CPU {cpu_ms:8.2f} ms/step (setup {cpu_setup:4.1f}s) | "
              f"GPU {gpu_ms:7.2f} ms/step (setup {gpu_setup:4.1f}s, "
              f"{gpu_mem:6.0f} MiB) | speedup {sp:5.1f}x")

    print(f"\n--- 35-day ({TOTAL_STEPS} steps) total time-to-solution ---")
    print(f"{'res':6s}{'CPU loop':>11s}{'CPU+setup':>11s}"
          f"{'GPU loop':>10s}{'GPU+setup':>11s}{'GPUmem':>9s}")
    for r in rows:
        cl = r["cpu_ms"] * TOTAL_STEPS / 1e3
        gl = r["gpu_ms"] * TOTAL_STEPS / 1e3
        print(f"{r['res']:6s}{cl:9.1f}s{cl + r['cpu_setup_s']:10.1f}s"
              f"{gl:8.1f}s{gl + r['gpu_setup_s']:10.1f}s"
              f"{r['gpu_peak_mib']:8.0f}M")

    out = dict(tag=tag, threads=nthr, total_steps=TOTAL_STEPS,
               run_days=RUN_DAYS, rows=rows, model="PyLBM-Isca")
    p = os.path.join(OUTDIR, f"benchmark_{tag}.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
