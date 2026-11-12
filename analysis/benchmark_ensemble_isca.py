#!/usr/bin/env python3
"""Ensemble/batch mode on PyLBM-Isca: parity + throughput (roadmap B2, Fig 7).

Isca-core counterpart of LBM_GPU_GMD/analysis/benchmark_ensemble.py.

Part A -- parity (correctness):
  * E identical-forcing members reproduce the single-member run to round-off;
  * E distinct-forcing members reproduce E separate single-member runs.

Part B -- throughput: batched integration for E = 1..512 on the GPU; wall-clock
per step, GPU memory, and 35-day experiments/hour vs ensemble size.

Writes <manuscript>/figures/ensemble_throughput.pdf (+ .png) and
<manuscript>/analysis/ensemble.json.
"""
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ISCA_ROOT = os.path.dirname(HERE)
sys.path.insert(0, ISCA_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from pylbm_isca import Config, LBM, Spectral, VerticalGrid          # noqa: E402
from pylbm_isca.backend import asnumpy                              # noqa: E402
from pylbm_isca.ensemble import (BatchedSpectral, EnsembleForcing,  # noqa: E402
                               build_heating_sweep, make_ensemble_model)

try:
    import cupy
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

MANU = os.path.normpath(os.path.join(ISCA_ROOT, "..", "LBM_GPU_GMD"))
FIGDIR = os.path.join(MANU, "figures")
ANADIR = os.path.join(MANU, "analysis")
os.makedirs(FIGDIR, exist_ok=True)
STEPS_PER_DAY = 72
RUN_DAYS_35 = 35.0


class ConstForcing:
    def __init__(self, gtf, xp):
        self.gtf = xp.asarray(gtf)

    def step(self):
        return self.gtf


def resting_bs(sp, vg, t0=250.0):
    s = (sp.nlat, sp.nlon, vg.kmax)
    return types.SimpleNamespace(
        u=np.zeros(s), v=np.zeros(s), t=np.full(s, t0),
        q=np.zeros(s), ps=np.full((sp.nlat, sp.nlon), 1000.0))


def gaussian_heat(sp, vg, lon0=180.0, lat0=0.0, amp=2.0, width=20.0):
    lon = np.degrees(sp.lon)[None, :]
    lat = sp.lat_deg[:, None]
    horiz = np.exp(-((lon - lon0) ** 2 + (lat - lat0) ** 2) / (2 * width ** 2))
    prof = np.exp(-((np.arange(vg.kmax) - 9) / 4.0) ** 2)
    return (amp / 86400.0) * horiz[:, :, None] * prof[None, None, :]


def sync():
    if _HAS_CUPY:
        cupy.cuda.runtime.deviceSynchronize()


def max_rel_diff(a, b):
    a, b = asnumpy(a), asnumpy(b)
    scale = np.abs(a).max() or 1.0
    return float(np.abs(a - b).max() / scale)


def parity(device="gpu"):
    print("=== Part A: parity ===")
    sp = Spectral(device=device)
    vg = VerticalGrid(device=device)
    bs = resting_bs(sp, vg)
    base = gaussian_heat(sp, vg)
    days = 4.0

    m0 = LBM(bs, sp, vg, Config(ndays=days))
    m0.run(mjo_forcing=ConstForcing(base, sp.xp), writer=None, verbose=False)
    ref = {k: asnumpy(getattr(m0.ga, k)) for k in ("u", "v", "t", "ps")}

    E = 4
    mE, spb, vgb = make_ensemble_model(bs, E, Config(ndays=days), device)
    heat_id = np.broadcast_to(base, (E,) + base.shape).copy()
    mE.run(mjo_forcing=EnsembleForcing(heat_id, spb.xp), verbose=False)
    d_id = max(max_rel_diff(ref[k], asnumpy(getattr(mE.ga, k))[e])
               for k in ref for e in range(E))
    print(f"  (1) E={E} identical members vs single: max rel diff = {d_id:.2e}")

    E2 = 3
    heat_sw = build_heating_sweep(base, E2, mode="longitude", xp=np)
    mS, spb2, vgb2 = make_ensemble_model(bs, E2, Config(ndays=days), device)
    mS.run(mjo_forcing=EnsembleForcing(heat_sw, spb2.xp), verbose=False)
    worst = 0.0
    for e in range(E2):
        sp_e = Spectral(device=device)
        vg_e = VerticalGrid(device=device)
        me = LBM(resting_bs(sp_e, vg_e), sp_e, vg_e, Config(ndays=days))
        me.run(mjo_forcing=ConstForcing(np.asarray(heat_sw[e]), sp_e.xp),
               verbose=False)
        for k in ref:
            worst = max(worst, max_rel_diff(asnumpy(getattr(me.ga, k)),
                                            asnumpy(getattr(mS.ga, k))[e]))
    print(f"  (2) E={E2} distinct members vs {E2} separate runs: "
          f"max rel diff = {worst:.2e}")
    ok = d_id < 1e-8 and worst < 1e-8
    print(f"  parity: {'PASS' if ok else 'FAIL'} (threshold 1e-8)")
    return dict(identical=d_id, distinct=worst, passed=ok)


def throughput(device="gpu", sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
    print("\n=== Part B: throughput ===")
    rows = []
    warm_days, timed_days = 0.3, 1.0
    for E in sizes:
        if _HAS_CUPY:
            cupy.get_default_memory_pool().free_all_blocks()
        try:
            sp0 = Spectral(device=device)
            vg0 = VerticalGrid(device=device)
            bs = resting_bs(sp0, vg0)
            base = gaussian_heat(sp0, vg0)
            heat = build_heating_sweep(base, E, mode="longitude", xp=np)

            mW, spb, vgb = make_ensemble_model(bs, E, Config(ndays=warm_days),
                                               device)
            frc = EnsembleForcing(heat, spb.xp)
            mW.run(mjo_forcing=frc, verbose=False)
            sync()

            mT, spb, vgb = make_ensemble_model(bs, E, Config(ndays=timed_days),
                                               device)
            frc = EnsembleForcing(heat, spb.xp)
            sync()
            t0 = time.time()
            mT.run(mjo_forcing=frc, verbose=False)
            sync()
            dt = time.time() - t0
        except Exception as exc:
            print(f"  E={E:4d}: stopped ({type(exc).__name__})")
            break

        nsteps = int(round(timed_days * STEPS_PER_DAY))
        ms_step = dt / nsteps * 1e3
        run35_s = ms_step * RUN_DAYS_35 * STEPS_PER_DAY / 1e3
        runs_per_hr = E / run35_s * 3600.0
        mem = (cupy.get_default_memory_pool().total_bytes() / 1024 / 1024
               if _HAS_CUPY else np.nan)
        rows.append(dict(E=E, ms_step=ms_step, run35_s=run35_s,
                         runs_per_hr=runs_per_hr, mem_mib=mem))
        print(f"  E={E:4d}: {ms_step:7.2f} ms/step  {run35_s:6.1f}s/35-day-batch"
              f"  {runs_per_hr:8.1f} runs/hr  {mem:7.0f} MiB")
    return rows


def make_figure(rows):
    E = np.array([r["E"] for r in rows])
    thru = np.array([r["runs_per_hr"] for r in rows])
    ms = np.array([r["ms_step"] for r in rows])
    mem = np.array([r["mem_mib"] for r in rows])
    thru1 = thru[0]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    axL.loglog(E, thru, "o-", color="tab:blue", lw=2, ms=6, label="measured")
    axL.loglog(E, thru1 * E, "k:", lw=1.5, label="ideal (linear in E)")
    axL.set_xlabel("ensemble size (experiments per integration)")
    axL.set_ylabel("throughput [35-day experiments / hour]")
    axL.set_title("(a) GPU throughput vs ensemble size (T42L20)",
                  loc="left", fontsize=10)
    axL.grid(True, which="both", alpha=0.3); axL.legend(frameon=False)

    axR.semilogx(E, ms / E, "s-", color="tab:green", lw=2, ms=6,
                 label="per-member cost")
    axR.set_xlabel("ensemble size")
    axR.set_ylabel("GPU time per member per step [ms]", color="tab:green")
    axR.tick_params(axis="y", labelcolor="tab:green")
    axR.set_title("(b) per-member cost and GPU memory", loc="left", fontsize=10)
    axR.grid(True, which="both", alpha=0.3)
    ax2 = axR.twinx()
    ax2.semilogx(E, mem / 1024, "^--", color="tab:red", lw=1.6, ms=5)
    ax2.set_ylabel("peak GPU memory [GiB]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "ensemble_throughput.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p[:-4] + ".png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")


def _worker_timing(days):
    """One single-thread single-member timed run; prints 'ms_step=<value>'.

    Launched as a subprocess with OMP/MKL threads pinned to 1 so that N of these
    running concurrently emulate N independent members on a fully loaded CPU node.
    """
    sp = Spectral(device="cpu"); vg = VerticalGrid(device="cpu")
    bs = resting_bs(sp, vg); base = gaussian_heat(sp, vg)
    LBM(bs, sp, vg, Config(ndays=0.3)).run(
        mjo_forcing=ConstForcing(base, sp.xp), verbose=False)          # warm-up
    mT = LBM(bs, sp, vg, Config(ndays=days))
    t0 = time.time()
    mT.run(mjo_forcing=ConstForcing(base, sp.xp), verbose=False)
    dt = time.time() - t0
    print(f"ms_step={dt / (days * STEPS_PER_DAY) * 1e3:.6f}")


def cpu_throughput(nprocs, days=2.0):
    """Measure 35-day experiments/hour with `nprocs` concurrent 1-thread runs."""
    import subprocess
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--days", str(days)]
    print(f"\n=== CPU-node throughput: {nprocs} concurrent single-thread runs ===")
    t0 = time.time()
    procs = [subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
             for _ in range(nprocs)]
    outs = [p.communicate()[0] for p in procs]
    wall = time.time() - t0
    ms = [float(l.split("=")[1]) for o in outs for l in o.splitlines()
          if l.startswith("ms_step=")]
    run35 = np.median(ms) * RUN_DAYS_35 * STEPS_PER_DAY / 1e3        # s, under load
    thr = nprocs / run35 * 3600.0
    print(f"  per-run (under load): {np.median(ms):.1f} ms/step, {run35:.1f} s/35-day")
    print(f"  node throughput ({nprocs} procs): {thr:.0f} experiments/hour")
    return dict(nprocs=nprocs, ms_step_median=float(np.median(ms)),
                run35_s=float(run35), runs_per_hr=float(thr))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--cpu-procs", type=int, default=0,
                    help="measure CPU-node throughput with this many concurrent "
                         "single-thread processes (fully-utilized-node baseline)")
    args = ap.parse_args()

    if args.worker:
        _worker_timing(args.days)
    elif args.cpu_procs:
        res = cpu_throughput(args.cpu_procs, days=args.days)
        with open(os.path.join(HERE, "cpu_node_throughput.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {os.path.join(HERE, 'cpu_node_throughput.json')}")
    else:
        par = parity("gpu")
        rows = throughput("gpu")
        if rows:
            make_figure(rows)
        with open(os.path.join(ANADIR, "ensemble.json"), "w") as fh:
            json.dump(dict(parity=par, throughput=rows, model="PyLBM-Isca"), fh,
                      indent=2)
        print(f"wrote {os.path.join(ANADIR, 'ensemble.json')}")
