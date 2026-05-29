"""
Measure the multiprocessing scaling of the COS IM stage and plot it.

Two measurements (all wall-clock, on this machine):
  (a) speedup vs worker count for a persistent-pool "suite" workload (several
      logic-tree branches), the realistic full-run case;
  (b) throughput vs problem size, single-process vs pool, showing that tiny jobs
      lose to the per-worker startup while large jobs win.

Pin one BLAS thread per worker so N workers do not oversubscribe the cores; we
set VECLIB_MAXIMUM_THREADS / OMP_NUM_THREADS=1 here before importing numpy.

    python scripts/benchmark_multiprocessing.py
"""

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")    # Apple Accelerate
os.environ.setdefault("OMP_NUM_THREADS", "1")           # OpenBLAS / MKL

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_W = {}


def _init():
    sys.path.insert(0, str(ROOT))
    from saavg_cos import model, spectral_im
    _W.update(model=model, si=spectral_im, gmm=model.load_ground_motion_model("central"))


def _task(payload):
    mags, dists, rank, order = payload
    sc = _W["si"].compute_scenarios(_W["gmm"], np.asarray(mags), np.asarray(dists),
                                    pca_rank=rank, gh_order=order, s2s_mode="epistemic")
    return float(sc.k1.sum())


def run_single(gmm, si, mags, dists, rank, order, branches):
    si.compute_scenarios(gmm, mags[:2], dists[:2], pca_rank=rank, gh_order=order)  # warmup
    t0 = time.perf_counter()
    for _ in range(branches):
        si.compute_scenarios(gmm, mags, dists, pca_rank=rank, gh_order=order, s2s_mode="epistemic")
    return time.perf_counter() - t0


def run_pool(mags, dists, rank, order, branches, workers, chunks=3):
    blocks = [c for c in np.array_split(mags, chunks) if len(c)]
    tasks = [(list(c), list(dists), rank, order) for _ in range(branches) for c in blocks]
    ctx = __import__("multiprocessing").get_context("spawn")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init) as ex:
        list(ex.map(_task, tasks))
    return time.perf_counter() - t0


def main():
    from saavg_cos import model, spectral_im
    gmm = model.load_ground_motion_model("central")
    ncpu = os.cpu_count() or 8

    # (a) speedup vs workers, suite of branches at a fixed grid
    mags = np.linspace(2.5, 7.0, 26); dists = np.linspace(3.0, 40.0, 26)
    branches = 8
    t1 = run_single(gmm, spectral_im, mags, dists, 2, 17, branches)
    worker_grid = sorted(set([2, 4, 8, 12, min(16, ncpu), ncpu]))
    worker_grid = [w for w in worker_grid if 2 <= w <= ncpu]
    times = [run_pool(mags, dists, 2, 17, branches, w) for w in worker_grid]
    speedup = [t1 / t for t in times]
    print("suite speedup (order17, 26x26, 8 branches):  single = %.2fs" % t1)
    for w, t, s in zip(worker_grid, times, speedup):
        print(f"  workers={w:2d}: {t:6.2f}s  speedup x{s:.2f}")

    # (b) throughput vs problem size, single vs pool
    sizes = [11, 21, 31, 41]
    Z = gmm.zones.size
    thr_single, thr_pool = [], []
    for g in sizes:
        mg = np.linspace(2.5, 7.0, g); dd = np.linspace(3.0, 40.0, g)
        ts = run_single(gmm, spectral_im, mg, dd, 2, 7, 1)
        tp = run_pool(mg, dd, 2, 7, 1, min(16, ncpu))
        zsc = Z * g * g
        thr_single.append(zsc / ts); thr_pool.append(zsc / tp)
    print("\nthroughput (order7) zone-scenarios/s:")
    for g, a, b in zip(sizes, thr_single, thr_pool):
        print(f"  {g}x{g}: single {a:>10,.0f}   pool {b:>10,.0f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(worker_grid, speedup, "o-", color="C0", label="measured")
    ax[0].plot([1, max(worker_grid)], [1, max(worker_grid)], "k--", lw=1, alpha=0.5, label="ideal linear")
    ax[0].set_xlabel("worker processes"); ax[0].set_ylabel("speedup vs single process")
    ax[0].set_title(f"(a) persistent-pool suite speedup\n(order17, 26x26 grid, {branches} branches, {ncpu} cores)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    x = np.array([Z * g * g for g in sizes])
    ax[1].loglog(x, thr_single, "s-", label="single process")
    ax[1].loglog(x, thr_pool, "o-", label=f"pool ({min(16, ncpu)} workers)")
    ax[1].set_xlabel("problem size (zone-scenarios)"); ax[1].set_ylabel("throughput (zone-scenarios/s)")
    ax[1].set_title("(b) throughput vs problem size (order7)\nsmall jobs lose to startup; large jobs win")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = ROOT / "figures" / "fig_mp_scaling.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
