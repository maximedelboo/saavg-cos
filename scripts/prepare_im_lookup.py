"""
Build the COS intensity-measure lookup over the full GMM-V7 logic tree.

For every (branch, zone, magnitude, distance) this computes the compact SaAvg
representation -- the three leading cumulants (k1, k2, k3) -- with the COS
cumulant engine, and writes a chunked/compressed zarr. The full SaAvg
distribution is recovered on demand by COS inversion (``saavg_cos.cos_cache``);
nothing is histogrammed to disk.

Parallelism: a persistent process pool pays the per-worker spawn/import/config
cost once and streams (branch x magnitude-block) tasks, each worker pinned to a
single BLAS thread. Launch with VECLIB_MAXIMUM_THREADS=1 (Apple Accelerate) or
OMP_NUM_THREADS=1 (OpenBLAS/MKL) so the workers do not oversubscribe the cores.

Example
-------
    VECLIB_MAXIMUM_THREADS=1 python scripts/prepare_im_lookup.py \
        --workers 16 --nm 51 --nr 81 --rank 2 --order 7
"""

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---- worker side (separate process) --------------------------------------
_W = {}


def _init(rank, order, s2s_mode, dists):
    sys.path.insert(0, str(ROOT))
    from saavg_cos import model, spectral_im
    _W.update(model=model, spectral_im=spectral_im, gc=model.load_gmm_config(),
              rank=rank, order=order, s2s_mode=s2s_mode, dists=np.asarray(dists))


def _task(payload):
    branch_flat, btuple, m0, m1, mag_vals = payload
    gmm = _W["model"].build_branch_gmm(_W["gc"], *btuple)
    sc = _W["spectral_im"].compute_scenarios(
        gmm, np.asarray(mag_vals), _W["dists"], pca_rank=_W["rank"],
        gh_order=_W["order"], s2s_mode=_W["s2s_mode"])
    shp = (sc.zones.size, len(mag_vals), _W["dists"].size)
    return (branch_flat, m0, m1, sc.k1.reshape(shp).astype(np.float32),
            sc.k2.reshape(shp).astype(np.float32), sc.k3.reshape(shp).astype(np.float32))


# ---- driver --------------------------------------------------------------
def branch_table(gc, model):
    sizes = model.branch_sizes(gc)
    combos = list(itertools.product(*[range(s) for s in sizes]))
    w_median = gc["w_median"].mean("lt_median_choice").values
    w_tau, w_phiss, w_s2s = gc["w_tau"].values, gc["w_phiss"].values, gc["w_s2s"].values
    w_surface = np.full(sizes[4], 1.0 / sizes[4])
    labels = {d: list(gc[d].values) for d in model.BRANCH_DIMS}
    w = np.array([w_median[a] * w_tau[b] * w_phiss[c] * w_s2s[d] * w_surface[e]
                  for (a, b, c, d, e) in combos])
    return combos, w / w.sum(), labels, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--nm", type=int, default=51)
    ap.add_argument("--nr", type=int, default=81)
    ap.add_argument("--mmin", type=float, default=2.5)
    ap.add_argument("--mmax", type=float, default=7.0)
    ap.add_argument("--rmin", type=float, default=3.0)
    ap.add_argument("--rmax", type=float, default=40.0)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--order", type=int, default=7)
    ap.add_argument("--s2s", default="epistemic")
    ap.add_argument("--mag-chunks", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "data" / "im_lookup_cos.zarr"))
    ap.add_argument("--validate", type=int, default=40)
    args = ap.parse_args()

    from saavg_cos import model, spectral_im
    gc = model.load_gmm_config()
    combos, weights, labels, sizes = branch_table(gc, model)
    zones = np.asarray(gc["zone"].values); Z = zones.size
    mags = np.linspace(args.mmin, args.mmax, args.nm)
    dists = np.linspace(args.rmin, args.rmax, args.nr)
    n_branch = len(combos); zsc = n_branch * Z * args.nm * args.nr

    print(f"logic tree: {' x '.join(f'{d}({s})' for d, s in zip(model.BRANCH_DIMS, sizes))}"
          f" = {n_branch} branches")
    print(f"grid: {args.nm} M x {args.nr} R x {Z} zones = {zsc:,} zone-scenarios")
    print(f"engine: COS cumulants rank={args.rank} order={args.order} s2s={args.s2s}")
    print(f"pool: {args.workers} workers, VECLIB={os.environ.get('VECLIB_MAXIMUM_THREADS','default')}\n")

    k1 = np.empty((n_branch, Z, args.nm, args.nr), np.float32)
    k2 = np.empty_like(k1); k3 = np.empty_like(k1)
    idx = [c for c in np.array_split(np.arange(args.nm), args.mag_chunks) if c.size]
    tasks = [(bf, bt, int(c[0]), int(c[-1]) + 1, mags[int(c[0]):int(c[-1]) + 1].tolist())
             for bf, bt in enumerate(combos) for c in idx]

    ctx = __import__("multiprocessing").get_context("spawn")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init,
                             initargs=(args.rank, args.order, args.s2s, dists)) as ex:
        for bf, m0, m1, a1, a2, a3 in ex.map(_task, tasks):
            k1[bf, :, m0:m1, :] = a1; k2[bf, :, m0:m1, :] = a2; k3[bf, :, m0:m1, :] = a3
    compute_s = time.perf_counter() - t0
    print(f"COMPUTE: {compute_s:.2f} s ({zsc / compute_s:,.0f} zone-scenarios/s)")

    import xarray as xr
    combos_arr = np.array(combos)
    coords = dict(branch=np.arange(n_branch), zone=zones, mag=mags, dist=dists,
                  weight=("branch", weights))
    for j, d in enumerate(model.BRANCH_DIMS):
        coords[f"{d}_idx"] = ("branch", combos_arr[:, j])
        coords[f"{d}_label"] = ("branch", np.array([labels[d][i] for i in combos_arr[:, j]]))
    ds = xr.Dataset(dict(k1=(("branch", "zone", "mag", "dist"), k1),
                         k2=(("branch", "zone", "mag", "dist"), k2),
                         k3=(("branch", "zone", "mag", "dist"), k3)),
                    coords=coords,
                    attrs=dict(method="COS cumulants (no histogram)", pca_rank=args.rank,
                               gh_order=args.order, s2s_mode=args.s2s,
                               units_k1="ln(SaAvg) in cm/s2"))
    out = Path(args.out)
    if out.exists():
        import shutil; shutil.rmtree(out)
    ds.to_zarr(out, mode="w", encoding={v: {"chunks": (1, Z, args.nm, args.nr)}
                                        for v in ("k1", "k2", "k3")})
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"WRITE:   -> {out} ({size_mb:.0f} MB)")

    if args.validate:
        rng = np.random.default_rng(0); worst = 0.0
        for _ in range(args.validate):
            bf = int(rng.integers(n_branch)); mi = int(rng.integers(args.nm)); ri = int(rng.integers(args.nr))
            gmm = model.build_branch_gmm(gc, *combos[bf])
            sc = spectral_im.compute_scenarios(gmm, mags[mi:mi + 1], dists[ri:ri + 1],
                                               pca_rank=args.rank, gh_order=args.order, s2s_mode=args.s2s)
            for st, fr in ((k1, sc.k1), (k2, sc.k2), (k3, sc.k3)):
                worst = max(worst, float(np.max(np.abs(st[bf, :, mi, ri] - fr[:, 0]))))
        print(f"VALIDATION: max |stored - direct recompute| = {worst:.2e} "
              f"({'OK' if worst < 1e-4 else 'MISMATCH'})")
    print(f"\nTOTAL compute {compute_s:.1f}s for {zsc:,} zone-scenarios, {n_branch} branches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
