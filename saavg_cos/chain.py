"""
End-to-end driver for the spectral hazard & risk chain.

    model parameters (GMM-V7 + fragility)
        -> spectral IM stage      : per-scenario cumulants + Gaussian mixture
        -> source model           : Gutenberg-Richter rate(M, R)
        -> hazard integration     : lambda(SaAvg > x) per zone   (COS exceedance)
        -> risk integration       : annual P(dying) per zone     (closed-form)

Nothing is ever histogrammed; the IM distribution is carried as cumulants and
the downstream integrals are spectral / analytic.
"""

import time
from dataclasses import dataclass

import numpy as np

from . import model, spectral_im, source, fragility, risk_hazard, cos_cache


@dataclass
class ChainResult:
    scenarios: spectral_im.Scenarios
    hazard: risk_hazard.HazardResult
    risk: risk_hazard.AnnualRisk
    return_period_im: np.ndarray   # (Z, n_rp) SaAvg(g) at return periods
    return_periods: np.ndarray
    timings: dict


def run(branch="central", n_mag=51, n_dist=81,
        im_levels_g=None, return_periods=(475.0, 2475.0),
        pca_rank=2, gh_order=7, use_cache=False, verbose=True):
    if im_levels_g is None:
        im_levels_g = np.geomspace(1e-3, 3.0, 60)
    magnitudes = np.linspace(1.5, 6.5, n_mag)
    distances = np.geomspace(3.0, 40.0, n_dist)
    timings = {}

    t = time.perf_counter()
    gmm = model.load_ground_motion_model(branch=branch)
    frag = model.load_fragility_model(branch=branch)
    timings["load"] = time.perf_counter() - t

    t = time.perf_counter()
    scen = spectral_im.compute_scenarios(gmm, magnitudes, distances,
                                         pca_rank=pca_rank, gh_order=gh_order)
    timings["spectral_im"] = time.perf_counter() - t

    cache = cos_cache.build_standardized_cache() if use_cache else None

    t = time.perf_counter()
    rates = source.rate_vector(magnitudes, distances)
    haz = risk_hazard.hazard_curves(scen, rates, im_levels_g, cache=cache)
    rp_im = risk_hazard.hazard_at_return_periods(haz, return_periods)
    timings["hazard"] = time.perf_counter() - t

    t = time.perf_counter()
    risk_res = fragility.expected_pod(scen, frag)
    annual = risk_hazard.annual_risk(risk_res, rates)
    timings["risk"] = time.perf_counter() - t

    if verbose:
        n_cells = scen.zones.size * magnitudes.size * distances.size
        print(f"spectral chain: {scen.zones.size} zones x {n_mag} M x {n_dist} R "
              f"= {n_cells:,} scenarios")
        for k, val in timings.items():
            print(f"  {k:12}: {val:6.2f} s")
        print(f"  TOTAL       : {sum(timings.values()):6.2f} s")
        rp = np.asarray(return_periods)
        print(f"\n  SaAvg (g) at return periods {rp.tolist()} (median over zones):")
        for j, T in enumerate(rp):
            print(f"    {int(T)} yr: {np.median(rp_im[:, j]):.4f} g  "
                  f"[{rp_im[:, j].min():.4f}, {rp_im[:, j].max():.4f}]")
        # aggregate annual risk: max death prob across systems per zone (worst typology)
        worst = annual.annual_pod.max(axis=1)
        print(f"\n  annual P(dying), worst structural system per zone:")
        print(f"    median {np.median(worst):.2e}, range [{worst.min():.2e}, {worst.max():.2e}]")

    return ChainResult(scenarios=scen, hazard=haz, risk=annual,
                       return_period_im=rp_im, return_periods=np.asarray(return_periods),
                       timings=timings)


if __name__ == "__main__":
    run()
