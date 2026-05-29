"""
Fast correctness tests for the saavg_cos package (run with: pytest -q).

These are lightweight smoke + accuracy checks; the heavy MC validation lives in
scripts/validate_against_mc.py.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from saavg_cos import model, spectral_im, cos_cache, mc


def test_branch_loading():
    gc = model.load_gmm_config()
    sizes = model.branch_sizes(gc)
    assert sizes == (4, 3, 2, 3, 2)                     # GMM-V7 logic tree
    gmm = model.build_branch_gmm(gc, 0, 0, 0, 0, 0)
    assert gmm.periods.size == 10 and gmm.zones.size == 160


def test_cumulants_finite_and_shaped():
    gmm = model.load_ground_motion_model("central")
    sc = spectral_im.compute_scenarios(gmm, np.array([5.0, 6.0]), np.array([5.0, 20.0]),
                                       pca_rank=2, gh_order=7)
    assert sc.k1.shape == (160, 4) and sc.k2.shape == (160, 4)
    assert np.all(np.isfinite(sc.k1)) and np.all(sc.k2 > 0)
    assert np.all(sc.skewness <= 0.05)                  # SaAvg is negatively skewed / mild


def test_cos_cdf_valid():
    k1, k2, k3 = np.array([5.0]), np.array([0.25]), np.array([-0.03])
    x = np.linspace(3.0, 7.0, 200)
    cdf = cos_cache.cumulant_cos_cdf(x, k1, k2, k3)[0]
    cdf = np.maximum.accumulate(np.clip(cdf, 0, 1))     # production cleanup
    assert cdf[0] < 1e-3 and cdf[-1] > 1 - 1e-3 and np.all(np.diff(cdf) >= -1e-9)


def test_cos_matches_mc_cumulants():
    gmm = model.load_ground_motion_model("upper")
    M, R = 6.0, 5.0
    sc = spectral_im.compute_scenarios(gmm, np.array([M]), np.array([R]),
                                       pca_rank=2, gh_order=17, s2s_mode="epistemic")
    zi = int(np.argmax(np.abs(sc.skewness[:, 0])))
    x = mc.mc_saavg_epistemic(gmm, zi, M, R, n=2_000_000, seed=1)
    assert abs(sc.k1[zi, 0] - x.mean()) < 5e-3
    assert abs(sc.k2[zi, 0] - x.var()) < 5e-3


def test_cos_more_accurate_than_normal_in_tail():
    """At a skewed cell, COS PoE beats the normal PoE against MC in the tail."""
    from scipy.special import ndtr
    gmm = model.load_ground_motion_model("upper")
    M, R = 6.5, 4.0
    sc = spectral_im.compute_scenarios(gmm, np.array([M]), np.array([R]),
                                       pca_rank=2, gh_order=17, s2s_mode="epistemic")
    zi = int(np.argmax(np.abs(sc.skewness[:, 0])))
    x = mc.mc_saavg_epistemic(gmm, zi, M, R, n=5_000_000, seed=2)
    k1, k2, k3 = sc.k1[zi, 0], sc.k2[zi, 0], sc.k3[zi, 0]
    t = k1 + 1.6 * np.sqrt(k2)                           # upper tail threshold
    mc_poe = (x > t).mean()
    cos_poe = cos_cache.exceedance(np.array([t]), np.array([k1]), np.array([k2]), np.array([k3]))[0, 0]
    norm_poe = 1.0 - ndtr((t - k1) / np.sqrt(k2))
    assert abs(cos_poe - mc_poe) < abs(norm_poe - mc_poe)
