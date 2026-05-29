"""
Validate the COS method against Monte-Carlo truth.

Two checks, for the worst-case (most negatively skewed) scenario and a couple of
benign ones:
  (1) cumulants: COS k1/k2/skew vs deep-MC moments;
  (2) probability of exceedance: COS vs MC vs the normal method across a fragility
      threshold sweep, reporting the max |error| of each method against MC.

The COS error rides the MC sampling noise; the normal method carries a fixed
skew bias. Epistemic s2s mode (GMM-V7 production).

    python scripts/validate_against_mc.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from saavg_cos import model, spectral_im, cos_cache, mc

LOG_G = np.log(981.0)


def main():
    gmm = model.load_ground_motion_model("upper")          # worst-case reference branch
    M, R = 6.5, 20.93
    sc = spectral_im.compute_scenarios(gmm, np.array([M]), np.array([R]),
                                       pca_rank=2, gh_order=17, s2s_mode="epistemic")
    zi = int(np.argmax(np.abs(sc.skewness[:, 0])))
    print(f"worst-case scenario: zone {gmm.zones[zi]}, M={M}, R={R}, "
          f"COS skew={sc.skewness[zi, 0]:+.3f}\n")

    # (1) cumulants vs deep MC -------------------------------------------------
    cells = [("most-skewed", zi, M, R),
             ("mid M5.5/R10", 0, 5.5, 10.0),
             ("benign M3/R30", gmm.zones.size - 1, 3.0, 30.0)]
    print("(1) cumulants: COS vs Monte-Carlo (5M samples)")
    print(f"  {'cell':>14} {'k1 COS':>9} {'k1 MC':>9} {'k2 COS':>8} {'k2 MC':>8} {'skew COS':>9} {'skew MC':>9}")
    for name, z, m, r in cells:
        s = spectral_im.compute_scenarios(gmm, np.array([m]), np.array([r]),
                                          pca_rank=2, gh_order=17, s2s_mode="epistemic")
        x = mc.mc_saavg_epistemic(gmm, z, m, r, n=5_000_000, seed=z + 1)
        mu, var = x.mean(), x.var()
        sk = ((x - mu) ** 3).mean() / var ** 1.5
        print(f"  {name:>14} {s.k1[z,0]:>9.4f} {mu:>9.4f} {s.k2[z,0]:>8.4f} {var:>8.4f} "
              f"{s.skewness[z,0]:>9.3f} {sk:>9.3f}")

    # (2) PoE accuracy vs MC at the worst-case cell ----------------------------
    x = mc.mc_saavg_epistemic(gmm, zi, M, R, n=20_000_000, seed=42)
    k1, k2, k3 = sc.k1[zi, 0], sc.k2[zi, 0], sc.k3[zi, 0]
    std = np.sqrt(k2)
    thr = np.linspace(k1 - 2.4 * std, k1 + 2.4 * std, 60)          # q01..q99-ish band

    # (2a) RAW exceedance of the bare distribution -- the harshest test; this is
    # where the 3-cumulant truncation residual shows (reducible by adding k4).
    poe_mc = np.array([(x > t).mean() for t in thr])
    poe_cos = cos_cache.exceedance(thr, np.array([k1]), np.array([k2]), np.array([k3]))[0]
    poe_norm = 1.0 - ndtr((thr - k1) / std)
    keep = (poe_mc > 1e-4) & (poe_mc < 1 - 1e-4)
    err_cos = np.max(np.abs(poe_cos[keep] - poe_mc[keep]))
    err_norm = np.max(np.abs(poe_norm[keep] - poe_mc[keep]))
    print(f"\n(2a) RAW exceedance P(SaAvg>x) vs MC (20M), worst cell, q-band:")
    print(f"  max |COS    - MC| = {err_cos:.2e}   (3-cumulant truncation residual)")
    print(f"  max |normal - MC| = {err_norm:.2e}   ({err_norm / err_cos:.1f}x larger than COS)")

    # (2b) FRAGILITY-CONVOLVED PoE -- the risk-relevant quantity. The demand
    # dispersion adds s^2 to the variance, smoothing the tail; this is what the
    # SHRA integrates. Representative s_eff from URM1F_B (sigma/b1).
    s_eff = 0.9506 / 2.8145
    poe_mc_c = np.array([ndtr((x - t) / s_eff).mean() for t in thr])
    poe_cos_c = cos_cache.exceedance(thr, np.array([k1]), np.array([k2 + s_eff ** 2]), np.array([k3]))[0]
    poe_norm_c = 1.0 - ndtr((thr - k1) / np.sqrt(k2 + s_eff ** 2))
    keep_c = (poe_mc_c > 1e-4) & (poe_mc_c < 1 - 1e-4)
    ec = np.max(np.abs(poe_cos_c[keep_c] - poe_mc_c[keep_c]))
    en = np.max(np.abs(poe_norm_c[keep_c] - poe_mc_c[keep_c]))
    print(f"\n(2b) FRAGILITY-CONVOLVED PoE vs MC (the risk-relevant quantity):")
    print(f"  max |COS    - MC| = {ec:.2e}   (small deterministic 3-cumulant residual)")
    print(f"  max |normal - MC| = {en:.2e}   ({en / ec:.1f}x larger than COS)")
    print(f"\nNote: this is the single most-skewed cell with the *max* over a wide band -- the")
    print(f"harshest case for COS. Over a realistic scenario cloud the COS deviation is sub-%")
    print(f"and the gap to the normal method is ~20-30x in the deep tail (see fig_dispersion,")
    print(f"compare_risk_metrics). The normal error is a FIXED negative-skew bias; the COS")
    print(f"residual shrinks with Gauss-Hermite order and added cumulants (k4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
