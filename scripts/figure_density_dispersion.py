"""
Accuracy figures: SaAvg density and PoE dispersion (MC vs normal vs COS).

  figures/fig_density.png     worst-case SaAvg density, the three methods vs MC
  figures/fig_dispersion.png  PoE-ratio-to-MC cloud over many scenarios

Thesis setup: zero-correlation (aleatory) amplification residual, worst-case
reference branch, M=6.5 R=20.93 for the density. (MC-heavy: a few minutes.)

    python scripts/figure_density_dispersion.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from saavg_cos import model, spectral_im, cos_cache, mc

LOG_G = np.log(981.0)
OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)
gmm = model.load_ground_motion_model("upper")
EYE = np.eye(gmm.periods.size)


def figure_density():
    M, R = 6.5, 20.93
    sc = spectral_im.compute_scenarios(gmm, np.array([M]), np.array([R]),
                                       pca_rank=2, gh_order=17, s2s_mode="aleatory", af_correlation=EYE)
    zi = int(np.argmax(np.abs(sc.skewness[:, 0])))
    samples_g = mc.mc_saavg_aleatory(gmm, zi, M, R, n=3_000_000) - LOG_G
    k1g, std = sc.k1[zi, 0] - LOG_G, np.sqrt(sc.k2[zi, 0])
    mc_skew = float(((samples_g - samples_g.mean()) ** 3).mean() / samples_g.std() ** 3)
    x = np.linspace(k1g - 4.5 * std, k1g + 4.5 * std, 600)
    num = np.exp(-0.5 * ((x - k1g) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    cdf = cos_cache.cumulant_cos_cdf(x + LOG_G, sc.k1[zi:zi+1, 0], sc.k2[zi:zi+1, 0], sc.k3[zi:zi+1, 0])[0]
    cos_pdf = np.gradient(cdf, x)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(samples_g, bins=200, density=True, color="0.8", label="Monte Carlo (3M)")
    ax.plot(x, num, "r-", lw=2, label="normal (moment-matching)")
    ax.plot(x, cos_pdf, "b-", lw=2, label="COS (this work)")
    ax.set_xlabel("ln(SaAvg)  [g]"); ax.set_ylabel("probability density")
    ax.set_title(f"Worst-case SaAvg density (M={M}, R={R} km, zero correlation)\n"
                 f"zone {gmm.zones[zi]}, MC skewness = {mc_skew:+.3f}")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_density.png", dpi=140); plt.close(fig)
    print(f"wrote fig_density.png (zone {gmm.zones[zi]}, skew {mc_skew:+.3f})")


def figure_dispersion():
    df = pd.read_csv(ROOT / "data" / "Fragility_v7_MiddleBranch_20200127.csv")
    b0, b1, sf = df["b0"].to_numpy(float), df["b1"].to_numpy(float), df["sigma"].to_numpy(float)
    dl = df[["DL_DS2", "DL_DS3", "DL_DS4", "DL_CS1", "DL_CS2", "DL_CS3"]].to_numpy(float)
    s_eff = sf / b1; c = (np.log(dl) - b0[:, None]) / b1[:, None]
    mags, dists = [4.5, 5.5, 6.5], [3.0, 11.0, 20.93]
    zi_set = np.linspace(0, gmm.zones.size - 1, 20).astype(int)
    poe_mc, poe_num, poe_cos = [], [], []
    for M in mags:
        for R in dists:
            sc = spectral_im.compute_scenarios(gmm, np.array([M]), np.array([R]),
                                               pca_rank=2, gh_order=17, s2s_mode="aleatory", af_correlation=EYE)
            for zi in zi_set:
                samp = mc.mc_saavg_aleatory(gmm, zi, M, R, n=1_000_000, seed=int(zi) + 1) - LOG_G
                k1g, k2 = sc.k1[zi, 0] - LOG_G, sc.k2[zi, 0]
                w, mg, v = sc.weights, sc.means[zi, 0] - LOG_G, sc.variances[zi, 0]
                for si in range(b0.size):
                    poe_mc.extend(np.mean(ndtr((samp[:, None] - c[si][None, :]) / s_eff[si]), axis=0))
                    poe_num.extend(ndtr((k1g - c[si]) / np.sqrt(k2 + s_eff[si] ** 2)))
                    poe_cos.extend(w @ ndtr((mg[:, None] - c[si][None, :]) / np.sqrt(v[:, None] + s_eff[si] ** 2)))
    poe_mc, poe_num, poe_cos = map(np.array, (poe_mc, poe_num, poe_cos))
    keep = poe_mc > 1e-6
    poe_mc, poe_num, poe_cos = poe_mc[keep], poe_num[keep], poe_cos[keep]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(poe_mc, poe_num / poe_mc, s=4, alpha=0.25, color="r", label="normal / MC")
    ax.scatter(poe_mc, poe_cos / poe_mc, s=4, alpha=0.25, color="b", label="COS / MC")
    ax.axhline(1.0, color="k", lw=1); ax.set_xscale("log"); ax.set_ylim(0.5, 3.0)
    ax.set_xlabel("PoE (Monte Carlo)"); ax.set_ylabel("method PoE / MC PoE")
    ax.set_title(f"PoE dispersion vs Monte-Carlo ({poe_mc.size:,} PoEs)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_dispersion.png", dpi=140); plt.close(fig)
    for lo, hi in [(0.1, 1.0), (1e-3, 1e-2), (1e-6, 1e-3)]:
        m = (poe_mc >= lo) & (poe_mc < hi)
        if m.sum():
            print(f"  PoE[{lo:.0e},{hi:.0e}) num |dev|={np.abs(poe_num[m]/poe_mc[m]-1).mean():.1%} "
                  f"COS |dev|={np.abs(poe_cos[m]/poe_mc[m]-1).mean():.2%}")
    print("wrote fig_dispersion.png")


if __name__ == "__main__":
    figure_density()
    figure_dispersion()
