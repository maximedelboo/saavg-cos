"""
End-to-end risk metrics: COS vs Monte-Carlo vs the normal method.

Runs the three methods through ONE identical surrogate chain and compares the
aggregate risk metrics the Groningen SHRA reports: expected buildings reaching
each damage state, expected houses collapsed, and expected annual fatalities.

Real: GMM-V7 IM (central branch) + FCM-V7 fragility for URM1F_B (the most
vulnerable type, which dominates the risk). Surrogate: Gutenberg-Richter source
and 150k buildings ~ zone area, with the source rate calibrated so the riskiest
zone's COS LPR equals 1e-4 (the value the thesis reports for the riskiest
building). The three methods differ ONLY in how P(SaAvg > c) is computed
(MC samples; normal uses k1,k2; COS uses k1,k2,k3 -> captures the skew). The
robust, calibration-independent result is the method comparison (ratios to MC).

    python scripts/compare_risk_metrics.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from saavg_cos import model, spectral_im, cos_cache, gmm_V7, source

LOG_G = np.log(981.0)
DEATH = {"CS1": 0.0, "CS2": 0.0035, "CS3": 0.1}
MEIJDAM = 1e-5
N_BUILDINGS = 150_000
TYPE = "URM1F_B"


def main():
    nM = nR = 15
    n_mc = 10_000
    gmm = model.load_ground_motion_model("central")
    P = gmm.periods.size; Z = gmm.zones.size
    mags = np.linspace(2.5, 7.0, nM); dists = np.linspace(3.0, 40.0, nR)

    zsz = np.asarray(pickle.load(open(ROOT / "data" / "zone_sizes.pkl", "rb")), float)[:Z]
    N_zone = N_BUILDINGS * zsz / zsz.sum()

    df = pd.read_csv(ROOT / "data" / "Fragility_v7_MiddleBranch_20200127.csv")
    row = df[df["structural_system"] == TYPE].iloc[0]
    s_eff = float(row["sigma"]) / float(row["b1"])
    c_ds = (np.log(row[["DL_DS2", "DL_DS3", "DL_DS4"]].to_numpy(float)) - row["b0"]) / row["b1"] + LOG_G
    c_cs = (np.log(row[["DL_CS1", "DL_CS2", "DL_CS3"]].to_numpy(float)) - row["b0"]) / row["b1"] + LOG_G
    thr = np.concatenate([c_ds, c_cs])

    sc = spectral_im.compute_scenarios(gmm, mags, dists, pca_rank=2, gh_order=7, s2s_mode="epistemic")
    k1 = sc.k1.reshape(Z, nM, nR); k2 = sc.k2.reshape(Z, nM, nR); k3 = sc.k3.reshape(Z, nM, nR)
    rate1 = source.rate_grid(mags, dists, total_rate=1.0)
    k2e = (k2 + s_eff ** 2)

    lam = {}
    # normal + COS (vectorised)
    pn = np.empty((Z * nM * nR, thr.size))
    for j, t in enumerate(thr):
        pn[:, j] = 1.0 - ndtr((t - k1.ravel()) / np.sqrt(k2e.ravel()))
    pc = cos_cache.exceedance(thr, k1.ravel(), k2e.ravel(), k3.ravel())
    lam["normal"] = np.einsum("zmrs,mr->zs", pn.reshape(Z, nM, nR, thr.size), rate1)
    lam["COS"] = np.einsum("zmrs,mr->zs", pc.reshape(Z, nM, nR, thr.size), rate1)

    # MC (reference reused across zones)
    rng = np.random.default_rng(7); chol = np.linalg.cholesky(gmm.correlation)
    pmc = np.zeros((Z, nM, nR, thr.size))
    for mi, M in enumerate(mags):
        for ri, R in enumerate(dists):
            rm = gmm_V7.reference_median(R, M, gmm.periods, gmm.median_parameters, par_id=gmm.parameter_median)
            sig = np.sqrt(gmm_V7.reference_ac_variance(R, M, gmm.periods, gmm.tau, gmm.phiss))
            Xref = rm[None, :] + (rng.standard_normal((n_mc, P)) @ chol.T) * sig
            for zi in range(Z):
                afp = gmm.af_parameters[zi]
                af = gmm_V7.af_median(R, M, Xref, afp, par_id=gmm.parameter_af)
                phi = np.sqrt(np.maximum(gmm_V7.af_variance(Xref, afp, par_id=gmm.parameter_af), 0.0))
                sa = np.mean(Xref + af + gmm.s2s_epsilon * phi + gmm.wierde_factor[None, :], axis=1)
                pmc[zi, mi, ri] = np.mean(ndtr((sa[:, None] - thr[None, :]) / s_eff), axis=0)
    lam["MC"] = np.einsum("zmrs,mr->zs", pmc, rate1)

    def lpr(l, T): return DEATH["CS2"] * (l[:, 4] * T) + DEATH["CS3"] * (l[:, 5] * T)
    T = 1e-4 / lpr(lam["COS"], 1.0).max()
    def atleast(l, cols): return [float(np.sum(N_zone * (1 - np.exp(-l[:, s] * T)))) for s in cols]
    def fatal(l): return float(np.sum(N_zone * lpr(l, T)))

    methods = ("MC", "normal", "COS")
    rows = {}
    for s, nm in zip([0, 1, 2], ["DS1 (minor)", "DS2 (moderate)", "DS3 (severe)"]):
        rows[nm] = {m: atleast(lam[m], [s])[0] for m in methods}
    rows["houses collapsed"] = {m: atleast(lam[m], [5])[0] for m in methods}
    rows["E[fatalities]/yr"] = {m: fatal(lam[m]) for m in methods}

    print(f"calibrated total_rate={T:.4g} (riskiest COS LPR = 1e-4); type {TYPE}, {N_BUILDINGS:,} buildings\n")
    print(f"  {'metric':18} {'MC':>9} {'normal':>9} {'COS':>9} | {'normal/MC':>9} {'COS/MC':>8}")
    for nm, v in rows.items():
        print(f"  {nm:18} {v['MC']:9.1f} {v['normal']:9.1f} {v['COS']:9.1f} | "
              f"{v['normal']/v['MC']:9.3f} {v['COS']/v['MC']:8.3f}")
    print(f"\nthesis Meijdam over-count was numerical/MC = {1640/1330:.2f}")

    # figure: over-count ratio vs MC
    labels = list(rows); xpos = np.arange(len(labels))
    rn = [rows[l]["normal"] / rows[l]["MC"] for l in labels]
    rc = [rows[l]["COS"] / rows[l]["MC"] for l in labels]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(xpos - 0.2, rn, 0.4, color="r", alpha=0.8, label="normal / MC")
    ax.bar(xpos + 0.2, rc, 0.4, color="b", alpha=0.8, label="COS / MC")
    ax.axhline(1.0, color="k", lw=1)
    ax.axhline(1640 / 1330, color="r", ls="--", lw=1, alpha=0.6, label="thesis numerical/MC tail bias (1.23)")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("aggregate metric / Monte-Carlo")
    ax.set_ylim(0.9, max(rn) * 1.08)
    ax.set_title("End-to-end risk metrics relative to Monte-Carlo\n"
                 "COS tracks MC (~1.0); the normal method over-counts, more in the deeper tail")
    ax.legend()
    for i, (a, b) in enumerate(zip(rn, rc)):
        ax.text(i - 0.2, a + 0.005, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = ROOT / "figures" / "fig_risk_metrics.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
