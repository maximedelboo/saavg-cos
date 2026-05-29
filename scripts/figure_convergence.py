"""
Convergence figure: does COS match the Monte-Carlo *truth*?

As the MC sample size N grows, the RMS PoE difference between MC and a *correct*
reference falls like the MC sampling noise (~1/sqrt(N)); the difference between
MC and a *biased* reference plateaus. We drive MC to 1e8 (batched) on the
worst-case scenario and track MC-vs-COS and MC-vs-normal.

  figures/fig_convergence.png

VERY MC-heavy (1e8 samples; ~10-20 min). Reduce N_BATCHES for a quick look.

    python scripts/figure_convergence.py
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
from saavg_cos import model, spectral_im, gmm_V7

LOG_G = np.log(981.0)
OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)
BATCH, N_BATCHES = 2_000_000, 50            # 1e8 samples


def main():
    gmm = model.load_ground_motion_model("upper")
    P = gmm.periods.size; eye = np.eye(P)
    M, R = 6.5, 20.93
    sc = spectral_im.compute_scenarios(gmm, np.array([M]), np.array([R]),
                                       pca_rank=2, gh_order=17, s2s_mode="aleatory", af_correlation=eye)
    zi = int(np.argmax(np.abs(sc.skewness[:, 0])))
    df = pd.read_csv(ROOT / "data" / "Fragility_v7_MiddleBranch_20200127.csv")
    b0, b1, sf = df["b0"].to_numpy(float), df["b1"].to_numpy(float), df["sigma"].to_numpy(float)
    dl = df[["DL_DS2", "DL_DS3", "DL_DS4", "DL_CS1", "DL_CS2", "DL_CS3"]].to_numpy(float)
    s_eff = ((sf / b1)[:, None] * np.ones(6)).reshape(-1)
    c = ((np.log(dl) - b0[:, None]) / b1[:, None]).reshape(-1)

    w, mg, v = sc.weights, sc.means[zi, 0] - LOG_G, sc.variances[zi, 0]
    k1g, k2 = sc.k1[zi, 0] - LOG_G, sc.k2[zi, 0]
    poe_cos = w @ ndtr((mg[:, None] - c[None, :]) / np.sqrt(v[:, None] + s_eff[None, :] ** 2))
    poe_num = ndtr((k1g - c) / np.sqrt(k2 + s_eff ** 2))

    chol = np.linalg.cholesky(gmm.correlation)
    rm = gmm_V7.reference_median(R, M, gmm.periods, gmm.median_parameters, par_id=gmm.parameter_median)
    sig = np.sqrt(gmm_V7.reference_ac_variance(R, M, gmm.periods, gmm.tau, gmm.phiss))
    afp = gmm.af_parameters[zi]; rng = np.random.default_rng(20260529)
    checkpoints = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8]
    sum_phi = np.zeros(c.size); done = 0
    rec = {"N": [], "cos": [], "num": [], "se": []}
    for bb in range(N_BATCHES):
        x = rm[None, :] + (rng.standard_normal((BATCH, P)) @ chol.T) * sig
        af = gmm_V7.af_median(R, M, x, afp, par_id=gmm.parameter_af)
        tau = np.sqrt(np.maximum(gmm_V7.af_variance(x, afp, par_id=gmm.parameter_af), 0.0))
        sa = np.mean(x + af + tau * rng.standard_normal((BATCH, P)) + gmm.wierde_factor[None, :], axis=1) - LOG_G
        sum_phi += ndtr((sa[:, None] - c[None, :]) / s_eff[None, :]).sum(0); done += BATCH
        if any(abs(done - cp) < BATCH for cp in checkpoints) or bb == N_BATCHES - 1:
            poe_mc = sum_phi / done; big = poe_cos > 1e-4
            rec["N"].append(done)
            rec["cos"].append(np.sqrt(np.mean((poe_mc[big] - poe_cos[big]) ** 2)))
            rec["num"].append(np.sqrt(np.mean((poe_mc[big] - poe_num[big]) ** 2)))
            rec["se"].append(np.sqrt(np.mean(poe_mc[big] * (1 - poe_mc[big]) / done)))
    N = np.array(rec["N"])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.loglog(N, rec["cos"], "bo-", label="MC vs COS")
    ax.loglog(N, rec["num"], "rs-", label="MC vs normal")
    ax.loglog(N, rec["se"], "k--", lw=1, label="MC sampling noise (~1/sqrt(N))")
    ax.set_xlabel("Monte-Carlo sample size N"); ax.set_ylabel("RMS PoE difference")
    ax.set_title("Convergence to the Monte-Carlo truth (worst-case scenario)\n"
                 "MC->COS falls with noise; MC->normal plateaus at the skew bias")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_convergence.png", dpi=140); plt.close(fig)
    print(f"COS RMS @1e8 = {rec['cos'][-1]:.2e}; normal RMS @1e8 = {rec['num'][-1]:.2e}")
    print("wrote fig_convergence.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
