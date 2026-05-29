"""
Monte-Carlo reference for SaAvg.

These samplers draw the surface average spectral acceleration directly from the
GMM-V7 model and are used only to *validate* the COS method -- they are the slow,
exact ground truth the deterministic method is checked against. Two site-to-site
(s2s) treatments are provided, matching ``spectral_im.compute_scenarios``:

* ``mc_saavg_epistemic`` -- s2s is a deterministic shift (GMM-V7 production mode);
  the only randomness is the correlated reference vector.
* ``mc_saavg_aleatory`` -- the amplification residual is random; pass the period
  correlation of that residual (identity = the "zero correlation" mode).

Both return ln(SaAvg) in cm/s^2.
"""

import numpy as np

from . import gmm_V7


def _reference_draw(gmm, M, R, n, rng):
    """n correlated reference log-Sa vectors, shape (n, P), in cm/s^2."""
    P = gmm.periods.size
    chol = np.linalg.cholesky(gmm.correlation)
    rm = gmm_V7.reference_median(R, M, gmm.periods, gmm.median_parameters,
                                 par_id=gmm.parameter_median)
    sig = np.sqrt(gmm_V7.reference_ac_variance(R, M, gmm.periods, gmm.tau, gmm.phiss))
    return rm[None, :] + (rng.standard_normal((n, P)) @ chol.T) * sig


def mc_saavg_epistemic(gmm, zi, M, R, n=1_000_000, seed=7):
    """MC ln(SaAvg) (cm/s^2), epistemic s2s: deterministic shift, no AF residual."""
    rng = np.random.default_rng(seed)
    x = _reference_draw(gmm, M, R, n, rng)
    afp = gmm.af_parameters[zi]
    af = gmm_V7.af_median(R, M, x, afp, par_id=gmm.parameter_af)
    phi = np.sqrt(np.maximum(gmm_V7.af_variance(x, afp, par_id=gmm.parameter_af), 0.0))
    surface = x + af + gmm.s2s_epsilon * phi + gmm.wierde_factor[None, :]
    return np.mean(surface, axis=1)


def mc_saavg_aleatory(gmm, zi, M, R, n=1_000_000, seed=7, af_correlation=None):
    """MC ln(SaAvg) (cm/s^2), aleatory s2s: random AF residual.

    ``af_correlation`` is the period-to-period correlation of the amplification
    residual; ``None`` -> identity (the zero-correlation mode).
    """
    rng = np.random.default_rng(seed)
    P = gmm.periods.size
    x = _reference_draw(gmm, M, R, n, rng)
    afp = gmm.af_parameters[zi]
    af = gmm_V7.af_median(R, M, x, afp, par_id=gmm.parameter_af)
    tau = np.sqrt(np.maximum(gmm_V7.af_variance(x, afp, par_id=gmm.parameter_af), 0.0))
    if af_correlation is None:
        eps = rng.standard_normal((n, P))
    else:
        chol = np.linalg.cholesky(af_correlation)
        eps = rng.standard_normal((n, P)) @ chol.T
    surface = x + af + tau * eps + gmm.wierde_factor[None, :]
    return np.mean(surface, axis=1)
