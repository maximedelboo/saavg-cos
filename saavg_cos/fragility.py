"""
Analytic fragility / consequence expectations.

The structural collapse fragility is a lognormal CDF in SaAvg,
``POE_ls(lnSaAvg) = Phi((lnSaAvg - c_ls)/s)`` with ``c_ls = (ln(DL_ls)-b0)/b1``
and ``s = sigma/b1``. The expected probability of dying is a death-weighted sum
of exceedance probabilities, so against the Gaussian-mixture SaAvg distribution
every expectation is **closed form**:

    E[POE_ls] = sum_n w_n * Phi( (m_n - c_ls) / sqrt(v_n + s^2) )
    E[POD]    = sum_ls (death-weight_ls) * E[POE_ls]

No intensity-measure grid, no histogram, no quadrature -- one error function.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr as _normal_cdf   # fast vectorised standard-normal CDF

from .model import FragilityModel
from .spectral_im import Scenarios

LOG_G_TO_CM = np.log(981.0)   # the fragility uses SaAvg in g; the mixture is in cm/s2


@dataclass
class RiskResult:
    zones: np.ndarray
    magnitudes: np.ndarray
    distances: np.ndarray
    systems: np.ndarray
    expected_pod: np.ndarray       # (Z, MR, S) expected probability of dying


def expected_pod(scenarios: Scenarios, fragility: FragilityModel,
                 mr_chunk=4000) -> RiskResult:
    """Closed-form expected probability of dying per (zone, M-R, system)."""
    w = scenarios.weights                              # (N,)
    m = scenarios.means                                # (Z, MR, N)
    v = scenarios.variances
    Z, MR, _ = m.shape
    S = fragility.systems.size

    b1 = fragility.b1
    s = fragility.sigma / b1                           # (S,) effective dispersion
    # collapse-state median capacities in lnSaAvg: c = (ln(DL) - b0)/b1
    c = (np.log(fragility.collapse_limits) - fragility.b0[:, None]) / b1[:, None]  # (S, n_cs)
    # death weights: pod_cond_poe_ls = pdeath_ls - pdeath_{ls-1}
    pdeath = fragility.death_probability
    death_weight = pdeath - np.concatenate([[0.0], pdeath[:-1]])   # (n_cs,)

    m_g = m - LOG_G_TO_CM                              # convert mixture means cm/s2 -> g
    out = np.empty((Z, MR, S))
    for si in range(S):
        c_s = c[si]                                    # (n_cs,)
        denom = np.sqrt(v[:, :, None, :] + s[si] ** 2)  # (Z, MR, n_cs, N)
        arg = (m_g[:, :, None, :] - c_s[None, None, :, None]) / denom
        e_poe = _normal_cdf(arg) @ w                    # (Z, MR, n_cs)
        out[:, :, si] = e_poe @ death_weight
    return RiskResult(zones=scenarios.zones, magnitudes=scenarios.magnitudes,
                      distances=scenarios.distances, systems=fragility.systems,
                      expected_pod=out)
