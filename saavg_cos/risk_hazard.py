"""
Hazard and risk integration.

Both quantities are LINEAR functionals of each scenario's SaAvg distribution,
so they are obtained by rate-weighting per-scenario spectral functionals -- no
histogram is ever formed:

    hazard:  lambda(SaAvg > x) = sum_{M,R} rate(M,R) * P(SaAvg > x | M,R)
    risk:    annual P(dying)    = sum_{M,R} rate(M,R) * E[POD | M,R]

``P(SaAvg > x | M,R)`` comes from the cached/COS exceedance of the cumulants;
``E[POD | M,R]`` is the closed-form fragility expectation.
"""

from dataclasses import dataclass

import numpy as np

from . import cos_cache
from .spectral_im import Scenarios
from .fragility import RiskResult

LOG_G_TO_CM = np.log(981.0)


@dataclass
class HazardResult:
    zones: np.ndarray
    im_levels_g: np.ndarray        # (L,) SaAvg levels in g
    rate_exceedance: np.ndarray    # (Z, L) annual rate lambda(SaAvg > x)


def hazard_curves(scenarios: Scenarios, rate_vector, im_levels_g,
                  cache: cos_cache.StandardizedCache = None) -> HazardResult:
    """Annual exceedance-rate curves per zone."""
    points = np.log(np.asarray(im_levels_g, float)) + LOG_G_TO_CM   # ln cm/s2
    Z = scenarios.zones.size
    out = np.empty((Z, points.size))
    for z in range(Z):
        if cache is None:
            exc = cos_cache.exceedance(points, scenarios.k1[z], scenarios.k2[z], scenarios.k3[z])
        else:
            exc = cache.exceedance(points, scenarios.k1[z], scenarios.k2[z], scenarios.k3[z])
        out[z] = rate_vector @ exc                     # (L,)
    return HazardResult(zones=scenarios.zones, im_levels_g=np.asarray(im_levels_g, float),
                        rate_exceedance=out)


def hazard_at_return_periods(hazard: HazardResult, return_periods):
    """Interpolate the SaAvg (g) at given return periods, per zone."""
    return_periods = np.asarray(return_periods, float)
    target_rate = 1.0 / return_periods
    ln_im = np.log(hazard.im_levels_g)
    out = np.empty((hazard.zones.size, return_periods.size))
    for z in range(hazard.zones.size):
        rate = hazard.rate_exceedance[z]
        # exceedance rate is decreasing in IM -> interpolate on log-rate vs ln(IM)
        order = np.argsort(rate)
        out[z] = np.exp(np.interp(np.log(target_rate), np.log(rate[order]), ln_im[order]))
    return out                                          # (Z, n_return_periods) in g


@dataclass
class AnnualRisk:
    zones: np.ndarray
    systems: np.ndarray
    annual_pod: np.ndarray         # (Z, S) annual probability of dying


def annual_risk(risk: RiskResult, rate_vector) -> AnnualRisk:
    """Rate-weighted annual probability of dying per zone per structural system."""
    annual = np.einsum("m,zms->zs", rate_vector, risk.expected_pod)
    return AnnualRisk(zones=risk.zones, systems=risk.systems, annual_pod=annual)
