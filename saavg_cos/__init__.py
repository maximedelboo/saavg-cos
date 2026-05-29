"""
saavg_cos -- a deterministic COS method for the distribution of average spectral
acceleration (SaAvg) in the Groningen seismic hazard and risk chain.

The package reconstructs the full, non-normal SaAvg distribution from its leading
cumulants by Fourier-cosine (COS) inversion of the cumulant characteristic
function, exploiting the conditional-Gaussian-mixture structure of SaAvg given
the reference ground motion. It is as accurate as Monte-Carlo and as fast as the
normal moment-matching method, and it never histograms intermediate
distributions to disk.

Modules
-------
model          load GMM-V7 parameters (any logic-tree branch) and fragility
gmm_V7         vendored GMM-V7 reference / amplification functions
spectral_im    per-scenario SaAvg cumulants via PCA + Gauss-Hermite quadrature
cos_cache      COS reconstruction of the CDF / exceedance from cumulants
fragility      closed-form fragility (probability-of-damage) expectation
source         transparent Gutenberg-Richter source stand-in
risk_hazard    hazard curves and annual risk
chain          end-to-end driver
mc             Monte-Carlo reference (validation only)
"""

from . import (chain, cos_cache, fragility, gmm_V7, mc, model, risk_hazard,
               source, spectral_im)
from .model import (FCM_PERIODS, build_branch_gmm, load_fragility_model,
                    load_gmm_config, load_ground_motion_model)
from .spectral_im import compute_scenarios

__all__ = [
    "model", "gmm_V7", "spectral_im", "cos_cache", "fragility", "source",
    "risk_hazard", "chain", "mc",
    "load_ground_motion_model", "load_gmm_config", "build_branch_gmm",
    "load_fragility_model", "compute_scenarios", "FCM_PERIODS",
]
__version__ = "1.0.0"
