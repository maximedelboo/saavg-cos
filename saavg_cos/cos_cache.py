"""
Cached COS reconstruction.

The marginal SaAvg distribution is reconstructed from its leading cumulants
``(k1, k2, k3)`` by Fourier-cosine (COS) inversion of the cumulant
characteristic function ``exp(i k1 u - 1/2 k2 u^2 - i k3 u^3 / 6)``, whose
magnitude stays Gaussian-damped so the series converges and the cubic phase
injects the skewness.

After standardising ``z = (x - k1)/sqrt(k2)`` the shape depends only on the
skewness ``g1 = k3 / k2^1.5``. We therefore precompute the standardised CDF on
a 2-D table over ``(z, g1)`` ONCE; reconstructing the CDF at any thresholds for
any scenario is then a bilinear table lookup -- no per-scenario inversion, no
histogram. The direct (un-cached) inversion is also provided and is used to
build (and validate) the table.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr as _normal_cdf   # fast vectorised standard-normal CDF

SQRT_2PI = np.sqrt(2.0 * np.pi)


def _normal_pdf(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / SQRT_2PI


def cumulant_cos_cdf(points, k1, k2, k3, truncation_l=10.0, cos_terms=64):
    """Direct COS CDF from cumulants.

    ``points`` are the (shared) evaluation abscissae, shape (E,); ``k1,k2,k3``
    are per-scenario cumulants, shape (C,). Returns the CDF, shape (C, E).
    """
    points = np.atleast_1d(np.asarray(points, float))
    k1 = np.atleast_1d(np.asarray(k1, float))
    k2 = np.atleast_1d(np.asarray(k2, float))
    k3 = np.atleast_1d(np.asarray(k3, float))
    std = np.sqrt(k2)
    a = k1 - truncation_l * std
    b = k1 + truncation_l * std
    width = (b - a)[:, None]
    k = np.arange(cos_terms, dtype=float)
    u = k[None, :] * np.pi / width
    phi = np.exp(-0.5 * k2[:, None] * u ** 2) * np.exp(1j * (k1[:, None] * u - k3[:, None] * u ** 3 / 6.0))
    coeff = (2.0 / width) * np.real(phi * np.exp(-1j * u * a[:, None]))
    theta = np.clip((points[None, :] - a[:, None]) / width, 0.0, 1.0)
    cdf = 0.5 * coeff[:, :1] * (np.clip(points[None, :], a[:, None], b[:, None]) - a[:, None])
    if cos_terms > 1:
        sine = np.sin(k[1:][None, None, :] * np.pi * theta[:, :, None])
        factor = coeff[:, 1:] * width / (k[1:][None, :] * np.pi)
        cdf = cdf + np.einsum("ck,cek->ce", factor, sine)
    cdf = np.where(points[None, :] <= a[:, None], 0.0, cdf)
    cdf = np.where(points[None, :] >= b[:, None], 1.0, cdf)
    return np.clip(cdf, 0.0, 1.0)


def exceedance(points, k1, k2, k3, **kw):
    """P(IM > x) at thresholds ``points`` for scenarios with cumulants k."""
    return 1.0 - cumulant_cos_cdf(points, k1, k2, k3, **kw)


@dataclass
class StandardizedCache:
    gamma_grid: np.ndarray         # (G,) skewness nodes
    z_grid: np.ndarray             # (Zn,) standardized abscissae
    cdf_table: np.ndarray          # (G, Zn) standardized CDF

    def cdf(self, points, k1, k2, k3):
        """CDF at ``points`` (E,) for scenarios with cumulants (C,) via lookup."""
        k1 = np.atleast_1d(np.asarray(k1, float))
        k2 = np.atleast_1d(np.asarray(k2, float))
        k3 = np.atleast_1d(np.asarray(k3, float))
        std = np.sqrt(k2)
        g1 = np.clip(k3 / std ** 3, self.gamma_grid[0], self.gamma_grid[-1])
        z = (np.atleast_1d(points)[None, :] - k1[:, None]) / std[:, None]   # (C, E)
        z = np.clip(z, self.z_grid[0], self.z_grid[-1])

        # bracket in skewness and interpolate two table rows, then interp in z
        gi = np.searchsorted(self.gamma_grid, g1).clip(1, self.gamma_grid.size - 1)
        g_lo, g_hi = self.gamma_grid[gi - 1], self.gamma_grid[gi]
        wg = ((g1 - g_lo) / (g_hi - g_lo))[:, None]
        out = np.empty_like(z)
        for c in range(z.shape[0]):
            lo = np.interp(z[c], self.z_grid, self.cdf_table[gi[c] - 1])
            hi = np.interp(z[c], self.z_grid, self.cdf_table[gi[c]])
            out[c] = (1 - wg[c]) * lo + wg[c] * hi
        return np.clip(out, 0.0, 1.0)

    def exceedance(self, points, k1, k2, k3):
        return 1.0 - self.cdf(points, k1, k2, k3)


def build_standardized_cache(gamma_range=(-0.4, 0.1), n_gamma=81,
                             z_max=8.0, n_z=2001, cos_terms=96):
    """Precompute the standardized CDF table over (z, skewness) once."""
    gamma_grid = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
    z_grid = np.linspace(-z_max, z_max, n_z)
    # standardized cumulants: mean 0, variance 1, third cumulant = g1
    cdf_table = cumulant_cos_cdf(
        z_grid, np.zeros(n_gamma), np.ones(n_gamma), gamma_grid,
        truncation_l=z_max, cos_terms=cos_terms,
    )
    return StandardizedCache(gamma_grid=gamma_grid, z_grid=z_grid, cdf_table=cdf_table)
