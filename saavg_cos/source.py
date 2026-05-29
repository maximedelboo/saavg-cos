"""
Seismic source model: annual rate of an earthquake of magnitude M at rupture
distance R.

A simple, transparent, swappable component: a doubly-truncated Gutenberg-Richter
magnitude-frequency distribution combined with a distance distribution for
events uniformly distributed on a disk of radius Rmax around the site. This is
NOT the official Groningen seismicity forecast -- it is a stand-in so the rest
of the spectral chain can be exercised end-to-end; replace ``rate_grid`` with a
real source distribution when available.
"""

import numpy as np


def rate_grid(magnitudes, distances, total_rate=0.3, b_value=0.9,
              m_min=1.5, m_max=6.5, r_max=40.0):
    """Annual rate of an event in each (M, R) cell, shape (nM, nR), row-major.

    ``total_rate`` is the annual number of events with M >= m_min.
    """
    magnitudes = np.asarray(magnitudes, float)
    distances = np.asarray(distances, float)

    # truncated Gutenberg-Richter magnitude pdf (per unit magnitude)
    beta = b_value * np.log(10.0)
    pdf_m = beta * np.exp(-beta * (magnitudes - m_min))
    pdf_m = np.where((magnitudes >= m_min) & (magnitudes <= m_max), pdf_m, 0.0)
    pdf_m /= 1.0 - np.exp(-beta * (m_max - m_min))      # truncation renormalisation
    # magnitude cell widths (trapezoidal weights)
    dM = np.gradient(magnitudes)
    rate_m = total_rate * pdf_m * dM                    # (nM,)

    # distance pdf for events uniform on a disk of radius r_max: f(R) = 2R/r_max^2
    pdf_r = np.where(distances <= r_max, 2.0 * distances / r_max ** 2, 0.0)
    dR = np.gradient(distances)
    p_r = pdf_r * dR
    p_r = p_r / p_r.sum() if p_r.sum() > 0 else p_r     # normalise over the grid

    return rate_m[:, None] * p_r[None, :]               # (nM, nR)


def rate_vector(magnitudes, distances, **kw):
    """Flattened rate vector matching the (M outer, R inner) scenario order."""
    return rate_grid(magnitudes, distances, **kw).reshape(-1)
