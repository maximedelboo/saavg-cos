"""
Spectral intensity-measure stage.

For every (zone, magnitude, rupture-distance) scenario, the surface SaAvg
distribution is represented as a finite Gaussian mixture obtained by
integrating the correlated reference ground motion through the non-linear
amplification with PCA + Gauss-Hermite quadrature. The reference quadrature
depends only on (M, R), so its eigendecomposition is computed ONCE and shared
across all zones; only the amplification is evaluated per zone, with
``exp(reference nodes)`` hoisted out of the zone loop.

The stage outputs, per scenario, the compact mixture ``(weights, means,
variances)`` and the leading cumulants ``(k1, k2, k3)`` -- never a histogram.
"""

from dataclasses import dataclass

import numpy as np

from . import gmm_V7
from .model import GroundMotionModel


@dataclass
class Scenarios:
    zones: np.ndarray              # (Z,)
    magnitudes: np.ndarray         # (nM,)
    distances: np.ndarray          # (nR,)
    weights: np.ndarray            # (N,)  mixture weights (shared)
    means: np.ndarray              # (Z, MR, N) component means of lnSaAvg
    variances: np.ndarray          # (Z, MR, N) component variances
    k1: np.ndarray                 # (Z, MR) mean
    k2: np.ndarray                 # (Z, MR) variance
    k3: np.ndarray                 # (Z, MR) third cumulant

    @property
    def skewness(self):
        return self.k3 / np.maximum(self.k2, 1e-300) ** 1.5


def _gauss_hermite(rank, order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes = np.sqrt(2.0) * nodes
    weights = weights / np.sqrt(np.pi)
    if rank == 1:
        return nodes[:, None], weights
    mesh = np.meshgrid(*([nodes] * rank), indexing="ij")
    grid = np.stack([m.reshape(-1) for m in mesh], axis=1)
    wmesh = np.meshgrid(*([weights] * rank), indexing="ij")
    w = np.prod(np.stack(wmesh, axis=0), axis=0).reshape(-1)
    return grid, w


def compute_scenarios(gmm: GroundMotionModel, magnitudes, distances,
                      pca_rank=2, gh_order=7, zone_chunk=16,
                      s2s_mode="epistemic", af_correlation=None) -> Scenarios:
    """Per-scenario SaAvg cumulants + mixture.

    ``s2s_mode``: "epistemic" (V7 default: site-to-site as a deterministic shift,
    no aleatory amplification residual) or "aleatory" (amplification residual is
    random with period correlation ``af_correlation`` -- pass an identity matrix
    for the "zero correlation mode"; defaults to the reference correlation).
    """
    periods = gmm.periods
    P = periods.size
    par_af = gmm.parameter_af
    cor = gmm.correlation
    if af_correlation is None:
        af_correlation = np.eye(P) if s2s_mode == "aleatory" else cor
    nM, nR = magnitudes.size, distances.size
    n_mr = nM * nR
    M_mr = np.repeat(magnitudes, nR)
    R_mr = np.tile(distances, nM)

    # reference moments over the (M, R) grid, shared across zones
    R = distances[None, :, None]
    M = magnitudes[:, None, None]
    T = periods[None, None, :]
    ref_mean = gmm_V7.reference_median(R, M, T, gmm.median_parameters[None, None],
                                       par_id=gmm.parameter_median).reshape(n_mr, P)
    sig = np.sqrt(gmm_V7.reference_ac_variance(R, M, T, gmm.tau, gmm.phiss)).reshape(n_mr, P)
    ref_cov = cor[None] * sig[:, :, None] * sig[:, None, :]

    eigenvalues, eigenvectors = np.linalg.eigh(ref_cov)
    eigenvalues = np.clip(eigenvalues[:, ::-1], 0.0, None)
    eigenvectors = eigenvectors[:, :, ::-1]
    rank = min(pca_rank, P)
    unit_nodes, weights = _gauss_hermite(rank, gh_order)
    n_nodes = weights.size
    scaled = unit_nodes[None, :, :] * np.sqrt(eigenvalues[:, None, :rank])
    nodes = ref_mean[:, None, :] + np.einsum("cnr,cpr->cnp", scaled, eigenvectors[:, :, :rank])
    residual_cov = np.einsum("cpr,cr,cqr->cpq", eigenvectors[:, :, rank:],
                             eigenvalues[:, rank:], eigenvectors[:, :, rank:])
    exp_nodes = np.exp(nodes)                       # shared across zones
    s2s = gmm.s2s_epsilon
    wierde = gmm.wierde_factor

    Z = gmm.zones.size
    means = np.empty((Z, n_mr, n_nodes))
    variances = np.empty((Z, n_mr, n_nodes))

    for z0 in range(0, Z, zone_chunk):
        z1 = min(z0 + zone_chunk, Z)
        c = gmm_V7.gen_dict_like(gmm.af_parameters[z0:z1], par_af)

        def cz(name):
            return c[name][:, None, None, :]    # (Zc,1,1,P)

        lnR = np.log(R_mr)[None, :, None]
        Mb = M_mr[None, :, None]
        Ma_, Mb_ = cz("Ma")[:, :, 0, :], cz("Mb")[:, :, 0, :]
        Mref1 = np.clip(Ma_ + (Mb_ - Ma_) * (lnR - np.log(3.0)) / np.log(20.0),
                        np.minimum(Ma_, Mb_), np.maximum(Ma_, Mb_))
        af_lin = ((cz("a0")[:, :, 0, :] + cz("a1")[:, :, 0, :] * lnR)
                  + (cz("b0")[:, :, 0, :] + cz("b1")[:, :, 0, :] * lnR) * (np.minimum(Mb, Mref1) - Mref1)
                  + cz("a2")[:, :, 0, :] * (lnR - np.log(cz("Rref")[:, :, 0, :])) ** 2
                  + cz("b2")[:, :, 0, :] * (np.minimum(Mb, Mref1) - cz("Mref2")[:, :, 0, :]) ** 2
                  + cz("a3")[:, :, 0, :] * (np.maximum(Mb, Mref1) - Mref1))[:, :, None, :]
        Y = exp_nodes[None] / cz("AFscale")
        nonlin = cz("f2") * np.log((Y + cz("f3")) / cz("f3"))
        af_med = np.clip(af_lin + nonlin, np.log(cz("AFmin")), np.log(cz("AFmax")))
        # amplification residual std; the dispersion model is defined on lnSA in g,
        # so shift the reference nodes from cm/s2 to g before the clip (matches gmm_V7.af_variance)
        lnY_g = nodes[None] - np.log(cz("AFscale"))
        phi = np.clip(cz("s1") + (cz("s2") - cz("s1")) * (lnY_g - np.log(cz("xl"))) / np.log(cz("xh") / cz("xl")),
                      np.minimum(cz("s1"), cz("s2")), np.maximum(cz("s1"), cz("s2")))
        # epistemic: deterministic s2s shift; aleatory: no shift (random residual)
        af_eff = af_med if s2s_mode == "aleatory" else af_med + s2s * phi
        cond_mean = np.mean(nodes[None] + af_eff + wierde[None, None, None, :], axis=3)

        # first/second derivatives of the (active) non-linear amplification
        active = (af_med > np.log(cz("AFmin")) + 1e-9) & (af_med < np.log(cz("AFmax")) - 1e-9)
        deriv = np.where(active, cz("f2") * Y / (Y + cz("f3")), 0.0)
        deriv2 = np.where(active, cz("f2") * cz("f3") * Y / (Y + cz("f3")) ** 2, 0.0)
        res_diag = np.diagonal(residual_cov, axis1=1, axis2=2)        # (MR, P)

        # Jensen curvature correction: the omitted reference directions shift the
        # conditional mean by 1/2 * af'' * Var, captured to second order
        cond_mean = cond_mean + 0.5 * np.einsum("zmip,mp->zmi", deriv2, res_diag) / P

        # linearised residual variance from the omitted reference directions
        sens = (1.0 + deriv) / P
        cond_var = np.einsum("zmip,mpq,zmiq->zmi", sens, residual_cov, sens)

        # aleatory amplification residual: tau^T R_af tau / P^2  (phi = af std at nodes)
        if s2s_mode == "aleatory":
            cond_var = cond_var + np.einsum("zmip,pq,zmiq->zmi", phi, af_correlation, phi) / (P * P)

        means[z0:z1] = cond_mean
        variances[z0:z1] = cond_var

    k1 = means @ weights
    delta = means - k1[:, :, None]
    k2 = np.maximum((delta ** 2 + variances) @ weights, 1e-300)
    k3 = (delta ** 3 + 3.0 * delta * variances) @ weights

    return Scenarios(zones=gmm.zones, magnitudes=magnitudes, distances=distances,
                     weights=weights, means=means, variances=variances, k1=k1, k2=k2, k3=k3)
