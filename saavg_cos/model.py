"""
Model inputs for the spectral chain.

Loads the GMM-V7 reference / amplification parameters (from the parsed
parameter store) and the structural fragility / consequence parameters (from
the V7 fragility table) into plain numpy structures. The heavy zarr/xarray
machinery is used only here, at load time; everything downstream is numpy.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"

# the ten FCM spectral periods (Crowley & Pinho)
FCM_PERIODS = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0])

# conditional probability of dying given a structural collapse state
# (representative Groningen values; the consequence model is a swappable input)
PROBABILITY_OF_DYING = {"CS1": 0.0, "CS2": 0.0035, "CS3": 0.1}


@dataclass
class GroundMotionModel:
    """GMM-V7 parameters at a chosen logic-tree branch (numpy)."""

    periods: np.ndarray            # (P,)
    zones: np.ndarray              # (Z,) zone ids
    median_parameters: np.ndarray  # (P, n_med) reference median coefficients
    parameter_median: list         # names
    tau: np.ndarray                # (P,) source standard deviation
    phiss: np.ndarray              # (P,) path standard deviation
    af_parameters: np.ndarray      # (Z, P, n_af) amplification coefficients
    parameter_af: list             # names
    correlation: np.ndarray        # (P, P) period-to-period correlation
    wierde_factor: np.ndarray      # (P,) deterministic surface term
    s2s_epsilon: float             # site-to-site epistemic epsilon (scalar)


@dataclass
class FragilityModel:
    """Structural fragility / consequence parameters (numpy)."""

    systems: np.ndarray            # (S,) structural system names
    b0: np.ndarray                 # (S,)
    b1: np.ndarray                 # (S,)
    sigma: np.ndarray              # (S,) demand dispersion
    collapse_limits: np.ndarray    # (S, n_cs) displacement limits for CS1..CS3
    collapse_states: list          # ["CS1","CS2","CS3"]
    death_probability: np.ndarray  # (n_cs,) P(dying | collapse state)


def _branch_choice(da, dim, mode):
    """Pick a logic-tree branch index along ``dim`` (mean/central or upper)."""
    size = da.sizes[dim]
    if mode == "central":
        return size // 2
    if mode == "upper":
        # the branch with the largest summed effect (conservative)
        reduce_dims = [d for d in da.dims if d != dim]
        return int(np.argmax(da.sum(reduce_dims).values))
    raise ValueError(f"unknown branch mode: {mode}")


# logic-tree branch dimensions of the GMM-V7 config, in enumeration order
BRANCH_DIMS = ("b_median", "b_tau", "b_phiss", "b_s2s", "surface_condition")


def load_gmm_config():
    """Open and fully load the GMM-V7 config group (small; held in memory)."""
    import xarray as xr

    return xr.open_zarr(DATA / "gmm_config.zarr", group="GMM-V7").load()


def branch_sizes(gc):
    """Sizes of each logic-tree branch dimension, in BRANCH_DIMS order."""
    return tuple(int(gc.sizes[d]) for d in BRANCH_DIMS)


def build_branch_gmm(gc, b_median, b_tau, b_phiss, b_s2s, surface_condition):
    """Build a GroundMotionModel at an explicit logic-tree branch index tuple.

    Only the reference median (``b_median``), the source/path dispersions
    (``b_tau``/``b_phiss``), the site-to-site epistemic epsilon (``b_s2s``) and
    the deterministic surface term (``surface_condition``) depend on the branch;
    the per-zone amplification coefficients and the period correlation are shared
    across all branches.
    """
    periods = np.array([float(s[3:-1]) for s in gc["IM"].values])
    return GroundMotionModel(
        periods=periods,
        zones=np.asarray(gc["zone"].values),
        median_parameters=gc["median_parameters"].isel(b_median=b_median).transpose("IM", "parameter_median").values,
        parameter_median=list(gc["parameter_median"].values),
        tau=gc["tau"].isel(b_tau=b_tau).transpose("IM").values,
        phiss=gc["phiss"].isel(b_phiss=b_phiss).transpose("IM").values,
        af_parameters=gc["af_parameters"].transpose("zone", "IM", "parameter_af").values,
        parameter_af=list(gc["parameter_af"].values),
        correlation=gc["correlation_matrix"].transpose("IM", "IM_T").values,
        wierde_factor=gc["wierde_factor"].isel(surface_condition=surface_condition).transpose("IM").values,
        s2s_epsilon=float(gc["s2s_epsilons"].isel(b_s2s=b_s2s).values),
    )


def load_ground_motion_model(branch="central"):
    """Load GMM-V7 parameters at a single named logic-tree branch."""
    gc = load_gmm_config()
    idx = dict(
        b_median=_branch_choice(gc["median_parameters"].sel(parameter_median="m0"), "b_median", branch),
        b_tau=_branch_choice(gc["tau"], "b_tau", branch),
        b_phiss=_branch_choice(gc["phiss"], "b_phiss", branch),
        b_s2s=_branch_choice(gc["s2s_epsilons"], "b_s2s", branch),
        surface_condition=_branch_choice(gc["wierde_factor"], "surface_condition", branch),
    )
    return build_branch_gmm(gc, **idx)


def load_fragility_model(branch="central"):
    """Load structural fragility/consequence parameters from the V7 table.

    The probability of exceeding a collapse limit state is
    ``Phi((lnSaAvg - c)/s)`` with ``c = (ln(displacement_limit) - b0)/b1`` and
    ``s = sigma/b1`` -- a lognormal CDF in SaAvg, hence the risk expectation is
    closed-form against the Gaussian-mixture SaAvg distribution.
    """
    import pandas as pd

    df = pd.read_csv(DATA / "Fragility_v7_MiddleBranch_20200127.csv")
    collapse_states = ["CS1", "CS2", "CS3"]
    return FragilityModel(
        systems=df["structural_system"].to_numpy(),
        b0=df["b0"].to_numpy(float),
        b1=df["b1"].to_numpy(float),
        sigma=df["sigma"].to_numpy(float),
        collapse_limits=df[["DL_CS1", "DL_CS2", "DL_CS3"]].to_numpy(float),
        collapse_states=collapse_states,
        death_probability=np.array([PROBABILITY_OF_DYING[s] for s in collapse_states]),
    )
