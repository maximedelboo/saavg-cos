"""
Groningen per-zone hazard map from the COS method.

Integrates the COS SaAvg exceedance against an annual source-rate grid to get a
hazard curve per site-response zone, reads off the SaAvg (g) at the 475- and
2475-year return periods, joins to the geological-zone polygons (the V7 zone ids
match the shapefile's ID_V6 column) and draws the choropleth over the field.

SCOPE: the GMM-V7 intensity model and the zone geometry are real; the seismic
source rate is the transparent Gutenberg-Richter stand-in (saavg_cos.source), so
the spatial pattern reflects real site response but absolute levels are
illustrative until the official Groningen seismicity forecast is plugged in.

    python scripts/make_hazard_map.py
"""

import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from saavg_cos import model, spectral_im, cos_cache, source

LOG_G = np.log(981.0)
DATA = ROOT / "data"
SHP = DATA / "Geological_zones_V6" / "Geological_zones_V6.shp"


def pick_central_branch(ds_labels):
    sel = ((ds_labels["b_median"] == "CentralUpper") & (ds_labels["b_tau"] == "Central") &
           (ds_labels["b_phiss"] == "Upper") & (ds_labels["b_s2s"] == "Central") &
           (ds_labels["surface_condition"] == "regular"))
    return sel


def main():
    gmm = model.load_ground_motion_model("central")
    zones = gmm.zones.astype(str); Z = zones.size
    mags = np.linspace(2.5, 7.0, 41); dists = np.linspace(3.0, 40.0, 41)
    sc = spectral_im.compute_scenarios(gmm, mags, dists, pca_rank=2, gh_order=7, s2s_mode="epistemic")
    k1 = sc.k1.reshape(Z, mags.size, dists.size)
    k2 = sc.k2.reshape(Z, mags.size, dists.size)
    k3 = sc.k3.reshape(Z, mags.size, dists.size)

    levels_g = np.geomspace(6e-4, 3.0, 60)
    pts = np.log(levels_g) + LOG_G
    rate = source.rate_grid(mags, dists, total_rate=1.0)
    exc = cos_cache.exceedance(pts, k1.ravel(), k2.ravel(), k3.ravel()).reshape(
        Z, mags.size, dists.size, levels_g.size)
    lam = np.einsum("zmrl,mr->zl", exc, rate)              # annual exceedance rate per zone

    def rp_level(T):
        target = 1.0 / T; out = np.full(Z, np.nan)
        for z in range(Z):
            m = lam[z] > 0
            if m.sum() < 2:
                continue
            xp = lam[z, m][::-1]; fp = levels_g[m][::-1]
            if xp[0] <= target <= xp[-1]:
                out[z] = np.exp(np.interp(np.log(target), np.log(xp), np.log(fp)))
        return out
    sa = {475: rp_level(475.0), 2475: rp_level(2475.0)}
    for T in (475, 2475):
        print(f"SaAvg(g) @ {T}yr: min/median/max = "
              f"{np.nanmin(sa[T]):.3f}/{np.nanmedian(sa[T]):.3f}/{np.nanmax(sa[T]):.3f}")

    gdf = gpd.read_file(SHP); gdf["ID_V6"] = gdf["ID_V6"].astype(str)
    lut = {z: i for i, z in enumerate(zones)}
    for T in (475, 2475):
        gdf[f"sa{T}"] = gdf["ID_V6"].map(lambda i: sa[T][lut[i]] if i in lut else np.nan)
    print(f"polygons matched: {gdf['sa2475'].notna().sum()}/{len(gdf)}")

    field = None
    f = DATA / "Groningen_field_outline.csv"
    if f.exists():
        a = np.genfromtxt(f, delimiter=",", names=True)
        field = (a["x"], a["y"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, T in zip(axes, (475, 2475)):
        gdf.plot(column=f"sa{T}", ax=ax, cmap="YlOrRd", legend=True, edgecolor="0.5", linewidth=0.2,
                 legend_kwds={"label": f"SaAvg [g] @ {T}-yr return period", "shrink": 0.7},
                 missing_kwds={"color": "lightgrey"})
        if field is not None:
            ax.plot(field[0], field[1], "b-", lw=1.2, alpha=0.7)
        ax.set_title(f"Groningen SaAvg hazard, {T}-yr return period")
        ax.set_xlabel("RD easting [m]"); ax.set_ylabel("RD northing [m]"); ax.set_aspect("equal")
    fig.suptitle("Per-zone hazard from the COS method  "
                 "[real GMM-V7 + zone geometry; illustrative Gutenberg-Richter source]")
    fig.tight_layout()
    out = ROOT / "figures" / "fig_groningen_hazard_map.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
