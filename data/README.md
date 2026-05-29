# Bundled data — provenance and licence

All files here are model **inputs**; they are small and bundled so the package
runs out of the box. They derive from the publicly released TNO SHRA-Groningen
model chain (EUPL-1.2). The large `im_lookup_cos.zarr` is **not** bundled — it is
a generated artifact (`scripts/prepare_im_lookup.py`).

| file | what it is | origin |
|------|------------|--------|
| `gmm_config.zarr` (group `GMM-V7`) | GMM-V7 reference-median, source/path (tau, phiss), per-zone amplification coefficients, period-to-period correlation, site-to-site epsilons, surface (wierde) factors, and logic-tree branch weights, over 160 site-response zones and 10 FCM periods | TNO SHRA-Groningen-hazard-risk-models (GMM-V7) |
| `Fragility_v7_MiddleBranch_20200127.csv` | FCM-V7 fragility for 35 structural systems: lognormal `b0, b1, sigma` and displacement limits `DL_DS2..DL_DS4` (damage) and `DL_CS1..DL_CS3` (collapse) | TNO FCM-V7 (middle branch) |
| `Geological_zones_V6/` | shapefile of the geological / site-response zone polygons (EPSG:28992, Dutch RD). The `ID_V6` field matches the GMM-V7 zone ids 160/160 | TNO / NAM geological zonation V6 |
| `Groningen_field_outline.csv`, `coast_outline.csv` | geographic context for the maps (RD coordinates) | TNO SHRA-Groningen-seismic-source-model `res/` |
| `zone_sizes.pkl` | per-zone area (numpy array, 160 values); used only as a transparent **exposure proxy** in `compare_risk_metrics.py` | site-response amplification model |

## The seismic source is NOT real here

This repository ships **no seismicity forecast**. `saavg_cos.source` provides a
transparent Gutenberg-Richter stand-in so the chain runs end to end. Absolute
hazard / risk levels are therefore illustrative; the spatial pattern reflects
real site response and the **method comparison** (COS vs Monte-Carlo vs normal)
is independent of the source scale. To produce real Groningen numbers, replace
`source.rate_grid` with the official forecast (TNO SHRA-Groningen-seismic-source-
model + its Zenodo inputs, DOI 10.5281/zenodo.10245813) and add a building
exposure database.

## Licence

EUPL-1.2, consistent with the upstream TNO model chain. See `../LICENSE`.
