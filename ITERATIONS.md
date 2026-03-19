# Project Iterations Log

Tracks every meaningful change to the Duke Climate Risk Engine — what changed, why, what was measured, and what was learned. Failed experiments are documented alongside successes.

---

## v0.1 — 2026-03-19

- Scaffolded the full project today
- Ingestion, terrain features, Rothermel simulation, CNN-ViT backbone, XGBoost scorer, SHAP explainer, digital twin, FastAPI, and Folium map are all framed out
- Flood module, wind field precomputation, PDF reports, and validation scripts are implemented but need real data to drive them
- No real data run yet so no metrics
- **Next:** get Durham County LiDAR through the pipeline and run a first synthetic fire spread test to sanity check the Rothermel math

---

## v0.2 — 2026-03-19

- Fixed HLI formula — was using `(aspect - 225)` as the reference azimuth which gives maximum heat load for NE-facing slopes, the physical opposite of truth; corrected to `(aspect - 45)` per McCune & Keon (2002) where NE is the cold/moist reference
- Fixed grass-vs-litter spread test — GR2 has `M_x=0.15` so live fuel at 70% moisture extinguishes reaction intensity entirely, not a model bug but a wrong test expectation; switched to GR4 (higher dead fuel fraction, `M_x=0.40`) with cured-grass moisture (`M_lh=0.06`)
- Updated HLI test to compare SW-facing vs NE-facing rather than SW vs N, which is the correct comparison against the cold reference azimuth
- 57/57 tests pass
- **Next:** run the LiDAR pipeline on Durham County and smoke test the Rothermel simulation on a real DEM tile

---
