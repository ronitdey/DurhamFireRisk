# Project Iterations Log

This document tracks every meaningful change made to the Duke Climate Risk Engine,
including what was changed, why, what was measured, and what was learned.
Failed experiments are documented alongside successes.

---

## v0.1 — 2026-03-19

- Scaffolded the full project today. 
- Ingestion, terrain features, Rothermel simulation, CNN-ViT backbone, XGBoost scorer, SHAP explainer, digital twin, FastAPI, and Folium map are all framed out. 
- Flood module, wind field precomputation, PDF reports, and validation
scripts are implemented but need real data to drive them. 
- No real data run yet so no metrics. 
- Next step is getting Durham County LiDAR through the pipeline and running a
first synthetic fire spread test to sanity check the Rothermel math.

---
