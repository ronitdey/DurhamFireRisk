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

## v0.3 — 2026-03-19

- Pivoted from full-Durham ingestion to single-building PoC on Randolph Residence Hall (50 Brodie Gym Drive) after hitting blocked/unreliable external endpoints for county-scale LiDAR
- Added `laspy` + `scipy` fallback in `ncmap_downloader.py` so LiDAR processing works on Colab without the PDAL CLI, which can't be pip-installed on Python 3.12; successfully produced DEM, DSM, CHM, building mask, and intensity rasters at 186×271 from the Randolph Hall LAS file
- Switched NOAA weather from LCD to GHCND dataset — LCD was returning persistent 500 errors even with a valid token; updated the record parser since GHCND uses a pivoted (one row per datatype) format instead of LCD's flat hourly rows; added synthetic Durham NC climatology fallback when CDO times out

---

## v0.4 - 2026-03-19


- Fixed LANDFIRE ingestion: the API requires an email address (was silently returning empty responses without one), layer names need year-versioned prefixes (`LF2024_FBFM40`, `LF2024_CC`, etc.), and all products now submit as one job; added synthetic TU1 fallback (East Campus open lawn/hardwood setting, not TU5 West Campus) for when the API is unreachable; fixed FBFM40 validation range from `[1, 99]` to `[1, 299]` to cover full Scott-Burgan integer encoding
- Fixed wind rose fire-weather thresholds — original thresholds (RH≤25%, wind≥15mph, temp≥90°F) were designed for hourly data but GHCND is daily averages so zero days matched; relaxed to daily-equivalent thresholds (RH≤40%, wind≥8mph, TMAX≥85°F)
- Added local file loading to parcel fetcher — now auto-detects GeoJSON/Shapefile/GeoPackage in the parcels directory before falling back to ArcGIS REST or synthetic; handles filenames with spaces
- Added job-ID TIF splitter to LANDFIRE fetcher — when a manually downloaded clip-and-ship TIF is placed in the landfire directory, it inspects the band count and splits into per-product files automatically
- Dropped Orange County from parcel scope; study area bbox tightened to 1km around Randolph Hall

