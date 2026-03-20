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

## v0.4 — 2026-03-19

- Fixed LANDFIRE ingestion: the API requires an email address (was silently returning empty responses without one), layer names need year-versioned prefixes (`LF2024_FBFM40`, `LF2024_CC`, etc.), and all products now submit as one job; added synthetic TU1 fallback (East Campus open lawn/hardwood setting, not TU5 West Campus) for when the API is unreachable; fixed FBFM40 validation range from `[1, 99]` to `[1, 299]` to cover full Scott-Burgan integer encoding
- Fixed wind rose fire-weather thresholds — original thresholds (RH≤25%, wind≥15mph, temp≥90°F) were designed for hourly data but GHCND is daily averages so zero days matched; relaxed to daily-equivalent thresholds (RH≤40%, wind≥8mph, TMAX≥85°F)
- Added local file loading to parcel fetcher — now auto-detects GeoJSON/Shapefile/GeoPackage in the parcels directory before falling back to ArcGIS REST or synthetic; handles filenames with spaces
- Added job-ID TIF splitter to LANDFIRE fetcher — when a manually downloaded clip-and-ship TIF is placed in the landfire directory, it inspects the band count and splits into per-product files automatically
- Dropped Orange County from parcel scope; study area bbox tightened to 1km around Randolph Hall

---

## v0.5 — 2026-03-19

### The CRS bug hunt

This iteration was a deep dive into coordinate reference systems — one of those bugs that passes every check silently and only shows up as zeros in the final output.

**The symptom:** PropertyTwin reported `Slope: 0.0°, Aspect: 0.0°, HLI: 0.0, TWI: 0.0` — every terrain feature was zero despite having valid LiDAR data producing a reasonable-looking DEM (elevation range 409–416 ft).

**Layer 1 — laspy's ScaledArrayView:** The CRS detection function called `las.x.mean()` to check coordinate ranges and infer whether we're in NC State Plane (x ~ 2,000,000) or UTM (x ~ 600,000). But laspy returns `ScaledArrayView` objects, not numpy arrays — they don't have a `.mean()` method. The call was inside a `try/except` that swallowed the `AttributeError`, so it silently fell through to the VLR metadata check, which returned the wrong CRS (EPSG:32617 for data that was actually in EPSG:2264 NC State Plane feet). Fix: `float(np.mean(las.x))`.

**Layer 2 — Missing georeferenced coordinates:** After fixing the CRS, terrain features were still zero. The xarray Dataset from `compute_terrain_features` used bare dimension names `["y", "x"]` without assigning coordinate arrays — so `x` and `y` were just array indices (0, 1, 2, ...). When the twin builder queried `.sel(x=687853, y=3986718)`, it was comparing UTM meters against array indices. `method="nearest"` clamped to the edge of the array, returning edge-of-grid NaN values. Fix: compute real georeferenced coordinate arrays from the rasterio transform and assign them to the xarray Dataset.

**Layer 3 — Parcel centroid outside LiDAR extent:** Even with correct coordinates, the parcel centroid (687854, 3986718 in UTM) was 90 meters east of the LiDAR tile's right edge (687762). The original LAS file covered only ~80m × 57m in real-world space — a 186×271 raster at 1-foot resolution in NC State Plane, which shrinks to an 8×6 grid at 10m UTM resolution. A 97-acre Duke East Campus parcel's centroid will never fall inside an 80m tile. Fix: when the centroid falls outside terrain extent, use the spatial mean of all valid cells instead of a point query — since the tile is entirely within the parcel, the mean is representative.

**Layer 4 — Risk scorer never called:** After all the terrain fixes, risk was still 0.0. Turned out `_build_single_twin` populated features but never ran the `WildfireScorer`. The scorer existed with a complete rule-based fallback (terrain up to 25 pts, vegetation up to 30, structure up to 35, exposure up to 10) but was never wired into the twin builder. Fix: added `_score_risk()` call using `WildfireScorer._fallback_score`.

**What I learned:** Silent exception handling (`except Exception: pass`) is dangerous in geospatial code — a wrong CRS propagates through every downstream computation without ever raising an error. Every step looks fine in isolation (DEM renders correctly, terrain computes non-NaN values, twin builds successfully) but the coordinates don't align across layers. The lesson is to add CRS assertions at layer boundaries, not just at input.

### Scaling from single-tile to East Campus

- Replaced the single Randolph Hall LAS tile (`209894_0.las`, 80m × 57m) with a full East Campus tile (`209938_0.las`) that covers Randolph and the surrounding campus — the parcel centroid now has a chance of falling within the LiDAR extent
- Changed terrain reprojection resolution from 10m to 1m to preserve detail from high-resolution LiDAR (1-foot NC State Plane data was being downsampled to 8×6 cells at 10m)
- Added CRS reprojection in twin builder — `_reproject_centroid()` transforms parcel centroids from their native CRS (EPSG:4326 or EPSG:2264) to match the terrain Dataset CRS (EPSG:32617) using pyproj, eliminating coordinate system mismatches at the extraction step
- **Next:** verify the new tile produces non-zero terrain features and a meaningful risk score; if the full East Campus tile works, consider adding West Campus tiles and running multi-parcel twin builds

---

## v0.6 — 2026-03-20

- Swapped `scipy.griddata` for numpy binned rasterization — 8.3M point East Campus tile went from ~2 hours to ~2 seconds. Had to add nearest-neighbor gap-filling afterward since binning leaves empty cells as NaN and `np.gradient` chokes on those.
- Wired the Rothermel fire spread simulation into the pipeline. It was fully implemented and tested since v0.2 but had never actually run on real terrain. Runs 3 scenarios (worst-case 35mph SW, moderate, low) and saves time-of-arrival grids. Ignition starts at the upwind edge.
- Twin builder now pulls fire arrival time and ember exposure from simulation results, so risk scores reflect actual simulated fire behavior instead of just terrain + structure defaults.
- Built a 3D Plotly visualization: DEM surface + fire spread heatmap + building extrusions from the CHM. First time seeing the full engine output visually.
- **Next:** NAIP imagery for real vegetation indices, per-building parcels, CNN-ViT roof classifier

---

## v0.7 - 2026-03-20


- Implement step for generating an interactive risk map using Folium, including satellite imagery and fire spread isochrones.
- Update risk scoring details in popups with breakdowns for terrain, vegetation, structure, and exposure.
- Add defensible space zones and Duke-owned property outlines to the map.
- Improve GeoDataFrame conversion to include additional risk factors and attributes.

---

## v0.8 - 2026-03-20


- Implemented `fetch_osm_buildings` to retrieve building data from OpenStreetMap.
- Updated pipeline to prefer OSM building footprints over county parcel boundaries.
- Adjusted TwinBuilder to load OSM buildings if available, enhancing data accuracy.

