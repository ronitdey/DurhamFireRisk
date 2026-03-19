# Technical Methodology

*Duke Climate Risk Engine — Property-Level Wildfire and Flood Risk Platform*

---

## 1. Problem Framing

### What location-based risk scores miss

Traditional property risk tools assign risk based on administrative geography: zip code, census tract, or county-level hazard severity zone. This conflates the risk of a property with the risk of its neighborhood.

Two adjacent properties with different roof materials, vegetation density, and topographic position face fundamentally different loss profiles under the same fire event. This platform demonstrates that gap empirically and quantifies it by feature.

**Research question:** How much of property-level wildfire risk variance is explained by property-specific features (roof material, vegetation proximity, defensible space) that are invisible to location-based models?

---

## 2. Data Sources and Quality Assessment

| Source | Product | Resolution | Coverage | Quality Notes |
|---|---|---|---|---|
| NC OneMap | LiDAR (LAZ) | 1m bare earth | Durham + Orange Co. | Ground-classified (class 2/6 used) |
| USDA FSA / AWS | NAIP Aerial Imagery | 0.6m, 4-band | Same | Annual; use most recent year |
| LANDFIRE | FBFM40, CC, CH, CBD, CBH, EVT | 30m → 10m | National | Reprojected to UTM Zone 17N |
| USGS 3DEP | DEM | 1/3 arc-sec (~10m) | National | Supplemented by NC OneMap 1m |
| NOAA CDO | ASOS Hourly | — | RDU (USW00013722) | 10 years; gaps filled by interpolation |
| Durham/Orange GIS | Parcel boundaries | — | Study area | Attribute completeness varies |

**Known data quality issues:**
- LANDFIRE is updated on 3-5 year cycles; fuel models may not reflect recent burns or land-cover change
- NAIP acquisition years vary by county; check acquisition date before analysis
- Parcel attribute fields (year_built, roof material) are often null for older structures

---

## 3. Feature Engineering

### 3.1 Terrain Features

All terrain features are computed at 10m resolution on the 1/3 arc-second DEM, reprojected to UTM Zone 17N (EPSG:32617).

**Slope:** Horn's method (3×3 neighborhood). Critical thresholds: <5° flat, 5–30° moderate, >30° extreme. Fire spread rate increases with the *square* of slope gradient in Rothermel's formulation.

**Aspect:** Degrees clockwise from north. Converted to northness (cos(aspect)) and eastness (sin(aspect)) for ML input to avoid circular discontinuity at 0°/360°.

**Topographic Position Index (TPI):** Each cell's elevation relative to mean elevation within a 300m neighborhood. Positive = ridges (ember source, elevated exposure); negative = valleys (smoke/ember accumulation).

**Heat Load Index (McCune & Keon 2002):**
```
HLI = (1 - cos(π/180 × (aspect - 225))) / 2 × sin(slope × π/180)
```
Range 0–1; south/southwest-facing steep slopes score highest. Captures moisture stress (lower soil moisture = drier fuels = higher flammability) that solar radiation models quantify directly.

**Upslope Profile:** For each property, the mean slope in the uphill direction within 100m, 300m, and 500m buffers. This captures the fire approach vector — a critical predictor for properties on mid- or lower-slope positions. A fire burning uphill toward a structure accelerates exponentially with slope. This is entirely invisible to location-based risk models.

**Topographic Wetness Index (TWI):**
```
TWI = ln(upstream_area_m² / tan(slope))
```
High TWI = high soil moisture = flood potential. Also inversely related to fuel dryness and wildfire risk.

### 3.2 Vegetation Features

**NDVI, NDWI, EVI:** Derived from 4-band NAIP (R, G, B, NIR) at native 0.6m resolution, aggregated to parcel statistics (mean, 90th percentile, standard deviation).

**Spectral Mixture Analysis:** Linear unmixing into three endmembers — green vegetation (GV), non-photosynthetic vegetation (NPV / dry material), and soil. NPV fraction is a direct indicator of cured/dead fuel load and fire danger.

**Defensible Space Zone Analysis:**
- Zone 1 (0–5ft): Immediate ignition zone. Any combustible material here creates near-certain ignition from ember contact.
- Zone 2 (5–30ft): Ember landing zone. Quantified by total fuel load (tons/acre), ladder fuel presence, and dominant fuel model.
- Zone 3 (30–100ft): Extended fuel break. Fuel continuity index: fraction of cells with fuel load above ignition threshold.

**LANDFIRE FBFM40 Mapping:** Each Scott-Burgan fuel model code maps to physical parameters (fuel load by size class, fuel bed depth, moisture of extinction, surface-to-volume ratio, heat content) that feed directly into the Rothermel equations. No lookup table approximation — this is the actual physics input.

**Ladder Fuel Detection:** Cells where shrub/understory fuel models coexist with canopy heights >2m indicate vertical fuel continuity enabling surface-to-crown fire transition.

**Neighbor Proximity:** Distance to nearest adjacent structure. At <15m, radiant heat from a burning neighboring structure can directly ignite exterior walls — the primary mechanism of neighborhood-scale destruction documented in post-fire surveys (Maranghides & Mell 2009).

### 3.3 Structure Classification (CNN-ViT Architecture)

See `models/vision/backbone.py` for implementation.

**Architecture:** Hybrid CNN-ViT backbone. CNN stem (3 MBConv stages, EfficientNet-style) reduces 256×256 NAIP patches to 16×16 feature maps. ViT body (6 transformer encoder layers) captures spatial context. Shared backbone with task-specific heads.

**CNN stem rationale:** Convolutional layers are ideal for local texture classification — distinguishing metal panel grain from asphalt granule texture, or clay tile from built-up membrane. The 256×256 @ 0.6m input covers ~150m × 150m, enough context to see the full building plus immediate surroundings.

**ViT body rationale:** Self-attention enables each token to attend to all other tokens — capturing long-range spatial relationships like vegetation-to-structure distance, neighboring building proximity, and terrain context that convolutional features would miss.

**Training strategy:**
1. ImageNet initialization (general visual features)
2. NAIP + LANDFIRE weakly supervised pre-training (vegetation type segmentation at 10m using LANDFIRE labels as weak supervision)
3. Fine-tuning on hand-labeled roof patches (Duke Facilities Management + OSM + field survey)

**Roof classes (8):** Metal standing seam, metal corrugated, concrete/clay tile, asphalt shingles, wood shingles/shake, built-up (tar + gravel), membrane/flat, unknown/occluded.

**Opening detection (experimental):** DETR-based detector for roof vents, soffits, chimney openings. ~70% detection accuracy on labeled patches. Unscreened openings are the primary ember entry pathway during wildfire events.

---

## 4. Fire Spread Simulation

### 4.1 Rothermel Model Implementation

Reference: Rothermel, R.C. (1972). *A Mathematical Model for Predicting Fire Spread in Wildland Fuels.* USDA Forest Service Research Paper INT-115.

The Rothermel model computes rate of spread (ft/min) from fuel physical properties and weather:

```
R = (I_R × ξ × (1 + φ_w + φ_s)) / (ρ_b × ε × Q_ig)
```

Where:
- `I_R`: Reaction intensity (BTU/ft²/min) — energy released per unit area
- `ξ`: Propagating flux ratio — fraction of energy transferred ahead of the fire
- `φ_w`: Wind coefficient — amplification from midflame wind
- `φ_s`: Slope coefficient — amplification from terrain gradient
- `ρ_b`: Bulk density (lb/ft³) — fuel packing
- `ε`: Effective heating number — pre-heating efficiency
- `Q_ig`: Heat of ignition (BTU/lb) — function of fuel moisture

Derived outputs:
- Byram's fireline intensity: `I_B = h × w × R / 60` (BTU/ft/s)
- Flame length: `FL = 0.45 × I_B^0.46` (Byram 1959)

**Limitations vs. full CFD:** The Rothermel model assumes steady-state spread on uniform terrain and fuel. It does not model:
- Spotting (handled separately by the ember transport module)
- Crown fire transition (canopy fuel parameters are used as input but crown ignition is simplified)
- Pyroconvection (fire-induced wind field changes)
- Sub-grid-scale fuel heterogeneity

Full physics approaches (WFDS, FDS) would capture these but require hours of compute per scenario. Rothermel provides a validated, transparent baseline that demonstrates the correct physics framework. This distinction is documented honestly throughout the codebase.

### 4.2 Huygens Wavelet Propagation

Fire perimeter expansion uses a simplified Huygens principle: each perimeter point advances at the Rothermel spread rate in the direction of maximum spread, modulated by wind direction alignment.

Time-of-arrival grid: for each cell in the simulation domain, the time (minutes from ignition) at which the fire front reaches that cell. Cells that are not reached within the simulation window are marked NaN.

Monte Carlo simulations: N runs with randomized wind speed (from 90th percentile range of historical fire-weather observations) and fuel moisture (from FWI-derived distributions) produce a probability-of-exposure raster — the fraction of realistic fire scenarios that reach each cell.

### 4.3 Ember Transport

Spotting model based on Albini (1979) empirical equations for firebrand launch height and lognormal transport distance distribution. Ember landing density mapped across the domain; spot fire ignition probability computed as a function of landing ember count, fuel moisture, and fuel receptivity.

This is the primary mechanism that makes fire behavior nonlinear and why traditional hazard maps — which model contiguous spread only — systematically underestimate risk for properties in or near WUI.

### 4.4 Wind Field Modeling

Terrain-following wind adjustment using documented multipliers (ridge: ×1.4, valley: ×0.6) and pre-computed wind field rasters for 8 × 3 = 24 direction/speed combinations. Avoids real-time CFD at query time.

---

## 5. Flood Risk Modeling

HAND (Height Above Nearest Drainage) method:

```
HAND = cell_elevation - elevation_of_nearest_drainage_cell
```

Low HAND → floods first. Return-period inundation depths estimated from USGS StreamStats peak discharge data for Ellerbe Creek and neighboring gages, converted to water surface elevation via Manning's equation.

Validation against FEMA NFHL 100-year floodplain boundaries.

---

## 6. Risk Scoring Model

XGBoost gradient-boosted tree trained on simulation-derived labels. Training labels are structural survival probabilities from Monte Carlo fire spread simulations: a property that is reached by fire in 70% of stochastic simulation runs under historical fire weather conditions scores approximately 70/100.

Features: 19 inputs spanning terrain, vegetation, structure, and exposure domains. See `models/risk/wildfire_scorer.py` for the complete feature list.

Score calibration: the model is calibrated against historical NC fire incident data and FEMA structural damage records from Hurricane Helene to ensure that a score of X corresponds meaningfully to X-th percentile loss probability in the historical record.

---

## 7. Attribution (SHAP)

SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) decomposes the model prediction into exact additive contributions from each input feature:

```
f(x) = φ_0 + Σ φ_i
```

where `φ_0` is the expected model output across the reference population and `φ_i` is the marginal contribution of feature `i` for this specific property.

Key properties: efficiency (SHAP values sum to model output), symmetry, dummy axiom, and local accuracy. These properties guarantee a mathematically rigorous attribution — not a heuristic approximation.

**For mitigation recommendations:** If `φ_{roof_material}` = +18.4 risk points for a wood shake roof, then replacing it with metal eliminates that exact contribution. The mitigation scenario runner implements this directly via counterfactual: modify the feature, recompute the score, report the delta.

---

## 8. Validation

### 8.1 Fire Simulation Validation

**Dataset:** NC Forest Service fire incident database (historical perimeters, 2010–2023). Hold out 20% from simulation development.

**Metrics:**
- Sørensen similarity coefficient: |A ∩ B| × 2 / (|A| + |B|), where A = simulated perimeter, B = actual perimeter
- Time-of-arrival correlation: Pearson r between simulated and estimated arrival time at perimeter points
- Symmetric area difference: (area predicted wrong) / (area of actual fire)

**Expected performance:** Rothermel simulations on real-world fires typically achieve Sørensen scores of 0.4–0.7 (Andrews et al. 2011). Values above 0.6 indicate strong physical fidelity.

### 8.2 Structural Vulnerability Calibration

**Dataset:** FEMA building damage dataset from Hurricane Helene (2024, western NC). Correlate predicted vulnerability scores with observed damage rates by census tract.

**Metric:** Calibration curve — does a predicted vulnerability score of 70 correspond to ~70% empirical damage rate?

### 8.3 Physical Invariant Tests

Automated unit tests (`tests/test_simulation.py`) verifying:
- Higher slope → higher spread rate (always)
- Higher wind → higher spread rate (always)
- Higher fuel moisture → lower spread rate (always)
- Metal roof → lower risk score than wood shake (always)
- Zero fuel load → zero spread rate (always)
- Moisture above extinction moisture → near-zero spread (always)

These tests run on every code change. If any invariant fails, the physics implementation is broken and simulation results cannot be trusted.

### 8.4 Feature Plausibility Tests

For each feature, compare property-level predictions against county-level benchmarks from NC Forest Service hazard severity zone maps and FEMA flood zone designations. Systematic deviation indicates possible feature extraction error.

---

## 9. Comparison to Location-Based Scoring

*(Populated after running on real data)*

**Experiment design:** For each Duke campus building, compute:
1. Location-based risk score (county fire hazard severity zone × FEMA flood zone)
2. Property-level risk score from this platform

Compute: variance explained by property-level features beyond location.

**Hypothesis:** Properties within the same campus zone will show substantial risk heterogeneity driven by roof material, vegetation proximity, and defensible space — features invisible to location-based models.

**Expected finding:** Based on stand insurance research: property-level features explain 40–60% of variance *within* location-based risk strata.

---

## 10. Limitations and Future Work

**Simulation limitations:**
- Rothermel is a steady-state model; dynamic wind shifts during fire events are not captured
- Crown fire transition is simplified; full Van Wagner (1977) crown fire initiation model could be added
- Fuel moisture is held constant during simulation; real fires occur over hours with changing moisture
- Spotting model uses empirical regression; high-resolution CFD (WFDS) would improve accuracy

**Data limitations:**
- LANDFIRE fuel models at 30m (resampled to 10m) may miss fine-scale fuel heterogeneity
- Roof material classification from 0.6m NAIP has ~75-80% expected accuracy; LiDAR intensity would improve this
- Parcel attributes (year built, owner) have >30% null rates in some counties

**Future extensions:**
- Real-time fire weather integration (RAWS stations, Red Flag Warnings)
- Climate scenario projections (2050 fire weather under RCP 4.5/8.5)
- Multi-structure interaction modeling (fire brands from burning buildings)
- Integration with insurance pricing models (expected annual loss calculation)
- Expansion to Orange County Mountain and western NC high-risk areas

---

*For questions about methodology, see the inline documentation in `models/simulation/fire_spread.py` and the original Rothermel (1972) reference.*
