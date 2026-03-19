# Duke Climate Risk Engine

A property-level wildfire and flood risk platform built on physics-based simulation. Applied to Duke University's campus and the Durham/Orange County region.

---

## Why This Exists

Traditional property risk tools ask: *where is this property?*

This platform asks: *given this property's exact physical characteristics, how would fire or flood actually behave here and what specific changes would reduce that risk?*

Those are fundamentally different questions. The first produces a zip-code score. The second produces a mitigation plan with a price tag per risk point reduced.

This project implements it for Duke's campus and demonstrates it on a real, underexplored geography.

---

## Key Architecture

```
Sensor Data (LiDAR, NAIP, LANDFIRE, Weather)
        ↓
Feature Engineering (Terrain, Vegetation, Structure)
        ↓
Digital Twin (PropertyTwin — one per parcel)
        ↓
Physics Simulation (Rothermel fire spread, Huygens propagation)
        ↓
Risk Scoring (XGBoost + SHAP attribution)
        ↓
Mitigation Ranking (counterfactual scenario runner)
        ↓
API + Interactive Map
```

---

## Key Results

*(Populated after running on real data)*

| Metric | Value |
|---|---|
| Duke campus buildings analyzed | — |
| Highest-risk building | — |
| Top risk driver (campus-wide) | — |
| Top mitigation action | — |
| Avg risk reduction from top action | — % |
| Simulation: fire reaches N buildings in 30 min (SW wind 25 mph) | — |

---

## Methodology at a Glance

### Fire Spread: Rothermel (1972)
Implements the same foundational physics model used in FARSITE, FlamMap, and BehavePlus. Rate of spread is a function of:
- Fuel load, surface-to-volume ratio, moisture of extinction (from LANDFIRE FBFM40)
- Actual fuel moisture (from NOAA CDO)
- Wind speed and direction (with terrain adjustment)
- Terrain slope and aspect

This is not a location-based lookup. Every cell in the simulation domain has individually computed fire behavior.

### Ember Transport: Albini (1979) / Cohen & Stratton (2008)
Spotting is the primary mechanism by which fires jump roads and firebreaks. During the 2025 Pacific Palisades fire, embers were carried over a mile ahead of the main front. The platform models this explicitly.

### Structure Classification: CNN-ViT Backbone
A hybrid CNN-ViT architecture (adapted from DISTRACT) classifies roof material and detects vegetation-to-structure proximity from 4-band NAIP imagery at 0.6m resolution.

### Attribution: SHAP
SHAP values decompose each property's risk score into contributions from every feature — separating controllable (structure, vegetation) from uncontrollable (terrain, location) factors. This is the technical foundation for mitigation prioritization.

### Why NC?
NC has real wildfire exposure — the Piedmont and mountains see 3,000–5,000 fires annually — and demonstrated flood risk from events like Hurricane Helene (September 2024). Durham and Orange County are data-rich and underserved by modern climate risk tools.

---

## Installation

```bash
# Clone repo and install deps
pip install -r requirements.txt

# Install PDAL (for LiDAR processing) via conda
conda install -c conda-forge pdal python-pdal

# Set API keys (see configs/data_sources.yaml for required env vars)
export NOAA_CDO_TOKEN=your_token_here
export PC_SDK_SUBSCRIPTION_KEY=your_key_here

# Run ingestion pipeline (downloads ~10GB of data)
python ingestion/pipeline_runner.py

# Build digital twins
python twin/twin_builder.py

# Start API
uvicorn api.main:app --reload

# Generate risk map
python visualization/risk_map.py
```

### Google Colab Setup
All training scripts support Colab. Set `COLAB_MODE = True` at the top of:
- `ingestion/pipeline_runner.py`
- `models/vision/train_roof.py`
- `twin/twin_builder.py`

Then mount your Google Drive and point configs to `/content/drive/MyDrive/DurhamFireRisk/`.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/assess` | POST | Full risk assessment for a property |
| `/mitigate` | POST | Counterfactual: quantify risk reduction from actions |
| `/explain/{parcel_id}` | GET | SHAP attribution breakdown |
| `/simulate` | POST | Run Rothermel fire spread simulation |
| `/campus-overview` | GET | Aggregated Duke campus statistics |

Example:
```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{"address": "Duke Chapel, Durham NC", "include_wildfire": true}'
```

---

## Validation

Four-part validation strategy (see `METHODOLOGY.md` §8):

1. **NC fire incident perimeter comparison** — simulated spreads vs. NC Forest Service perimeters
2. **Cross-validation vs. USFS fire hazard severity zones** — systematic bias check
3. **Structural loss correlation** — FEMA Helene building damage data vs. vulnerability scores
4. **Physical invariant tests** — automated tests proving physics relationships hold (see `tests/test_simulation.py`)

---

## Repository Structure

```
duke-climate-risk-engine/
├── ingestion/          # Data download pipelines (LiDAR, NAIP, LANDFIRE, NOAA, parcels)
├── features/           # Feature engineering (terrain, vegetation, structure, flood)
├── models/             # Vision backbone, risk scorer, fire simulation, SHAP explainer
├── twin/               # Digital twin construction and mitigation scenario runner
├── visualization/      # Interactive Folium map + report generator
├── api/                # FastAPI endpoints
├── tests/              # Physics invariant and unit tests
└── configs/            # All hyperparameters and data source URLs
```
