"""
Export pipeline data to static GeoJSON for the Vercel frontend.

Run on Colab after the pipeline completes:
    %run scripts/export_web_data.py

Or locally:
    python scripts/export_web_data.py

Outputs to web/public/data/:
    buildings.geojson   — Building footprints with risk properties
    fire_isochrones.geojson — Fire arrival time contour lines
    fire_intensity.geojson  — Sampled fireline intensity points
    stats.json          — Campus-wide aggregate statistics
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLAB_MODE = os.getenv("COLAB_MODE", "false").lower() == "true"


def get_data_root() -> Path:
    if COLAB_MODE:
        return Path("/content/drive/MyDrive/DurhamFireRisk/data/processed")
    return Path("data/processed")


def export_buildings(data_root: Path, out_dir: Path) -> dict:
    """Load twins and export as GeoJSON FeatureCollection."""
    from shapely.geometry import mapping
    from twin.property_twin import PropertyTwin

    twin_dir = data_root / "twins"
    if not twin_dir.exists():
        print(f"No twins directory at {twin_dir}")
        return {}

    twin_files = sorted(twin_dir.glob("*.json"))
    print(f"Found {len(twin_files)} twin files")

    features = []
    scores = []

    for tf in twin_files:
        twin = PropertyTwin.load(tf)
        if twin.geometry is None:
            continue

        score = twin.wildfire_risk_score
        scores.append(score)

        # Risk category
        if score < 30:
            category = "Low"
        elif score < 55:
            category = "Moderate"
        elif score < 75:
            category = "High"
        else:
            category = "Very High"

        # Risk breakdown (same logic as risk_map.py)
        from models.risk.wildfire_scorer import ROOF_MATERIAL_MAP, TPI_CLASS_MAP

        slope = twin.slope_degrees
        hli = twin.heat_load_index
        tpi_enc = TPI_CLASS_MAP.get(twin.tpi_class, 1)
        terrain_score = min(slope * 0.5, 15) + hli * 5 + tpi_enc * 2

        z1, z2, z3 = twin.zone1_fuel_load, twin.zone2_fuel_load, twin.zone3_fuel_load
        ladder = float(twin.ladder_fuel_present)
        veg_score = min(z1 * 20, 12) + min(z2 * 5, 10) + min(z3 * 2, 5) + ladder * 8

        roof_enc = ROOF_MATERIAL_MAP.get(twin.roof_material, 2)
        roof_pts = {0: 0, 1: 5, 2: 12, 3: 20}.get(roof_enc, 10)
        vent = 1.0 if twin.vent_screening_status == "screened" else 0.0
        struct_score = roof_pts + (1 - vent) * 12

        nbr = twin.neighbor_distance_m
        ember = twin.ember_exposure_probability
        exposure_score = (max(0, 10 - nbr * 0.5) if nbr < 20 else 0) + ember * 5

        # Convert geometry to WGS84
        import geopandas as gpd
        from shapely.geometry import shape

        gdf = gpd.GeoDataFrame(
            [{"geometry": twin.geometry}], crs="EPSG:32617"
        ).to_crs("EPSG:4326")
        geom_wgs84 = gdf.geometry.iloc[0]

        feature = {
            "type": "Feature",
            "geometry": mapping(geom_wgs84),
            "properties": {
                "id": twin.parcel_id,
                "name": twin.name or None,
                "address": twin.address or None,
                "riskScore": round(score, 1),
                "riskCategory": category,
                "terrain": round(terrain_score, 1),
                "terrainMax": 25,
                "vegetation": round(veg_score, 1),
                "vegetationMax": 30,
                "structure": round(struct_score, 1),
                "structureMax": 35,
                "exposure": round(exposure_score, 1),
                "exposureMax": 10,
                "slope": round(slope, 1),
                "aspect": round(twin.aspect_degrees, 1),
                "heatLoadIndex": round(hli, 3),
                "tpiClass": twin.tpi_class,
                "roofMaterial": twin.roof_material.replace("_", " ").title(),
                "ventScreening": twin.vent_screening_status.title(),
                "yearBuilt": twin.year_built if twin.year_built > 0 else None,
                "stories": twin.stories,
                "buildingSqFt": round(twin.building_sf, 0),
                "canopyCover": round(twin.canopy_cover_pct, 1),
                "ndviMean": round(twin.ndvi_mean, 3),
                "ladderFuel": twin.ladder_fuel_present,
                "fireArrivalMin": (
                    round(twin.fire_arrival_time_p50, 1)
                    if twin.fire_arrival_time_p50 < 1e6
                    else None
                ),
                "emberProbability": round(ember, 3),
                "isDuke": twin.is_duke_owned,
            },
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}
    out_path = out_dir / "buildings.geojson"
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    print(f"Exported {len(features)} buildings → {out_path}")

    return {"scores": scores, "count": len(features)}


def export_fire_simulation(data_root: Path, out_dir: Path) -> dict:
    """Export fire isochrones and intensity from simulation NetCDF."""
    import xarray as xr
    from pyproj import Transformer

    sim_path = data_root / "terrain" / "fire_simulation.nc"
    if not sim_path.exists():
        print(f"No fire simulation at {sim_path}")
        return {}

    ds = xr.open_dataset(sim_path)
    toa = ds["time_of_arrival"].values.copy()
    xs = ds.x.values.copy()
    ys = ds.y.values.copy()
    intensity = (
        ds["fireline_intensity"].values.copy()
        if "fireline_intensity" in ds
        else None
    )
    crs = ds.attrs.get("crs", "EPSG:32617")
    ds.close()

    if xs.max() < 1000:
        print("Simulation has no georeferenced coordinates — skipping.")
        return {}

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    valid_toa = toa[~np.isnan(toa)]
    max_toa = float(valid_toa.max()) if valid_toa.size > 0 else 0
    n_burned = int((~np.isnan(toa)).sum())
    print(f"Simulation: {n_burned} burned cells, max arrival={max_toa:.1f} min")

    # Isochrones
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import mapping
    from shapely.geometry import Polygon as ShapelyPolygon

    toa_filled = np.nan_to_num(toa, nan=9999.0)

    # Generate time levels dynamically based on max_toa
    if max_toa <= 5:
        levels = [0.5, 1, 2, 3, 4, 5]
    elif max_toa <= 30:
        levels = [1, 2, 5, 10, 15, 20, 30]
    elif max_toa <= 120:
        levels = [5, 10, 20, 30, 60, 90, 120]
    else:
        levels = [10, 30, 60, 120, 180, 240, 360]

    levels = [l for l in levels if l <= max_toa * 1.5]

    iso_features = []
    fig, ax = plt.subplots()
    for time_min in levels:
        try:
            cs = ax.contour(xs, ys, toa_filled, levels=[time_min])
            segments = cs.allsegs[0] if hasattr(cs, "allsegs") else []
            for verts in segments:
                if len(verts) < 3:
                    continue
                lons, lats = transformer.transform(verts[:, 0], verts[:, 1])
                coords = list(zip(lons.tolist(), lats.tolist()))
                iso_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                    "properties": {
                        "minutes": time_min,
                        "label": f"{time_min} min" if time_min >= 1 else f"{int(time_min*60)} sec",
                    },
                })
        except Exception as e:
            print(f"Contour at {time_min}min failed: {e}")
        ax.clear()
    plt.close(fig)

    iso_geojson = {"type": "FeatureCollection", "features": iso_features}
    iso_path = out_dir / "fire_isochrones.geojson"
    with open(iso_path, "w") as f:
        json.dump(iso_geojson, f)
    print(f"Exported {len(iso_features)} isochrone lines → {iso_path}")

    # Intensity heatmap points
    intensity_features = []
    if intensity is not None:
        step = max(1, min(toa.shape) // 100)
        for r in range(0, toa.shape[0], step):
            for c in range(0, toa.shape[1], step):
                if not np.isnan(toa[r, c]) and intensity[r, c] > 0:
                    lon, lat = transformer.transform(float(xs[c]), float(ys[r]))
                    intensity_features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat],
                        },
                        "properties": {
                            "intensity": round(float(intensity[r, c]), 1),
                            "arrivalMin": round(float(toa[r, c]), 2),
                        },
                    })

    int_geojson = {"type": "FeatureCollection", "features": intensity_features}
    int_path = out_dir / "fire_intensity.geojson"
    with open(int_path, "w") as f:
        json.dump(int_geojson, f)
    print(f"Exported {len(intensity_features)} intensity points → {int_path}")

    return {"maxArrival": max_toa, "burnedCells": n_burned, "totalCells": toa.size}


def export_stats(building_stats: dict, fire_stats: dict, out_dir: Path) -> None:
    """Export aggregate campus statistics."""
    scores = building_stats.get("scores", [])
    stats = {
        "buildingCount": building_stats.get("count", 0),
        "meanRisk": round(float(np.mean(scores)), 1) if scores else 0,
        "maxRisk": round(float(np.max(scores)), 1) if scores else 0,
        "minRisk": round(float(np.min(scores)), 1) if scores else 0,
        "highRiskCount": sum(1 for s in scores if s >= 55),
        "moderateRiskCount": sum(1 for s in scores if 30 <= s < 55),
        "lowRiskCount": sum(1 for s in scores if s < 30),
        "fireMaxArrival": fire_stats.get("maxArrival", 0),
        "fireBurnedCells": fire_stats.get("burnedCells", 0),
    }

    out_path = out_dir / "stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Exported stats → {out_path}")


def main():
    data_root = get_data_root()
    out_dir = Path("web/public/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {data_root}")
    print(f"Output: {out_dir}")
    print()

    building_stats = export_buildings(data_root, out_dir)
    fire_stats = export_fire_simulation(data_root, out_dir)
    export_stats(building_stats, fire_stats, out_dir)
    print("\nDone! Data exported for frontend.")


if __name__ == "__main__":
    main()
