"""
Orchestrates the full data ingestion and feature engineering pipeline.

Runs all steps in order:
  LiDAR → LANDFIRE → Weather → Parcels → Terrain Features → Fuel Models → PropertyTwin

Usage:
    python ingestion/pipeline_runner.py [--colab]

Colab:
    Set COLAB_MODE = True before running on Google Colab, or pass --colab flag.
"""

from __future__ import annotations

# ── Colab mode flag ──────────────────────────────────────────────────────────
COLAB_MODE = False  # Set to True when running on Google Colab
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from ingestion.config_loader import get_paths, get_study_area, load_config, ensure_dirs


def run_full_pipeline(colab_mode: bool = False) -> dict:
    """
    Execute all ingestion and feature engineering steps.

    Parameters
    ----------
    colab_mode:
        If True, resolve paths from Google Drive mount.

    Returns
    -------
    dict with keys: status (step → bool), paths, outputs.
    """
    cfg = load_config("data_sources.yaml")
    paths = get_paths(colab_mode)
    ensure_dirs(paths)
    sa = get_study_area(cfg)
    bbox = (
        sa["bbox"]["xmin"], sa["bbox"]["ymin"],
        sa["bbox"]["xmax"], sa["bbox"]["ymax"],
    )

    status: dict[str, bool] = {}
    outputs: dict[str, object] = {}

    # 1. LiDAR ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: LiDAR Processing")
    t0 = time.time()
    try:
        from ingestion.ncmap_downloader import (
            collect_local_laz_files, download_lidar_tiles, process_lidar_to_rasters,
        )

        # Check if DEM already exists (cached from previous run)
        existing_dems = list(paths["processed_terrain"].glob("*_dem.tif"))
        if existing_dems:
            logger.info(f"LiDAR already processed — {len(existing_dems)} DEM(s) cached, skipping.")
            outputs["lidar"] = {"bare_earth_dem": existing_dems[0]}
            status["lidar"] = True
        else:
            all_laz = collect_local_laz_files(paths["raw_lidar"])
            if not all_laz:
                logger.info("No local LAZ files found — attempting NC OneMap download...")
                for county in cfg["lidar"]["counties"]:
                    laz_files = download_lidar_tiles(county, bbox, paths["raw_lidar"])
                    all_laz.extend(laz_files)
            if all_laz:
                lidar_outputs = process_lidar_to_rasters(all_laz, paths["processed_terrain"])
                outputs["lidar"] = lidar_outputs
            status["lidar"] = len(all_laz) > 0
        logger.info(f"LiDAR done in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.error(f"LiDAR step failed: {e}")
        status["lidar"] = False

    # 2. LANDFIRE ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: LANDFIRE Products")
    t0 = time.time()
    try:
        from ingestion.landfire_fetcher import (
            download_landfire_products, validate_landfire_values, create_synthetic_landfire,
        )

        lf_products = download_landfire_products(
            products=cfg["landfire"]["products"],
            bbox=bbox,
            out_dir=paths["raw_landfire"],
            output_crs=cfg["landfire"]["output_crs"],
            output_res_m=cfg["landfire"]["output_resolution_m"],
        )
        for product, path in lf_products.items():
            validate_landfire_values(path, product)
        status["landfire"] = len(lf_products) > 0
        logger.info(f"LANDFIRE done in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.warning(f"LANDFIRE API failed: {e} — using synthetic fallback")
        try:
            from ingestion.landfire_fetcher import create_synthetic_landfire
            # Get UTM bbox from DEM if available, else convert from WGS84
            dem_path = outputs.get("lidar", {}).get("bare_earth_dem")
            if dem_path and dem_path.exists():
                import rasterio
                with rasterio.open(dem_path) as src:
                    b = src.bounds
                    bbox_utm = (b.left, b.bottom, b.right, b.top)
            else:
                from pyproj import Transformer
                t = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
                x1, y1 = t.transform(bbox[0], bbox[1])
                x2, y2 = t.transform(bbox[2], bbox[3])
                bbox_utm = (x1, y1, x2, y2)
            create_synthetic_landfire(paths["processed_vegetation"], bbox_utm)
            status["landfire"] = True
        except Exception as e2:
            logger.error(f"Synthetic LANDFIRE also failed: {e2}")
            status["landfire"] = False

    # 3. Weather ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: NOAA Weather Data")
    t0 = time.time()
    try:
        from ingestion.noaa_weather import fetch_hourly_data, build_wind_rose

        out_weather = paths["raw_weather"]
        df = fetch_hourly_data(
            station_id=cfg["weather"]["station_id"],
            years=cfg["weather"]["years_history"],
            out_dir=out_weather,
        )
        if not df.empty:
            df.to_parquet(out_weather / "hourly_weather.parquet")
            rose = build_wind_rose(df)
            with open(out_weather / "wind_rose.json", "w") as f:
                json.dump(rose, f, indent=2)
        status["weather"] = not df.empty
        logger.info(f"Weather done in {time.time() - t0:.1f}s — {len(df)} records")
    except Exception as e:
        logger.error(f"Weather step failed: {e}")
        status["weather"] = False

    # 4. Building Footprints & Parcels ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Building Footprints (OSM) & Parcels")
    t0 = time.time()
    try:
        from ingestion.parcel_fetcher import (
            fetch_osm_buildings, fetch_parcels, identify_duke_parcels,
        )

        # Prefer individual OSM building footprints over county parcel boundaries
        buildings = fetch_osm_buildings(bbox, paths["raw_parcels"])
        if not buildings.empty:
            logger.info(f"Using {len(buildings)} OSM building footprints")
            status["parcels"] = True
        else:
            # Fall back to county parcel boundaries
            parcels = fetch_parcels(bbox, paths["raw_parcels"])
            parcels = identify_duke_parcels(parcels)
            status["parcels"] = len(parcels) > 0
        logger.info(f"Buildings/parcels done in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.error(f"Parcels step failed: {e}")
        status["parcels"] = False

    # 5. Terrain Features ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Terrain Feature Engineering")
    t0 = time.time()
    try:
        from features.terrain.slope_aspect import compute_terrain_features

        dem_path = outputs.get("lidar", {}).get("bare_earth_dem")
        if dem_path is None:
            # Find DEM from processed terrain directory
            dems = list(paths["processed_terrain"].glob("*_dem.tif"))
            dem_path = dems[0] if dems else None

        nc_path = paths["processed_terrain"] / "terrain_features.nc"
        if nc_path.exists():
            logger.info("Terrain features already cached, skipping.")
            status["terrain"] = True
        elif dem_path and dem_path.exists():
            terrain = compute_terrain_features(dem_path, resolution_m=1.0)
            terrain.to_netcdf(nc_path)
            outputs["terrain"] = terrain
            status["terrain"] = True
            logger.info(f"Terrain done in {time.time() - t0:.1f}s")
        else:
            logger.error("No DEM found — cannot compute terrain features")
            status["terrain"] = False
    except Exception as e:
        logger.error(f"Terrain step failed: {e}")
        status["terrain"] = False

    # 6. Fuel Models ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Fuel Model Classification")
    t0 = time.time()
    try:
        from features.vegetation.fuel_classifier import map_fuel_models

        fbfm_path = paths["processed_vegetation"] / "FBFM40_10m_utm17n.tif"
        if not fbfm_path.exists():
            fbfm_path = paths["raw_landfire"] / "FBFM40_10m_utm17n.tif"

        if fbfm_path.exists():
            fuel_codes, rothermel = map_fuel_models(fbfm_path)
            outputs["fuels"] = (fuel_codes, rothermel)
            status["fuels"] = True
            logger.info(f"Fuels done in {time.time() - t0:.1f}s")
        else:
            logger.warning("No FBFM40 raster found — skipping fuel classification")
            status["fuels"] = False
    except Exception as e:
        logger.error(f"Fuel classification failed: {e}")
        status["fuels"] = False

    # 7. Fire Spread Simulation ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Fire Spread Simulation")
    t0 = time.time()
    try:
        from models.simulation.fire_spread import FireSpreadSimulator
        from ingestion.config_loader import load_config as _load_cfg

        sim_cfg = _load_cfg("simulation_config.yaml")
        nc_path = paths["processed_terrain"] / "terrain_features.nc"
        sim_out_path = paths["processed_terrain"] / "fire_simulation.nc"

        if sim_out_path.exists():
            logger.info("Fire simulation already cached, skipping.")
            outputs["simulation"] = xr.open_dataset(sim_out_path)
            status["simulation"] = True
        elif nc_path.exists() and status.get("fuels"):
            terrain = xr.open_dataset(nc_path)
            slope_arr = terrain["slope_deg"].values
            aspect_arr = terrain["aspect_deg"].values

            # Get fuel codes array — use the codes from fuel classification
            fuel_codes_da, _ = outputs.get("fuels", (None, None))
            if fuel_codes_da is not None:
                from features.vegetation.fuel_classifier import _FBFM40_CODE_TO_MODEL
                # The fuel grid may be a different shape than terrain.
                # Create a uniform fuel code array matching the terrain grid.
                fuel_code_val = 161  # TU1 default from synthetic LANDFIRE
                if hasattr(fuel_codes_da, 'values'):
                    unique_models = np.unique(fuel_codes_da.values)
                    # Map model name back to code
                    model_to_code = {v: k for k, v in _FBFM40_CODE_TO_MODEL.items()}
                    for m in unique_models:
                        if m not in ("NB1", "NB9"):
                            fuel_code_val = model_to_code.get(m, 161)
                            break
                fuel_codes_grid = np.full(slope_arr.shape, fuel_code_val, dtype="int32")
            else:
                fuel_codes_grid = np.full(slope_arr.shape, 161, dtype="int32")

            # Replace NaN in slope/aspect with 0 for simulation
            slope_arr = np.nan_to_num(slope_arr, nan=0.0)
            aspect_arr = np.nan_to_num(aspect_arr, nan=0.0)

            # Run 3 scenarios from config
            scenarios = sim_cfg.get("scenarios", {})
            all_results = {}
            for name, params in scenarios.items():
                logger.info(f"  Running scenario: {name} "
                            f"(wind={params['wind_speed_mph']}mph @ {params['wind_direction_deg']}°)")

                sim = FireSpreadSimulator(
                    fuel_params_grid={},
                    fuel_model_codes=fuel_codes_grid,
                    slope_grid=slope_arr,
                    aspect_grid=aspect_arr,
                    resolution_m=float(terrain.attrs.get("resolution_m", 1.0)),
                    wind_speed_mph=params["wind_speed_mph"],
                    wind_dir_deg=params["wind_direction_deg"],
                    fuel_moisture={
                        "M_1hr": params.get("fuel_moisture_1hr", 0.06),
                        "M_10hr": params.get("fuel_moisture_1hr", 0.06) * 1.3,
                        "M_100hr": 0.10,
                        "M_lh": 0.80,
                        "M_lw": 1.00,
                    },
                )

                # Ignite from the SW tree line near East Campus
                rows, cols = slope_arr.shape
                ign_lat, ign_lon = 36.00631175301652, -78.91858266750188
                from pyproj import Transformer as _Transformer
                _t = _Transformer.from_crs("EPSG:4326", terrain.attrs.get("crs", "EPSG:32617"), always_xy=True)
                ign_x, ign_y = _t.transform(ign_lon, ign_lat)
                # Map UTM coordinate to grid row/col
                xs, ys = terrain.x.values, terrain.y.values
                ign_col = int(np.argmin(np.abs(xs - ign_x)))
                ign_row = int(np.argmin(np.abs(ys - ign_y)))
                ign_row = max(0, min(rows - 1, ign_row))
                ign_col = max(0, min(cols - 1, ign_col))
                logger.info(f"  Ignition at ({ign_lat:.4f}, {ign_lon:.4f}) → row={ign_row}, col={ign_col}")

                result = sim.simulate_spread(ign_row, ign_col, max_time_minutes=180)
                all_results[name] = result

            # Save worst-case scenario as the primary output
            worst = all_results.get("worst_case", list(all_results.values())[0])
            # Add georeferenced coordinates from terrain dataset
            if "x" in terrain.coords and "y" in terrain.coords:
                worst = worst.assign_coords(x=terrain.x.values, y=terrain.y.values)
                worst.attrs["crs"] = terrain.attrs.get("crs", "EPSG:32617")
            worst.to_netcdf(sim_out_path)
            outputs["simulation"] = worst
            outputs["simulation_all"] = all_results
            status["simulation"] = True

            n_burned = int((~np.isnan(worst["time_of_arrival"].values)).sum())
            logger.info(f"Simulation done in {time.time() - t0:.1f}s — "
                        f"{n_burned} cells burned (worst case)")
        else:
            logger.warning("Missing terrain or fuel data — skipping simulation")
            status["simulation"] = False
    except Exception as e:
        logger.error(f"Fire simulation failed: {e}")
        import traceback
        traceback.print_exc()
        status["simulation"] = False

    # 8. PropertyTwin ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Build PropertyTwin(s)")
    t0 = time.time()
    try:
        from twin.twin_builder import TwinBuilder

        builder = TwinBuilder(paths=paths)
        twins = builder.build_all_twins()
        outputs["twins"] = twins
        status["twins"] = len(twins) > 0
        for twin in twins:
            logger.info(f"  Twin '{twin.name}': risk={twin.composite_risk_score:.1f} "
                        f"[{twin.risk_category()}]")
            if twin.fire_arrival_time_p50 < float("inf"):
                logger.info(f"    Fire arrival: {twin.fire_arrival_time_p50:.1f}min (p50), "
                            f"ember exposure: {twin.ember_exposure_probability:.2f}")
        logger.info(f"Twins done in {time.time() - t0:.1f}s — {len(twins)} twin(s)")
    except Exception as e:
        logger.error(f"Twin building failed: {e}")
        status["twins"] = False

    # 9. Risk Map Visualization ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 9: Interactive Risk Map")
    t0 = time.time()
    try:
        from visualization.risk_map import build_risk_map, twins_to_geodataframe

        twins = outputs.get("twins", [])
        if twins:
            gdf = twins_to_geodataframe(twins)
            map_path = paths["processed"] / "campus_risk_map.html"
            build_risk_map(gdf, map_path, paths=paths)
            outputs["risk_map_path"] = map_path
            status["visualization"] = True
            logger.info(f"Map done in {time.time() - t0:.1f}s → {map_path}")
        else:
            logger.warning("No twins — skipping visualization")
            status["visualization"] = False
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        status["visualization"] = False

    # Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    for step, ok in status.items():
        icon = "✓" if ok else "✗"
        logger.info(f"  {icon} {step}")

    return {"status": status, "paths": paths, "outputs": outputs}


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Durham Fire Risk pipeline")
    parser.add_argument("--colab", action="store_true", help="Use Google Drive paths (Colab mode)")
    args = parser.parse_args()

    colab = args.colab or COLAB_MODE
    result = run_full_pipeline(colab_mode=colab)
    sys.exit(0 if all(result["status"].values()) else 1)


if __name__ == "__main__":
    main()
