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
import sys
import time
from pathlib import Path

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

        # Delete old rasters to force reprocessing with latest CRS fix
        for f in paths["processed_terrain"].glob("*.tif"):
            f.unlink()
            logger.info(f"Deleted old raster: {f.name}")

        all_laz = collect_local_laz_files(paths["raw_lidar"])
        if not all_laz:
            logger.info("No local LAZ files found — attempting NC OneMap download...")
            for county in cfg["lidar"]["counties"]:
                laz_files = download_lidar_tiles(county, bbox, paths["raw_lidar"])
                all_laz.extend(laz_files)
        if all_laz:
            lidar_outputs = process_lidar_to_rasters(all_laz, paths["processed_terrain"])
            outputs["lidar"] = lidar_outputs

            # Verify CRS
            import rasterio
            dem_path = lidar_outputs.get("bare_earth_dem")
            if dem_path and dem_path.exists():
                with rasterio.open(dem_path) as src:
                    logger.info(f"DEM CRS: {src.crs}, Bounds: {src.bounds}")

        status["lidar"] = len(all_laz) > 0
        logger.info(f"LiDAR done in {time.time() - t0:.1f}s — {len(all_laz)} file(s)")
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

    # 4. Parcels ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Parcel Boundaries")
    t0 = time.time()
    try:
        from ingestion.parcel_fetcher import fetch_parcels, identify_duke_parcels

        parcels = fetch_parcels(bbox, paths["raw_parcels"])
        parcels = identify_duke_parcels(parcels)
        status["parcels"] = len(parcels) > 0
        logger.info(f"Parcels done in {time.time() - t0:.1f}s — {len(parcels)} parcels")
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

        if dem_path and dem_path.exists():
            terrain = compute_terrain_features(dem_path)
            nc_path = paths["processed_terrain"] / "terrain_features.nc"
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

    # 7. PropertyTwin ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Build PropertyTwin(s)")
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
        logger.info(f"Twins done in {time.time() - t0:.1f}s — {len(twins)} twin(s)")
    except Exception as e:
        logger.error(f"Twin building failed: {e}")
        status["twins"] = False

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
