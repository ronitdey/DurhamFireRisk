"""
Orchestrates the full data ingestion pipeline.

Runs all ingestion steps in order: LiDAR → NAIP → LANDFIRE → DEM → Weather → Parcels.

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
import sys
import time
from pathlib import Path

from loguru import logger

from ingestion.config_loader import get_paths, get_study_area, load_config, ensure_dirs
from ingestion.landfire_fetcher import download_landfire_products, validate_landfire_values
from ingestion.ncmap_downloader import collect_local_laz_files, download_lidar_tiles, process_lidar_to_rasters
from ingestion.noaa_weather import fetch_hourly_data, build_wind_rose, compute_fire_weather_index
from ingestion.parcel_fetcher import fetch_parcels, identify_duke_parcels


def run_full_pipeline(colab_mode: bool = False) -> dict[str, bool]:
    """
    Execute all ingestion steps and return a status report.

    Parameters
    ----------
    colab_mode:
        If True, resolve paths from Google Drive mount.

    Returns
    -------
    dict mapping step name → success bool.
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

    # 1. LiDAR ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: LiDAR Processing")
    t0 = time.time()
    try:
        # Local LAZ files take priority (single-building PoC path)
        all_laz = collect_local_laz_files(paths["raw_lidar"])
        if not all_laz:
            logger.info("No local LAZ files found — attempting NC OneMap download...")
            for county in cfg["lidar"]["counties"]:
                laz_files = download_lidar_tiles(county, bbox, paths["raw_lidar"])
                all_laz.extend(laz_files)
        if all_laz:
            process_lidar_to_rasters(all_laz, paths["processed_terrain"])
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
        logger.error(f"LANDFIRE step failed: {e}")
        status["landfire"] = False

    # 3. Weather ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: NOAA Weather Data")
    t0 = time.time()
    try:
        import json
        import pandas as pd

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
        logger.info(f"Weather done in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.error(f"Weather step failed: {e}")
        status["weather"] = False

    # 4. Parcels ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Parcel Boundaries")
    t0 = time.time()
    try:
        parcels = fetch_parcels(bbox, paths["raw_parcels"], cfg["lidar"].get("output_crs", "EPSG:32617"))
        parcels = identify_duke_parcels(parcels)
        status["parcels"] = len(parcels) > 0
        logger.info(f"Parcels done in {time.time() - t0:.1f}s — {len(parcels)} parcels")
    except Exception as e:
        logger.error(f"Parcels step failed: {e}")
        status["parcels"] = False

    # Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE COMPLETE")
    for step, ok in status.items():
        icon = "✓" if ok else "✗"
        logger.info(f"  {icon} {step}")

    return status


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Durham Fire Risk data ingestion pipeline")
    parser.add_argument("--colab", action="store_true", help="Use Google Drive paths (Colab mode)")
    args = parser.parse_args()

    colab = args.colab or COLAB_MODE
    status = run_full_pipeline(colab_mode=colab)
    sys.exit(0 if all(status.values()) else 1)


if __name__ == "__main__":
    main()
