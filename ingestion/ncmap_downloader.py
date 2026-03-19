"""
NC OneMap LiDAR downloader and PDAL processing pipeline.

Downloads LAZ point cloud tiles for Durham/Orange County and processes them
into raster products: bare earth DEM, canopy height model (CHM), building
footprints, and intensity rasters.

Usage:
    python ingestion/ncmap_downloader.py

Colab:
    Set COLAB_MODE = True at the top of this file before running on Colab.
"""

from __future__ import annotations

# ── Colab mode flag ──────────────────────────────────────────────────────────
COLAB_MODE = False  # Set to True when running on Google Colab
# ─────────────────────────────────────────────────────────────────────────────

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

from ingestion.config_loader import get_paths, get_study_area, load_config


def download_lidar_tiles(
    county: str,
    bbox: tuple[float, float, float, float],
    out_dir: Path,
) -> list[Path]:
    """
    Query NC OneMap for available LiDAR tiles intersecting bbox and download them.

    Parameters
    ----------
    county:
        County name, e.g. "Durham" or "Orange".
    bbox:
        (xmin, ymin, xmax, ymax) in EPSG:4326.
    out_dir:
        Destination directory for LAZ files.

    Returns
    -------
    List of local LAZ file paths that were downloaded.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bbox

    # NC OneMap WCS / REST endpoint — substitute real tile index when available
    # The tile index is a WFS layer listing available LiDAR tiles
    wfs_url = "https://geodata.lib.ncsu.edu/geoserver/ows"
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "nc:lidar_tile_index",
        "outputFormat": "application/json",
        "bbox": f"{xmin},{ymin},{xmax},{ymax},EPSG:4326",
        "CQL_FILTER": f"COUNTY='{county}'",
    }

    logger.info(f"Querying NC OneMap tile index for {county} County...")
    try:
        resp = requests.get(wfs_url, params=params, timeout=30)
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except Exception as e:
        logger.warning(f"Tile index query failed: {e}. Using offline tile list if available.")
        features = _load_cached_tile_index(county, out_dir)

    downloaded: list[Path] = []
    for feat in features:
        props = feat.get("properties", {})
        download_url: Optional[str] = props.get("download_url") or props.get("url")
        if not download_url:
            continue

        filename = Path(download_url).name
        local_path = out_dir / county / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            logger.debug(f"Already downloaded: {local_path.name}")
            downloaded.append(local_path)
            continue

        logger.info(f"Downloading {filename} ...")
        _download_file(download_url, local_path)

        if _validate_laz(local_path):
            downloaded.append(local_path)
            _log_tile_metadata(local_path, props)
        else:
            logger.warning(f"Integrity check failed for {filename}; removing.")
            local_path.unlink(missing_ok=True)

    logger.info(f"Downloaded {len(downloaded)} LiDAR tiles for {county} County.")
    return downloaded


def process_lidar_to_rasters(
    laz_paths: list[Path],
    out_dir: Path,
    resolution_m: float = 1.0,
) -> dict[str, Path]:
    """
    Run PDAL pipelines on LAZ files to produce GeoTIFF rasters.

    Products:
        - bare_earth_dem: Ground-classified (class 2) DEM at resolution_m
        - dsm: Digital Surface Model (highest return)
        - chm: Canopy Height Model = DSM - DEM
        - intensity: Mean LiDAR intensity raster (proxy for surface material)
        - building_footprints: Raster mask of class-6 (building) returns

    Parameters
    ----------
    laz_paths:
        List of LAZ file paths to process.
    out_dir:
        Output directory for GeoTIFF rasters.
    resolution_m:
        Output raster resolution in meters.

    Returns
    -------
    dict mapping product name → output GeoTIFF path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    for laz in laz_paths:
        stem = laz.stem
        logger.info(f"Processing {laz.name} with PDAL...")

        # --- Ground DEM ---
        dem_out = out_dir / f"{stem}_dem.tif"
        if not dem_out.exists():
            pipeline = _build_pdal_pipeline(
                laz, dem_out, resolution_m, filters=["ground"], writer_type="gdal"
            )
            _run_pdal(pipeline)
        outputs["bare_earth_dem"] = dem_out

        # --- DSM (all returns) ---
        dsm_out = out_dir / f"{stem}_dsm.tif"
        if not dsm_out.exists():
            pipeline = _build_pdal_pipeline(
                laz, dsm_out, resolution_m, filters=[], writer_type="gdal", dimension="Z"
            )
            _run_pdal(pipeline)
        outputs["dsm"] = dsm_out

        # --- Building mask (class 6) ---
        bldg_out = out_dir / f"{stem}_buildings.tif"
        if not bldg_out.exists():
            pipeline = _build_pdal_pipeline(
                laz, bldg_out, resolution_m,
                filters=["Classification == 6"], writer_type="gdal"
            )
            _run_pdal(pipeline)
        outputs["building_footprints"] = bldg_out

        # --- Intensity ---
        intensity_out = out_dir / f"{stem}_intensity.tif"
        if not intensity_out.exists():
            pipeline = _build_pdal_pipeline(
                laz, intensity_out, resolution_m,
                filters=[], writer_type="gdal", dimension="Intensity"
            )
            _run_pdal(pipeline)
        outputs["intensity"] = intensity_out

    # CHM = DSM - DEM (computed after both are ready)
    if "dsm" in outputs and "bare_earth_dem" in outputs:
        chm_out = out_dir / "chm.tif"
        if not chm_out.exists():
            _compute_chm(outputs["dsm"], outputs["bare_earth_dem"], chm_out)
        outputs["chm"] = chm_out

    return outputs


# ── Private helpers ──────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path) -> None:
    """Stream download a file with progress logging."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
    logger.debug(f"Saved {downloaded / 1e6:.1f} MB → {dest}")


def _validate_laz(path: Path) -> bool:
    """Basic integrity check: file exists and is non-empty."""
    return path.exists() and path.stat().st_size > 1000


def _log_tile_metadata(path: Path, props: dict) -> None:
    """Write tile metadata JSON alongside the LAZ file."""
    meta = {
        "filename": path.name,
        "md5": _md5(path),
        "size_bytes": path.stat().st_size,
        "properties": props,
    }
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cached_tile_index(county: str, out_dir: Path) -> list[dict]:
    """Load a previously saved tile index JSON if network is unavailable."""
    cache = out_dir / f"{county}_tile_index.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    return []


def _build_pdal_pipeline(
    laz: Path,
    out: Path,
    resolution: float,
    filters: list[str],
    writer_type: str = "gdal",
    dimension: str = "Z",
) -> dict:
    """Build a PDAL pipeline dict for processing a LAZ file."""
    pipeline: list[dict] = [{"type": "readers.las", "filename": str(laz)}]

    # SMRF ground classification refinement
    pipeline.append({
        "type": "filters.smrf",
        "slope": 0.15,
        "window": 18.0,
        "threshold": 0.5,
        "cell": 1.0,
    })

    for f in filters:
        if f == "ground":
            pipeline.append({"type": "filters.range", "limits": "Classification[2:2]"})
        elif "==" in f:
            pipeline.append({"type": "filters.range", "limits": f})

    pipeline.append({
        "type": f"writers.{writer_type}",
        "filename": str(out),
        "resolution": resolution,
        "dimension": dimension,
        "output_type": "mean",
        "gdalopts": "COMPRESS=LZW,TILED=YES,BLOCKXSIZE=256,BLOCKYSIZE=256",
    })

    return {"pipeline": pipeline}


def _run_pdal(pipeline: dict) -> None:
    """Serialize pipeline to JSON and invoke PDAL via subprocess."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pipeline, f)
        tmp = f.name
    result = subprocess.run(["pdal", "pipeline", tmp], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"PDAL error: {result.stderr}")
        raise RuntimeError(f"PDAL pipeline failed: {result.stderr[:500]}")
    Path(tmp).unlink(missing_ok=True)


def _compute_chm(dsm: Path, dem: Path, out: Path) -> None:
    """Compute Canopy Height Model = DSM - DEM using rasterio."""
    import numpy as np
    import rasterio

    with rasterio.open(dsm) as ds_src, rasterio.open(dem) as de_src:
        dsm_arr = ds_src.read(1).astype(float)
        dem_arr = de_src.read(1).astype(float)
        chm_arr = np.maximum(dsm_arr - dem_arr, 0.0)  # No negative heights
        profile = ds_src.profile.copy()
        profile.update(dtype="float32", compress="lzw")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(chm_arr.astype("float32"), 1)
    logger.info(f"CHM written to {out}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    paths = get_paths(COLAB_MODE)
    cfg = load_config("data_sources.yaml")
    sa = get_study_area(cfg)
    bbox = (
        sa["bbox"]["xmin"],
        sa["bbox"]["ymin"],
        sa["bbox"]["xmax"],
        sa["bbox"]["ymax"],
    )

    all_laz: list[Path] = []
    for county in cfg["lidar"]["counties"]:
        laz_files = download_lidar_tiles(
            county=county,
            bbox=bbox,
            out_dir=paths["raw_lidar"],
        )
        all_laz.extend(laz_files)

    if all_laz:
        rasters = process_lidar_to_rasters(
            laz_paths=all_laz,
            out_dir=paths["processed_terrain"],
            resolution_m=cfg["lidar"]["resolution_m"],
        )
        logger.info(f"Produced rasters: {list(rasters.keys())}")
    else:
        logger.warning("No LiDAR tiles downloaded. Check network access or tile index.")


if __name__ == "__main__":
    main()
