"""
LANDFIRE product downloader.

Downloads Scott-Burgan fuel models (FBFM40) and ancillary canopy/vegetation
layers for the study area, then reprojects to EPSG:32617 at 10m resolution.

Usage:
    python ingestion/landfire_fetcher.py

Colab:
    Set COLAB_MODE = True before running on Google Colab.
"""

from __future__ import annotations

# ── Colab mode flag ──────────────────────────────────────────────────────────
COLAB_MODE = False  # Set to True when running on Google Colab
# ─────────────────────────────────────────────────────────────────────────────

import time
import zipfile
from pathlib import Path

import requests
from loguru import logger

from ingestion.config_loader import get_paths, get_study_area, load_config


# LANDFIRE REST API endpoints
_LF_SUBMIT = (
    "https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/"
    "GPServer/LandfireProductService/submitJob"
)
_LF_STATUS = (
    "https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/"
    "GPServer/LandfireProductService/jobs/{job_id}"
)
_LF_RESULT = (
    "https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/"
    "GPServer/LandfireProductService/jobs/{job_id}/results/Output_File"
)


def download_landfire_products(
    products: list[str],
    bbox: tuple[float, float, float, float],
    out_dir: Path,
    output_crs: str = "EPSG:32617",
    output_res_m: int = 10,
) -> dict[str, Path]:
    """
    Submit clip-and-ship requests to LANDFIRE REST API for each product.

    Parameters
    ----------
    products:
        LANDFIRE layer codes, e.g. ["FBFM40", "CC", "CH", "CBD", "CBH", "EVT"].
    bbox:
        (xmin, ymin, xmax, ymax) in EPSG:4326.
    out_dir:
        Directory to save downloaded and processed rasters.
    output_crs:
        Target CRS for reprojected output (default UTM 17N for NC).
    output_res_m:
        Target resolution in meters (default 10m).

    Returns
    -------
    dict mapping product code → reprojected GeoTIFF path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bbox
    aoi_str = f"{xmin} {ymin} {xmax} {ymax}"
    results: dict[str, Path] = {}

    for product in products:
        final_path = out_dir / f"{product}_10m_utm17n.tif"
        if final_path.exists():
            logger.info(f"{product} already downloaded: {final_path.name}")
            results[product] = final_path
            continue

        logger.info(f"Requesting LANDFIRE product: {product}")
        job_id = _submit_landfire_job(product, aoi_str)
        if not job_id:
            logger.error(f"Failed to submit job for {product}")
            continue

        raw_zip = _poll_and_download(job_id, out_dir, product)
        if not raw_zip:
            continue

        extracted = _extract_landfire_zip(raw_zip, out_dir)
        if not extracted:
            continue

        reprojected = _reproject_raster(extracted, final_path, output_crs, output_res_m, product)
        if reprojected:
            results[product] = reprojected

    return results


def _submit_landfire_job(product: str, aoi_str: str) -> str | None:
    """Submit an async clip-and-ship job to LANDFIRE and return the job ID."""
    params = {
        "Layer_Names": product,
        "Area_Of_Interest": aoi_str,
        "Output_Projection": "4326",   # Download in WGS84, reproject locally
        "Resample_Resolution": "30",
        "f": "json",
    }
    try:
        resp = requests.post(_LF_SUBMIT, data=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("jobId")
        logger.debug(f"Submitted LANDFIRE job {job_id} for {product}")
        return job_id
    except Exception as e:
        logger.error(f"LANDFIRE job submission error: {e}")
        return None


def _poll_and_download(
    job_id: str,
    out_dir: Path,
    product: str,
    poll_interval: int = 15,
    max_wait: int = 900,
) -> Path | None:
    """Poll LANDFIRE job status until complete, then download the output ZIP."""
    elapsed = 0
    while elapsed < max_wait:
        try:
            resp = requests.get(
                _LF_STATUS.format(job_id=job_id) + "?f=json", timeout=30
            )
            resp.raise_for_status()
            status = resp.json().get("jobStatus", "")
            logger.debug(f"Job {job_id} status: {status} ({elapsed}s elapsed)")

            if status == "esriJobSucceeded":
                break
            elif status in ("esriJobFailed", "esriJobCancelled", "esriJobTimedOut"):
                logger.error(f"LANDFIRE job {job_id} ended with status: {status}")
                return None
        except Exception as e:
            logger.warning(f"Poll error: {e}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    if elapsed >= max_wait:
        logger.error(f"LANDFIRE job {job_id} timed out after {max_wait}s")
        return None

    # Download result ZIP
    try:
        result_resp = requests.get(
            _LF_RESULT.format(job_id=job_id) + "?f=json", timeout=30
        )
        result_resp.raise_for_status()
        download_url = result_resp.json().get("value", {}).get("url")
        if not download_url:
            logger.error("No download URL in LANDFIRE result")
            return None

        zip_path = out_dir / f"{product}_{job_id}.zip"
        logger.info(f"Downloading {product} from LANDFIRE...")
        _stream_download(download_url, zip_path)
        return zip_path
    except Exception as e:
        logger.error(f"LANDFIRE download error: {e}")
        return None


def _extract_landfire_zip(zip_path: Path, out_dir: Path) -> Path | None:
    """Extract LANDFIRE ZIP and return the primary GeoTIFF path."""
    extract_dir = out_dir / zip_path.stem
    extract_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        tifs = list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.img"))
        if not tifs:
            logger.error(f"No raster found in {zip_path.name}")
            return None
        return tifs[0]
    except Exception as e:
        logger.error(f"ZIP extraction error: {e}")
        return None
    finally:
        zip_path.unlink(missing_ok=True)


def _reproject_raster(
    src: Path, dst: Path, crs: str, res_m: int, product: str
) -> Path | None:
    """
    Reproject and resample a raster to the target CRS and resolution.

    Categorical layers (fuel models, vegetation type) use nearest-neighbor.
    Continuous layers (canopy cover, height, density) use bilinear.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    categorical = product in ("FBFM40", "EVT")
    resample_method = Resampling.nearest if categorical else Resampling.bilinear

    try:
        with rasterio.open(src) as src_ds:
            transform, width, height = calculate_default_transform(
                src_ds.crs, crs, src_ds.width, src_ds.height,
                *src_ds.bounds, resolution=res_m
            )
            profile = src_ds.profile.copy()
            profile.update(
                crs=crs,
                transform=transform,
                width=width,
                height=height,
                compress="lzw",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            with rasterio.open(dst, "w", **profile) as dst_ds:
                for i in range(1, src_ds.count + 1):
                    reproject(
                        source=rasterio.band(src_ds, i),
                        destination=rasterio.band(dst_ds, i),
                        src_transform=src_ds.transform,
                        src_crs=src_ds.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=resample_method,
                    )
        logger.info(f"Reprojected {product} → {dst.name}")
        return dst
    except Exception as e:
        logger.error(f"Reprojection failed for {product}: {e}")
        return None


def _stream_download(url: str, dest: Path) -> None:
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def validate_landfire_values(raster_path: Path, product: str) -> bool:
    """
    Check that raster values fall within expected LANDFIRE data dictionary ranges.
    Returns True if validation passes.
    """
    import numpy as np
    import rasterio

    expected_ranges: dict[str, tuple[float, float]] = {
        "FBFM40": (1, 99),
        "CC": (0, 100),
        "CH": (0, 3700),  # In units of 0.1m
        "CBD": (0, 60),   # kg/m³ × 100
        "CBH": (0, 3700),
        "EVT": (1, 9999),
    }

    if product not in expected_ranges:
        return True

    lo, hi = expected_ranges[product]
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1, masked=True)
            valid = data.compressed()
            if len(valid) == 0:
                logger.warning(f"{product}: No valid pixels found")
                return False
            vmin, vmax = float(valid.min()), float(valid.max())
            if vmin < lo or vmax > hi:
                logger.warning(
                    f"{product}: values [{vmin}, {vmax}] outside expected [{lo}, {hi}]"
                )
                return False
        logger.info(f"{product} validation passed: [{vmin:.1f}, {vmax:.1f}]")
        return True
    except Exception as e:
        logger.error(f"Validation error for {product}: {e}")
        return False


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    paths = get_paths(COLAB_MODE)
    cfg = load_config("data_sources.yaml")
    sa = get_study_area(cfg)
    bbox = (
        sa["bbox"]["xmin"], sa["bbox"]["ymin"],
        sa["bbox"]["xmax"], sa["bbox"]["ymax"],
    )

    products = cfg["landfire"]["products"]
    out_dir = paths["raw_landfire"]

    downloaded = download_landfire_products(
        products=products,
        bbox=bbox,
        out_dir=out_dir,
        output_crs=cfg["landfire"]["output_crs"],
        output_res_m=cfg["landfire"]["output_resolution_m"],
    )

    for product, path in downloaded.items():
        validate_landfire_values(path, product)
        logger.info(f"{product}: {path}")


if __name__ == "__main__":
    main()
