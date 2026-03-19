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

# Layer name mapping: short code → LANDFIRE versioned layer name
# Versions from UI: FBFM40/EVT are LF2020; canopy layers are LF2024
_LAYER_NAME_MAP = {
    "FBFM40": "LF2024_FBFM40",
    "CC":     "LF2024_CC",
    "CH":     "LF2024_CH",
    "CBD":    "LF2024_CBD",
    "CBH":    "LF2024_CBH",
    "EVT":    "LF2024_EVT",
}


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
    import os
    out_dir.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bbox
    # LANDFIRE AOI format: W S E N (space-separated)
    aoi_str = f"{xmin} {ymin} {xmax} {ymax}"
    results: dict[str, Path] = {}
    email = os.environ.get("LANDFIRE_EMAIL", "")

    # Check for any manually placed job-ID TIF (e.g. from LANDFIRE web UI download)
    _extract_jobid_tif(out_dir, products, output_crs, output_res_m)

    # Check if all products already downloaded
    missing = [p for p in products if not (out_dir / f"{p}_10m_utm17n.tif").exists()]
    for p in products:
        if p not in missing:
            path = out_dir / f"{p}_10m_utm17n.tif"
            logger.info(f"{p} already downloaded: {path.name}")
            results[p] = path

    api_failures = 0
    if missing:
        if not email:
            logger.warning("LANDFIRE_EMAIL not set — skipping API (email is required by LANDFIRE).")
            api_failures = len(missing)
        else:
            # Submit one job for all missing products
            logger.info(f"Requesting LANDFIRE products: {missing}")
            job_id = _submit_landfire_job(missing, aoi_str, email)
            if not job_id:
                api_failures = len(missing)
            else:
                raw_zip = _poll_and_download(job_id, out_dir, "LANDFIRE")
                if raw_zip:
                    extracted = _extract_landfire_zip(raw_zip, out_dir)
                    if extracted:
                        # ZIP contains one TIF per layer; reproject each
                        tifs = list(extracted.parent.rglob("*.tif")) + list(extracted.parent.rglob("*.img"))
                        for tif in tifs:
                            for p in missing:
                                if _LAYER_NAME_MAP.get(p, p).lower() in tif.stem.lower() or p.lower() in tif.stem.lower():
                                    final_path = out_dir / f"{p}_10m_utm17n.tif"
                                    reprojected = _reproject_raster(tif, final_path, output_crs, output_res_m, p)
                                    if reprojected:
                                        results[p] = reprojected
                else:
                    api_failures = len(missing)

    # If API failed for all missing products, fall back to synthetic rasters
    if api_failures >= len(missing) and len(results) < len(products):
        logger.warning(
            "LANDFIRE API unreachable for all products — using synthetic fallback "
            "(TU1 fuel model, representative of Duke East Campus open lawn/hardwood setting)."
        )
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", output_crs, always_xy=True)
        xmin_u, ymin_u = t.transform(xmin, ymin)
        xmax_u, ymax_u = t.transform(xmax, ymax)
        results = create_synthetic_landfire(out_dir, (xmin_u, ymin_u, xmax_u, ymax_u), output_res_m)

    return results


def _submit_landfire_job(products: list[str], aoi_str: str, email: str) -> str | None:
    """Submit a single async clip-and-ship job for all products to LANDFIRE."""
    layer_names = ";".join(_LAYER_NAME_MAP.get(p, p) for p in products)
    params = {
        "Layer_Names": layer_names,
        "Area_Of_Interest": aoi_str,
        "Output_Projection": "4326",   # Download in WGS84, reproject locally
        "Resample_Resolution": "30",
        "Email_Address": email,
        "f": "json",
    }
    try:
        resp = requests.post(_LF_SUBMIT, data=params, timeout=60)
        resp.raise_for_status()
        raw = resp.text.strip()
        if not raw:
            logger.warning("LANDFIRE API returned empty response — API may be down or email invalid.")
            return None
        data = resp.json()
        job_id = data.get("jobId")
        logger.info(f"Submitted LANDFIRE job {job_id} for layers: {layer_names}")
        return job_id
    except Exception as e:
        logger.error(f"LANDFIRE job submission error: {e}")
        return None


def _extract_jobid_tif(
    out_dir: Path,
    products: list[str],
    output_crs: str,
    output_res_m: int,
) -> None:
    """
    If a LANDFIRE job-ID TIF (e.g. j845f8fb...tif) exists in out_dir,
    split its bands into per-product files using the product order from
    the LANDFIRE layer name map.
    """
    import rasterio

    expected = {f"{p}_10m_utm17n.tif" for p in products}
    candidates = [
        f for f in out_dir.glob("*.tif")
        if f.name not in expected and not f.stem.startswith("chm")
    ]
    if not candidates:
        return

    src_path = candidates[0]
    try:
        with rasterio.open(src_path) as src:
            n_bands = src.count
            logger.info(f"Found job-ID TIF {src_path.name} with {n_bands} band(s) — splitting into products.")

            # LANDFIRE delivers bands in the order layers were requested
            band_to_product = {i + 1: p for i, p in enumerate(products[:n_bands])}

            for band_idx, product in band_to_product.items():
                final_path = out_dir / f"{product}_10m_utm17n.tif"
                if final_path.exists():
                    continue
                data = src.read(band_idx)
                profile = src.profile.copy()
                profile.update(count=1)
                # Write single-band TIF, then reproject
                tmp = out_dir / f"_tmp_{product}.tif"
                with rasterio.open(tmp, "w", **profile) as dst:
                    dst.write(data, 1)
                _reproject_raster(tmp, final_path, output_crs, output_res_m, product)
                tmp.unlink(missing_ok=True)
                logger.info(f"Extracted band {band_idx} → {final_path.name}")
    except Exception as e:
        logger.warning(f"Could not split {src_path.name}: {e}")


def create_synthetic_landfire(
    out_dir: Path,
    bbox_utm: tuple[float, float, float, float],
    resolution_m: int = 10,
) -> dict[str, Path]:
    """
    Generate synthetic LANDFIRE-equivalent rasters for the Randolph Hall PoC
    when the LANDFIRE REST API is unavailable.

    Values are representative of Duke East Campus (Randolph Hall area): open
    maintained lawns with scattered mature hardwoods — assigned TU1 (Light
    Timber/Grass/Shrub) fuel model, code 161 in Scott-Burgan 40.

    Parameters
    ----------
    out_dir:
        Output directory for synthetic rasters.
    bbox_utm:
        (xmin, ymin, xmax, ymax) in EPSG:32617.
    resolution_m:
        Raster resolution in meters.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    out_dir.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bbox_utm
    cols = max(1, int((xmax - xmin) / resolution_m))
    rows = max(1, int((ymax - ymin) / resolution_m))
    transform = from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    profile = {
        "driver": "GTiff", "dtype": "int16", "width": cols, "height": rows,
        "count": 1, "crs": "EPSG:32617", "transform": transform,
        "compress": "lzw", "nodata": -9999,
    }

    # Synthetic values representative of Duke East Campus (Randolph Hall area):
    # maintained lawns with scattered mature hardwoods, open residential setting
    synthetic: dict[str, tuple[int, str]] = {
        "FBFM40": (161, "int16"),   # TU1 — light timber/grass/shrub — open campus with scattered trees
        "CC":     (25,  "int16"),   # 25% canopy cover (more open than West Campus)
        "CH":     (120, "int16"),   # 12m canopy height (stored as 0.1m units → 120)
        "CBD":    (4,   "int16"),   # canopy bulk density (0.04 kg/m³ × 100)
        "CBH":    (15,  "int16"),   # 1.5m canopy base height (0.1m units)
        "EVT":    (7292, "int16"),  # Piedmont/Central Hardwood-Pine Forest EVT code
    }

    outputs: dict[str, Path] = {}
    for product, (value, dtype) in synthetic.items():
        path = out_dir / f"{product}_10m_utm17n.tif"
        if path.exists():
            outputs[product] = path
            continue
        p = profile.copy()
        p["dtype"] = dtype
        arr = np.full((rows, cols), value, dtype=dtype)
        with rasterio.open(path, "w", **p) as dst:
            dst.write(arr, 1)
        logger.info(f"Wrote synthetic {product} raster ({rows}x{cols}) → {path.name}")
        outputs[product] = path

    return outputs


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
        "FBFM40": (1, 299),  # Scott-Burgan codes: GR=101-107, SH=141-149, TU=161-165, TL=181-189, NB=191-199
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
