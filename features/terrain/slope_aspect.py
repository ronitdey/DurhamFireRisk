"""
Terrain feature computation: slope, aspect, TPI, TRI, Heat Load Index,
Topographic Wetness Index, and upslope slope profiles.

All features are computed at 10m resolution on the DEM.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from loguru import logger


def compute_terrain_features(dem_path: Path, resolution_m: float = 10.0) -> xr.Dataset:
    """
    Compute a full suite of terrain features from a DEM GeoTIFF.

    Parameters
    ----------
    dem_path:
        Path to a GeoTIFF DEM (any CRS; internally reprojected to UTM).
    resolution_m:
        Target resolution in meters for all output rasters.

    Returns
    -------
    xr.Dataset with variables:
        slope_deg, aspect_deg, northness, eastness, tpi, tri,
        heat_load_index, twi, upslope_100m, upslope_300m, upslope_500m.
    """
    dem, meta = _load_and_resample_dem(dem_path, resolution_m)

    slope, aspect = _compute_slope_aspect(dem, resolution_m)
    tpi_300 = _compute_tpi(dem, neighborhood_m=300, res=resolution_m)
    tri = _compute_tri(dem)
    hli = _compute_heat_load_index(slope, aspect)
    flow_acc = _compute_flow_accumulation(dem)
    twi = _compute_twi(slope, flow_acc, resolution_m)

    northness = np.cos(np.radians(aspect))
    eastness = np.sin(np.radians(aspect))

    ds = xr.Dataset(
        {
            "slope_deg": (["y", "x"], slope.astype("float32")),
            "aspect_deg": (["y", "x"], aspect.astype("float32")),
            "northness": (["y", "x"], northness.astype("float32")),
            "eastness": (["y", "x"], eastness.astype("float32")),
            "tpi": (["y", "x"], tpi_300.astype("float32")),
            "tri": (["y", "x"], tri.astype("float32")),
            "heat_load_index": (["y", "x"], hli.astype("float32")),
            "twi": (["y", "x"], twi.astype("float32")),
        },
        attrs={
            "crs": str(meta["crs"]),
            "transform": str(meta["transform"]),
            "resolution_m": resolution_m,
        },
    )

    logger.info(
        f"Terrain features computed: slope [{slope.min():.1f}, {slope.max():.1f}]°, "
        f"mean HLI={hli.mean():.3f}"
    )
    return ds


def compute_upslope_profiles(
    dem: np.ndarray,
    building_centroids: list[tuple[float, float]],
    resolution_m: float,
    buffers_m: list[int] = (100, 300, 500),
) -> dict[tuple[float, float], dict[int, float]]:
    """
    For each building centroid (row, col in array coordinates), compute the
    mean slope in the uphill direction within each buffer distance.

    This captures the fire approach vector from above the structure—a key
    predictor that location-based risk scores entirely miss.

    Parameters
    ----------
    dem:
        2D numpy array of elevations.
    building_centroids:
        List of (row, col) array coordinates for each building.
    resolution_m:
        DEM resolution in meters (used to convert buffer_m to cells).
    buffers_m:
        Distances (m) at which to compute upslope mean slope.

    Returns
    -------
    dict mapping (row, col) → {buffer_m: mean_slope_deg}.
    """
    slope, aspect = _compute_slope_aspect(dem, resolution_m)
    results: dict[tuple, dict] = {}

    for rc in building_centroids:
        row, col = int(rc[0]), int(rc[1])
        uphill_dir = (aspect[row, col] + 180) % 360  # direction fire approaches FROM
        dr = -math.cos(math.radians(uphill_dir))      # row offset per step
        dc = math.sin(math.radians(uphill_dir))       # col offset per step

        profile: dict[int, float] = {}
        for buf in buffers_m:
            n_cells = int(buf / resolution_m)
            slopes_uphill: list[float] = []
            for step in range(1, n_cells + 1):
                r = int(round(row + dr * step))
                c = int(round(col + dc * step))
                if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]:
                    slopes_uphill.append(float(slope[r, c]))
            profile[buf] = float(np.mean(slopes_uphill)) if slopes_uphill else 0.0
        results[(row, col)] = profile

    return results


def classify_tpi(tpi: np.ndarray) -> np.ndarray:
    """
    Classify TPI into landform categories:
        'ridge' (+), 'upper_slope', 'mid_slope', 'valley' (-)
    Returns integer array: 3=ridge, 2=upper, 1=mid, 0=valley.
    """
    std = tpi.std()
    classes = np.where(tpi > std, 3,
               np.where(tpi > 0.5 * std, 2,
               np.where(tpi > -0.5 * std, 1, 0)))
    return classes.astype("uint8")


# ── Private helpers ──────────────────────────────────────────────────────────

def _load_and_resample_dem(dem_path: Path, resolution_m: float) -> tuple[np.ndarray, dict]:
    """Load DEM, reproject to EPSG:32617 (UTM Zone 17N), resample to resolution_m."""
    with rasterio.open(dem_path) as src:
        target_crs = "EPSG:32617"  # Always output in UTM Zone 17N
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=resolution_m
        )
        meta = src.meta.copy()
        meta.update(crs=target_crs, transform=transform, width=width, height=height)

        dem_arr = np.empty((height, width), dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
    dem_arr[dem_arr < -9000] = np.nan
    return dem_arr, meta


def _compute_slope_aspect(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Horn's method (3×3 neighborhood) for slope and aspect.
    Returns slope in degrees and aspect in degrees clockwise from north.
    """
    dz_dx = np.gradient(dem, res, axis=1)  # East–West
    dz_dy = np.gradient(dem, res, axis=0)  # North–South (y increases downward)
    dz_dy = -dz_dy  # Flip: positive = uphill toward north

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dz_dx, dz_dy)
    aspect_deg = (np.degrees(aspect_rad) + 360) % 360

    return slope_deg.astype("float32"), aspect_deg.astype("float32")


def _compute_tpi(dem: np.ndarray, neighborhood_m: int, res: float) -> np.ndarray:
    """
    Topographic Position Index = elevation - mean elevation in neighborhood.
    Positive = ridges; negative = valleys.
    """
    from scipy.ndimage import uniform_filter
    radius_cells = max(1, int(neighborhood_m / res))
    size = 2 * radius_cells + 1
    local_mean = uniform_filter(dem, size=size, mode="nearest")
    return (dem - local_mean).astype("float32")


def _compute_tri(dem: np.ndarray) -> np.ndarray:
    """
    Terrain Ruggedness Index (Riley et al. 1999):
    TRI = sqrt(sum of squared differences to 8 neighbors).
    """
    from scipy.ndimage import generic_filter

    def _tri_kernel(window: np.ndarray) -> float:
        center = window[4]
        return float(np.sqrt(np.sum((window - center) ** 2)))

    tri = generic_filter(dem.astype(float), _tri_kernel, size=3, mode="nearest")
    return tri.astype("float32")


def _compute_heat_load_index(slope_deg: np.ndarray, aspect_deg: np.ndarray) -> np.ndarray:
    """
    Heat Load Index (McCune & Keon 2002):
    HL = (1 - cos(π/180 × (aspect - 225))) / 2 × sin(slope × π/180)

    Ranges 0–1; higher = hotter, drier, more fire-prone.
    South/southwest-facing slopes with steep grades score highest.
    """
    cos_term = (1 - np.cos(np.radians(aspect_deg - 45))) / 2
    sin_slope = np.sin(np.radians(slope_deg))
    hli = cos_term * sin_slope
    return np.clip(hli, 0, 1).astype("float32")


def _compute_flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """
    Simplified D8 flow accumulation (steepest descent to one neighbor).
    Returns upstream contributing area in number of cells.
    """
    rows, cols = dem.shape
    filled = _fill_depressions(dem)

    # D8 flow direction: map each cell to its lowest neighbor
    dr = [-1, -1, -1,  0,  0,  1,  1,  1]
    dc = [-1,  0,  1, -1,  1, -1,  0,  1]

    flow_dir = np.full_like(filled, -1, dtype=int)
    for r in range(rows):
        for c in range(cols):
            min_dz = 0.0
            for k in range(8):
                nr, nc = r + dr[k], c + dc[k]
                if 0 <= nr < rows and 0 <= nc < cols:
                    dz = filled[r, c] - filled[nr, nc]
                    if dz > min_dz:
                        min_dz = dz
                        flow_dir[r, c] = k

    # Accumulate
    acc = np.ones((rows, cols), dtype="float32")
    order = np.argsort(filled.ravel())[::-1]
    for idx in order:
        r, c = divmod(int(idx), cols)
        k = flow_dir[r, c]
        if k >= 0:
            nr, nc = r + dr[k], c + dc[k]
            if 0 <= nr < rows and 0 <= nc < cols:
                acc[nr, nc] += acc[r, c]

    return acc


def _fill_depressions(dem: np.ndarray) -> np.ndarray:
    """Simple iterative depression filling (Planchon-Darboux approximation)."""
    filled = np.where(np.isnan(dem), -9999.0, dem.copy().astype(float))
    edge_mask = np.zeros_like(filled, dtype=bool)
    edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = True
    filled[~edge_mask] = np.inf

    changed = True
    itr = 0
    while changed and itr < 100:
        changed = False
        itr += 1
        for r in range(1, filled.shape[0] - 1):
            for c in range(1, filled.shape[1] - 1):
                if dem[r, c] == filled[r, c]:
                    continue
                neighbors = filled[r-1:r+2, c-1:c+2]
                min_nbr = neighbors.min()
                new_val = max(dem[r, c], min_nbr + 1e-4)
                if new_val < filled[r, c]:
                    filled[r, c] = new_val
                    changed = True
    return filled


def _compute_twi(slope_deg: np.ndarray, flow_acc: np.ndarray, res: float) -> np.ndarray:
    """
    Topographic Wetness Index: TWI = ln(a / tan(β))
    where a = upstream area (m²), β = local slope (radians).
    High TWI = higher soil moisture = potential flood/wet area.
    """
    a = flow_acc * (res ** 2)  # Convert cells to m²
    beta = np.radians(np.maximum(slope_deg, 0.001))  # Avoid log(inf) on flat terrain
    twi = np.log(a / np.tan(beta))
    return np.clip(twi, 0, 30).astype("float32")
