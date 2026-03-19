"""
Wind field interpolation across complex terrain.

Pre-computes terrain-following wind fields for 8 directions × 3 speed bins
= 24 wind field rasters, cached to disk. Fire spread simulations look up
pre-computed wind fields rather than solving fluid dynamics at query time.

Based on:
    Forthofer et al. (2014) — WindNinja terrain wind model
    Ryan (1977) — terrain wind multipliers
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger


# Terrain wind multipliers by topographic class
TERRAIN_MULTIPLIERS: dict[str, float] = {
    "ridge": 1.4,
    "upper_slope": 1.2,
    "mid_slope": 1.0,
    "valley": 0.6,
}

# 8 primary wind directions (from, degrees clockwise from N)
WIND_DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]

# 3 representative speed bins (mph): low, moderate, extreme
WIND_SPEED_BINS = [10, 20, 35]


def precompute_wind_fields(
    slope_grid: np.ndarray,
    aspect_grid: np.ndarray,
    tpi_grid: np.ndarray,
    resolution_m: float = 10.0,
    cache_dir: Path | None = None,
) -> dict[tuple[float, float], np.ndarray]:
    """
    Pre-compute terrain-adjusted wind fields for all direction/speed combinations.

    Parameters
    ----------
    slope_grid, aspect_grid, tpi_grid:
        2D arrays of terrain features (degrees).
    resolution_m:
        Grid resolution in meters.
    cache_dir:
        If provided, save each wind field as a .npy file for reuse.

    Returns
    -------
    dict mapping (wind_dir_deg, wind_speed_mph) → 2D wind speed array (mph).
    """
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    tpi_std = tpi_grid.std()
    tpi_class = _classify_tpi(tpi_grid, tpi_std)

    wind_fields: dict[tuple, np.ndarray] = {}

    for wind_dir in WIND_DIRECTIONS:
        for wind_speed in WIND_SPEED_BINS:
            cache_key = (float(wind_dir), float(wind_speed))
            cache_file = cache_dir / f"wind_{wind_dir:03d}_{wind_speed:02d}mph.npy" if cache_dir else None

            if cache_file and cache_file.exists():
                field = np.load(cache_file)
                wind_fields[cache_key] = field
                continue

            field = _compute_terrain_wind(
                free_stream_speed=float(wind_speed),
                wind_dir_deg=float(wind_dir),
                slope_grid=slope_grid,
                aspect_grid=aspect_grid,
                tpi_class=tpi_class,
            )

            if cache_file:
                np.save(cache_file, field)
            wind_fields[cache_key] = field

    logger.info(
        f"Pre-computed {len(wind_fields)} wind fields "
        f"({len(WIND_DIRECTIONS)} directions × {len(WIND_SPEED_BINS)} speeds)"
    )
    return wind_fields


def get_wind_field(
    wind_dir_deg: float,
    wind_speed_mph: float,
    precomputed: dict[tuple, np.ndarray],
) -> np.ndarray:
    """
    Look up the nearest pre-computed wind field for given conditions.

    Snaps to the nearest direction (45° bins) and speed bin.

    Parameters
    ----------
    wind_dir_deg:
        Free-stream wind direction (degrees from N).
    wind_speed_mph:
        Free-stream 20-ft wind speed (mph).
    precomputed:
        Dict from precompute_wind_fields().

    Returns
    -------
    2D array of terrain-adjusted wind speeds (mph).
    """
    # Snap to nearest direction bin
    nearest_dir = float(min(WIND_DIRECTIONS, key=lambda d: min(abs(wind_dir_deg - d), 360 - abs(wind_dir_deg - d))))
    # Snap to nearest speed bin
    nearest_speed = float(min(WIND_SPEED_BINS, key=lambda s: abs(wind_speed_mph - s)))

    key = (nearest_dir, nearest_speed)
    if key in precomputed:
        # Scale by ratio of actual to bin speed
        scale = wind_speed_mph / max(nearest_speed, 0.1)
        return precomputed[key] * scale

    # Fallback: uniform field at free-stream speed
    if precomputed:
        shape = next(iter(precomputed.values())).shape
    else:
        return np.full((100, 100), wind_speed_mph, dtype="float32")
    return np.full(shape, wind_speed_mph, dtype="float32")


def _compute_terrain_wind(
    free_stream_speed: float,
    wind_dir_deg: float,
    slope_grid: np.ndarray,
    aspect_grid: np.ndarray,
    tpi_class: np.ndarray,
) -> np.ndarray:
    """
    Apply terrain wind multipliers and channeling to free-stream wind.

    Rules:
    1. Base multiplier from topographic class (ridge 1.4×, valley 0.6×)
    2. Wind-slope interaction: slope facing into wind accelerates flow
    3. Simple terrain channeling approximation (direction not modified in this version)
    """
    rows, cols = slope_grid.shape
    field = np.full((rows, cols), free_stream_speed, dtype="float32")

    # Topographic class multiplier
    class_mult = np.vectorize(lambda c: TERRAIN_MULTIPLIERS.get(_tpi_code_to_class(int(c)), 1.0))(tpi_class)
    field *= class_mult.astype("float32")

    # Wind-aspect interaction: wind on slopes facing the wind → speed-up
    wind_rad = np.radians(wind_dir_deg)
    aspect_rad = np.radians(aspect_grid)
    cos_diff = np.cos(aspect_rad - wind_rad)  # +1 = facing into wind, -1 = lee side
    slope_frac = np.sin(np.radians(slope_grid))

    # Speed-up on windward slopes (cos_diff > 0), slow-down on leeward
    terrain_boost = 1 + 0.3 * cos_diff * slope_frac
    terrain_boost = np.clip(terrain_boost, 0.5, 2.0)
    field *= terrain_boost.astype("float32")

    return field


def _classify_tpi(tpi: np.ndarray, tpi_std: float) -> np.ndarray:
    """Classify TPI into integer codes: 3=ridge, 2=upper, 1=mid, 0=valley."""
    classes = np.where(tpi > tpi_std, 3,
               np.where(tpi > 0.5 * tpi_std, 2,
               np.where(tpi > -0.5 * tpi_std, 1, 0)))
    return classes.astype("uint8")


def _tpi_code_to_class(code: int) -> str:
    return {3: "ridge", 2: "upper_slope", 1: "mid_slope", 0: "valley"}.get(code, "mid_slope")
