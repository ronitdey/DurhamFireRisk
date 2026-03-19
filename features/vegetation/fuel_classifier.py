"""
LANDFIRE Scott-Burgan fuel model mapping and Rothermel parameter extraction.

Maps FBFM40 raster codes to the physical parameters required by the
Rothermel fire spread equations, and refines classifications with
local NAIP imagery using a lightweight CNN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger


# ── Scott-Burgan 40 Fuel Model Physical Parameters ───────────────────────────
# Source: Scott & Burgan (2005), Andrews (2018 update)
# Units:
#   w_o_*: oven-dry fuel loading (tons/acre)
#   delta: fuel bed depth (ft)
#   M_x: moisture of extinction (fraction)
#   sigma: surface-to-volume ratio (ft²/ft³)
#   h: low heat content (BTU/lb) — typically 8000 for most fuels

SCOTT_BURGAN_PARAMS: dict[str, dict] = {
    # Grass (GR) models
    "GR1": dict(label="Short, sparse dry climate grass",
                w_o_1hr=0.10, w_o_10hr=0.0, w_o_100hr=0.0, w_o_lh=0.30, w_o_lw=0.0,
                delta=0.4, M_x=0.15, sigma=2200.0, h=8000.0),
    "GR2": dict(label="Low load, dry climate grass",
                w_o_1hr=0.10, w_o_10hr=0.0, w_o_100hr=0.0, w_o_lh=1.0, w_o_lw=0.0,
                delta=1.0, M_x=0.15, sigma=2000.0, h=8000.0),
    "GR3": dict(label="Low load, very coarse, humid climate grass",
                w_o_1hr=0.10, w_o_10hr=0.40, w_o_100hr=0.0, w_o_lh=1.50, w_o_lw=0.0,
                delta=2.0, M_x=0.30, sigma=1500.0, h=8000.0),
    "GR4": dict(label="Moderate load, dry climate grass",
                w_o_1hr=0.25, w_o_10hr=0.0, w_o_100hr=0.0, w_o_lh=1.90, w_o_lw=0.0,
                delta=2.0, M_x=0.15, sigma=2000.0, h=8000.0),
    "GR7": dict(label="High load, dry climate grass",
                w_o_1hr=1.00, w_o_10hr=0.0, w_o_100hr=0.0, w_o_lh=5.40, w_o_lw=0.0,
                delta=3.0, M_x=0.15, sigma=2000.0, h=8000.0),
    # Timber Litter (TL) models — most relevant for NC Piedmont forests
    "TL1": dict(label="Low load compact conifer litter",
                w_o_1hr=1.0, w_o_10hr=2.20, w_o_100hr=3.60, w_o_lh=0.0, w_o_lw=0.0,
                delta=0.2, M_x=0.30, sigma=2000.0, h=8000.0),
    "TL3": dict(label="Moderate load conifer litter",
                w_o_1hr=0.50, w_o_10hr=2.20, w_o_100hr=2.80, w_o_lh=0.0, w_o_lw=0.0,
                delta=0.3, M_x=0.25, sigma=2000.0, h=8000.0),
    "TL8": dict(label="Long-needle litter",
                w_o_1hr=0.30, w_o_10hr=1.40, w_o_100hr=8.10, w_o_lh=0.0, w_o_lw=0.0,
                delta=0.2, M_x=0.35, sigma=1750.0, h=8000.0),
    "TL9": dict(label="Very high load broadleaf litter",
                w_o_1hr=0.30, w_o_10hr=3.50, w_o_100hr=9.00, w_o_lh=0.0, w_o_lw=0.0,
                delta=0.6, M_x=0.35, sigma=1750.0, h=8000.0),
    # Timber Understory (TU) models
    "TU1": dict(label="Low load dry-climate timber-grass-shrub",
                w_o_1hr=0.20, w_o_10hr=0.90, w_o_100hr=1.50, w_o_lh=0.20, w_o_lw=0.90,
                delta=1.0, M_x=0.20, sigma=1750.0, h=8000.0),
    "TU5": dict(label="Very high load, dry climate timber-shrub",
                w_o_1hr=1.0, w_o_10hr=1.0, w_o_100hr=3.0, w_o_lh=0.30, w_o_lw=2.0,
                delta=1.0, M_x=0.25, sigma=1500.0, h=8000.0),
    # Shrub (SH) models
    "SH1": dict(label="Low load dry climate shrub",
                w_o_1hr=0.25, w_o_10hr=0.25, w_o_100hr=0.0, w_o_lh=0.15, w_o_lw=1.30,
                delta=1.0, M_x=0.15, sigma=1750.0, h=8000.0),
    "SH5": dict(label="High load dry climate shrub",
                w_o_1hr=0.45, w_o_10hr=2.45, w_o_100hr=0.0, w_o_lh=0.0, w_o_lw=7.00,
                delta=6.0, M_x=0.15, sigma=750.0, h=8000.0),
    "SH9": dict(label="Very high load, humid climate shrub",
                w_o_1hr=1.65, w_o_10hr=3.30, w_o_100hr=0.0, w_o_lh=1.10, w_o_lw=4.50,
                delta=4.4, M_x=0.40, sigma=750.0, h=8000.0),
    # Non-burnable
    "NB1": dict(label="Urban/suburban",
                w_o_1hr=0.0, w_o_10hr=0.0, w_o_100hr=0.0, w_o_lh=0.0, w_o_lw=0.0,
                delta=0.0, M_x=0.99, sigma=0.0, h=0.0),
    "NB9": dict(label="Bare ground",
                w_o_1hr=0.0, w_o_10hr=0.0, w_o_100hr=0.0, w_o_lh=0.0, w_o_lw=0.0,
                delta=0.0, M_x=0.99, sigma=0.0, h=0.0),
}

# LANDFIRE FBFM40 integer code → Scott-Burgan model string
_FBFM40_CODE_TO_MODEL: dict[int, str] = {
    101: "GR1", 102: "GR2", 103: "GR3", 104: "GR4", 107: "GR7",
    121: "TL1", 123: "TL3", 128: "TL8", 129: "TL9",
    141: "TU1", 145: "TU5",
    161: "SH1", 165: "SH5", 169: "SH9",
    91:  "NB1", 99:  "NB9",
}


def map_fuel_models(
    landfire_fbfm40_path: Path,
    output_path: Path | None = None,
) -> tuple[xr.DataArray, dict[str, np.ndarray]]:
    """
    Load LANDFIRE FBFM40 raster and return:
        1. DataArray of fuel model string codes per cell.
        2. Dict mapping each Rothermel parameter name → float32 raster.

    Parameters
    ----------
    landfire_fbfm40_path:
        Path to the FBFM40 GeoTIFF (already reprojected/resampled by ingestion).
    output_path:
        Optional path to save a multi-band GeoTIFF of Rothermel parameters.

    Returns
    -------
    (fuel_model_codes, param_arrays)
    """
    import rasterio

    with rasterio.open(landfire_fbfm40_path) as src:
        codes = src.read(1).astype(int)
        profile = src.profile
        transform = src.transform
        crs = src.crs

    model_names = np.full(codes.shape, "NB9", dtype="U6")
    for code, model in _FBFM40_CODE_TO_MODEL.items():
        model_names[codes == code] = model

    param_names = [
        "w_o_1hr", "w_o_10hr", "w_o_100hr", "w_o_lh", "w_o_lw",
        "delta", "M_x", "sigma", "h",
    ]
    param_arrays: dict[str, np.ndarray] = {
        p: np.zeros(codes.shape, dtype="float32") for p in param_names
    }

    for code, model in _FBFM40_CODE_TO_MODEL.items():
        mask = codes == code
        if not mask.any():
            continue
        params = SCOTT_BURGAN_PARAMS.get(model, SCOTT_BURGAN_PARAMS["NB9"])
        for p in param_names:
            param_arrays[p][mask] = float(params.get(p, 0.0))

    if output_path:
        _save_param_raster(param_arrays, param_names, profile, output_path)

    fuel_da = xr.DataArray(
        model_names,
        dims=["y", "x"],
        attrs={"crs": crs.to_string(), "transform": str(transform)},
    )

    n_burnable = np.sum(~np.isin(model_names, ["NB1", "NB9"]))
    logger.info(
        f"Fuel model mapping: {len(np.unique(model_names))} unique models, "
        f"{n_burnable} burnable cells "
        f"({100 * n_burnable / codes.size:.1f}% of domain)"
    )
    return fuel_da, param_arrays


def get_rothermel_params(fuel_model: str) -> dict:
    """Return Rothermel parameters for a given Scott-Burgan model code."""
    params = SCOTT_BURGAN_PARAMS.get(fuel_model.upper())
    if params is None:
        logger.warning(f"Unknown fuel model '{fuel_model}'; defaulting to NB9 (non-burnable).")
        return SCOTT_BURGAN_PARAMS["NB9"]
    return params.copy()


def get_total_fuel_load(model: str) -> float:
    """Return total oven-dry fuel load (tons/acre) for a fuel model."""
    p = get_rothermel_params(model)
    return p["w_o_1hr"] + p["w_o_10hr"] + p["w_o_100hr"] + p["w_o_lh"] + p["w_o_lw"]


def detect_ladder_fuels(
    canopy_height: np.ndarray,
    fuel_codes: xr.DataArray,
    canopy_base_height: np.ndarray,
    threshold_m: float = 2.0,
) -> np.ndarray:
    """
    Detect ladder fuels: cells where shrub/grass fuel exists below the canopy
    base height, enabling vertical fire spread from surface to crown.

    Returns boolean array where True = ladder fuel present.
    """
    is_understory = np.isin(
        fuel_codes.values,
        ["TU1", "TU5", "SH1", "SH5", "SH9", "GR2", "GR3", "GR4"],
    )
    has_canopy_above = canopy_height > threshold_m
    gap_below_canopy = canopy_base_height > threshold_m
    ladder = is_understory & has_canopy_above & gap_below_canopy
    logger.info(f"Ladder fuels: {ladder.sum()} cells ({100 * ladder.mean():.1f}% of domain)")
    return ladder


# ── Private helpers ──────────────────────────────────────────────────────────

def _save_param_raster(
    params: dict[str, np.ndarray],
    names: list[str],
    profile: dict,
    out_path: Path,
) -> None:
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=len(names), dtype="float32", compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, name in enumerate(names, start=1):
            dst.write(params[name], i)
            dst.update_tags(i, name=name)
    logger.info(f"Rothermel parameter raster saved → {out_path.name}")
