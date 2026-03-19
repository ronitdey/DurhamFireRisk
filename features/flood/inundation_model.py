"""
Simple 2D inundation depth estimation using the HAND method.

For each return period, estimates flood inundation extent and depth
by comparing water surface elevation (from Manning's equation applied
to gauged streamflow data) against cell elevations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from loguru import logger


def estimate_flood_inundation(
    dem: xr.DataArray,
    flow_accumulation: xr.DataArray,
    hand: xr.DataArray,
    return_periods: list[int] = (10, 25, 100),
    gage_peak_flows: Optional[dict[int, float]] = None,
    channel_width_m: float = 20.0,
    manning_n: float = 0.035,
) -> dict[int, xr.DataArray]:
    """
    Estimate flood inundation depth for each return period.

    Method: HAND (Height Above Nearest Drainage).
    Cells with HAND < water_surface_rise_m are inundated.
    Inundation depth = water_surface_rise_m - HAND.

    Parameters
    ----------
    dem:
        DEM DataArray (elevations in meters).
    flow_accumulation:
        Upstream contributing area DataArray (cells).
    hand:
        HAND DataArray from flow_accumulation.py (m above nearest drainage).
    return_periods:
        List of return period years (e.g., [10, 25, 100]).
    gage_peak_flows:
        dict mapping return period (years) → peak discharge (m³/s).
        If None, estimated from regional regression equations.
    channel_width_m:
        Approximate bankfull channel width (m) for Manning's calculation.
    manning_n:
        Manning's roughness coefficient for the channel.

    Returns
    -------
    dict mapping return period → inundation depth DataArray (m, NaN = dry).
    """
    peak_flows = gage_peak_flows or _estimate_peak_flows_nc(flow_accumulation, return_periods)

    results: dict[int, xr.DataArray] = {}
    for rp in return_periods:
        Q = peak_flows.get(rp, 0.0)
        if Q <= 0:
            results[rp] = xr.zeros_like(hand) * np.nan
            continue

        water_surface_rise_m = _manning_depth(Q, channel_width_m, manning_n)
        hand_arr = hand.values.astype("float64")
        depth = water_surface_rise_m - hand_arr
        depth[depth <= 0] = np.nan  # Dry cells
        inund = xr.DataArray(
            depth.astype("float32"),
            dims=hand.dims,
            coords=hand.coords,
            attrs={"return_period_years": rp, "peak_flow_cms": Q,
                   "water_surface_rise_m": water_surface_rise_m},
        )
        results[rp] = inund
        n_flooded = int((~np.isnan(depth)).sum())
        area_ha = n_flooded * (10 ** 2) / 10000  # 10m cells
        logger.info(
            f"{rp}-yr flood: Q={Q:.1f} m³/s, WSE rise={water_surface_rise_m:.2f}m, "
            f"{n_flooded} cells flooded ({area_ha:.1f} ha)"
        )

    return results


def compute_parcel_flood_risk(
    inundation_maps: dict[int, xr.DataArray],
    parcels,
    resolution_m: float = 10.0,
) -> dict:
    """
    Compute parcel-level flood risk metrics from inundation maps.

    For each parcel:
        - Maximum inundation depth by return period
        - Fraction of parcel area inundated
        - Simple flood risk score (0-100)

    Returns dict mapping parcel_id → {10yr_depth, 25yr_depth, 100yr_depth, flood_risk_score}.
    """
    results: dict = {}

    for _, row in parcels.iterrows():
        parcel_id = str(row.get("parcel_id", row.name))
        geom = row.get("geometry")
        if geom is None:
            continue

        cx, cy = geom.centroid.x, geom.centroid.y
        depths: dict[int, float] = {}

        for rp, da in inundation_maps.items():
            try:
                val = float(da.sel(x=cx, y=cy, method="nearest").values)
                depths[rp] = 0.0 if np.isnan(val) else val
            except Exception:
                depths[rp] = 0.0

        score = _flood_risk_score(depths)
        results[parcel_id] = {
            "flood_depth_10yr_m": depths.get(10, 0.0),
            "flood_depth_25yr_m": depths.get(25, 0.0),
            "flood_depth_100yr_m": depths.get(100, 0.0),
            "flood_risk_score": score,
        }

    return results


# ── Private helpers ──────────────────────────────────────────────────────────

def _manning_depth(Q: float, width: float, n: float, slope: float = 0.005) -> float:
    """
    Estimate water surface rise above bankfull using Manning's equation.
    Assumes wide rectangular channel approximation.

    Q = (1/n) × A × R^(2/3) × S^(1/2)
    For wide channel: R ≈ depth → depth = (Q × n / (width × sqrt(slope)))^(3/5)
    """
    if Q <= 0 or width <= 0:
        return 0.0
    depth = (Q * n / (width * (slope ** 0.5))) ** 0.6
    return float(depth)


def _estimate_peak_flows_nc(
    flow_accumulation: xr.DataArray,
    return_periods: list[int],
) -> dict[int, float]:
    """
    Estimate peak discharge for return periods using USGS regional
    regression equations for the NC Piedmont.

    Reference: Feaster et al. (2014) USGS SIR 2014-5090
    Q_T = a × A^b (A = drainage area in mi², Q in ft³/s)
    """
    cells = float(flow_accumulation.max().values)
    drainage_area_mi2 = cells * (10 ** 2) / (1609.34 ** 2)  # 10m cells → mi²

    nc_piedmont_params: dict[int, tuple] = {
        2:   (174, 0.68),
        5:   (303, 0.70),
        10:  (415, 0.71),
        25:  (586, 0.73),
        50:  (727, 0.74),
        100: (890, 0.75),
        500: (1330, 0.77),
    }

    peak_flows: dict[int, float] = {}
    for rp in return_periods:
        a, b = nc_piedmont_params.get(rp, (500, 0.72))
        Q_cfs = a * (drainage_area_mi2 ** b)
        Q_cms = Q_cfs * 0.0283168
        peak_flows[rp] = Q_cms

    return peak_flows


def _flood_risk_score(depths: dict[int, float]) -> float:
    """
    Convert flood inundation depths by return period into a 0-100 flood risk score.

    Scoring logic:
        - In 100-yr floodplain:                  base +30
        - Depth > 1m in 100-yr event:            +20
        - In 25-yr floodplain:                   +25
        - In 10-yr floodplain:                   +25
    """
    score = 0.0

    d10 = depths.get(10, 0.0)
    d25 = depths.get(25, 0.0)
    d100 = depths.get(100, 0.0)

    if d100 > 0:
        score += 30
        score += min(d100 * 10, 20)  # Up to +20 for depth
    if d25 > 0:
        score += 25
    if d10 > 0:
        score += 25

    return float(min(score, 100))
