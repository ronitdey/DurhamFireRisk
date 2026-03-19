"""
Ember (firebrand) transport model.

Simulates the spotting process by which embers are launched from the
main fire front, transported by wind, and potentially ignite new fires
ahead of the perimeter.

References:
    Albini, F.A. (1979). Spot fire distance from burning trees: A predictive model.
    Cohen, J.D. & Stratton, R.D. (2008). Home destruction examination: Grass Valley Fire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class EmberParams:
    """Parameters governing ember generation and transport."""
    # Minimum flame length to produce lofted embers (ft)
    min_flame_length_ft: float = 4.0
    # Ember size (diameter, cm) — lognormal distribution
    diameter_mean_cm: float = 0.5
    diameter_std_cm: float = 0.3
    # Number of embers launched per linear foot of active fireline per minute
    embers_per_ft_per_min: float = 0.05
    # Landing distribution: lognormal parameters (distance in meters)
    lognormal_mu: float = 5.5
    lognormal_sigma: float = 1.2


def simulate_ember_transport(
    fireline_intensity_grid: np.ndarray,  # BTU/ft/s
    flame_length_grid: np.ndarray,        # ft
    wind_speed_mph: float,
    wind_dir_deg: float,
    resolution_m: float = 10.0,
    n_embers_per_cell: int = 5,
    ember_params: EmberParams | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate ember transport across the domain and return an ember landing
    density map (embers/m²).

    Parameters
    ----------
    fireline_intensity_grid:
        2D array of Byram's fireline intensity at each burning cell.
    flame_length_grid:
        2D array of flame lengths in feet.
    wind_speed_mph:
        Free-stream 20-ft wind speed (mph).
    wind_dir_deg:
        Wind direction degrees clockwise from north (from).
    resolution_m:
        Grid resolution in meters.
    n_embers_per_cell:
        Embers launched per burning cell per simulation step.
    ember_params:
        Ember generation and transport parameters.
    rng:
        NumPy random Generator (for reproducibility).

    Returns
    -------
    2D float32 array: ember landing count per cell.
    """
    params = ember_params or EmberParams()
    rng = rng or np.random.default_rng(seed=42)

    rows, cols = fireline_intensity_grid.shape
    landing_map = np.zeros((rows, cols), dtype="float32")

    # Identify source cells: active fire with sufficient flame length
    source_mask = flame_length_grid >= params.min_flame_length_ft
    source_rows, source_cols = np.where(source_mask)

    if len(source_rows) == 0:
        return landing_map

    wind_rad = math.radians(wind_dir_deg)
    wind_speed_mps = wind_speed_mph * 0.44704

    for sr, sc in zip(source_rows, source_cols):
        fl = float(flame_length_grid[sr, sc])
        intensity = float(fireline_intensity_grid[sr, sc])

        # Number of embers from this cell (proportional to intensity)
        n = max(1, int(n_embers_per_cell * intensity / 500))
        n = min(n, 50)  # Cap to avoid runaway compute

        for _ in range(n):
            # Ember diameter (lognormal)
            d = float(rng.lognormal(
                math.log(params.diameter_mean_cm),
                params.diameter_std_cm / params.diameter_mean_cm,
            ))
            d = max(0.05, min(d, 5.0))

            # Loft height proportional to flame length
            loft_height_m = fl * 0.3048 * rng.uniform(0.5, 1.5)

            # Transport distance (lognormal, scaled by wind)
            base_dist = rng.lognormal(params.lognormal_mu, params.lognormal_sigma)
            wind_factor = 1 + 0.5 * (wind_speed_mps / 10.0)
            transport_m = base_dist * wind_factor

            # Landing direction = downwind direction
            jitter_rad = rng.normal(0, math.radians(20))
            landing_rad = wind_rad + jitter_rad

            # Convert transport distance to grid offsets
            delta_y = -transport_m * math.cos(landing_rad) / resolution_m
            delta_x = transport_m * math.sin(landing_rad) / resolution_m

            lr = int(round(sr + delta_y))
            lc = int(round(sc + delta_x))

            if 0 <= lr < rows and 0 <= lc < cols:
                landing_map[lr, lc] += 1.0

    logger.debug(
        f"Ember transport: {len(source_rows)} source cells, "
        f"{landing_map.sum():.0f} total ember landings, "
        f"max {landing_map.max():.0f} at single cell."
    )
    return landing_map


def compute_ignition_probability(
    ember_density: np.ndarray,
    fuel_model_codes: np.ndarray,
    fuel_moisture_1hr: float = 0.06,
) -> np.ndarray:
    """
    Compute probability of spot ignition at each cell from ember landing.

    Ignition probability depends on:
    - Number of embers landing (Poisson arrival model)
    - Fuel type receptivity
    - 1-hr fuel moisture (primary determinant)

    Parameters
    ----------
    ember_density:
        Array of ember count per cell from simulate_ember_transport().
    fuel_model_codes:
        Integer array of LANDFIRE FBFM40 codes.
    fuel_moisture_1hr:
        1-hr fuel moisture fraction (lower = higher ignition probability).

    Returns
    -------
    float32 array of ignition probability [0, 1] per cell.
    """
    # Base ignition probability from fuel moisture
    # Moisture curve from Rothermel (1983) ignition probability model
    # p_ign = 0.000048 * exp(−4.3 * M_1hr) scaled to [0,1]
    M = fuel_moisture_1hr
    p_moisture = np.clip(math.exp(-4.3 * M) * 0.9, 0, 1)

    # Fuel receptivity multiplier by fuel type
    receptivity: dict[int, float] = {
        101: 0.8,   # GR1 - low load grass
        102: 1.0,   # GR2 - dry climate grass
        104: 1.2,   # GR4 - moderate load
        107: 1.3,   # GR7 - high load
        121: 0.6,   # TL1 - compact litter
        123: 0.7,   # TL3 - moderate litter
        128: 0.8,   # TL8 - long-needle
        129: 0.9,   # TL9 - broadleaf
        141: 0.7,   # TU1 - understory
        161: 1.0,   # SH1 - shrub
        91:  0.0,   # NB1 - urban (non-burnable)
        99:  0.0,   # NB9 - bare ground
    }
    fuel_mult = np.vectorize(lambda c: receptivity.get(int(c), 0.7))(fuel_model_codes)

    # Poisson probability of at least one igniting ember
    # P(at least 1 ignition) = 1 - exp(-lambda)
    # lambda = ember_density * p_moisture * fuel_mult
    lam = ember_density * p_moisture * fuel_mult
    p_ignition = (1 - np.exp(-lam)).astype("float32")

    logger.debug(
        f"Ignition probability: mean={p_ignition.mean():.4f}, "
        f"cells >0.5: {(p_ignition > 0.5).sum()}"
    )
    return p_ignition


def estimate_spotting_distance(
    flame_length_ft: float,
    wind_speed_mph: float,
    fuel_type: str = "TL8",
) -> dict[str, float]:
    """
    Estimate maximum and probable spotting distances using
    Albini (1979) empirical equations.

    Parameters
    ----------
    flame_length_ft:
        Flame length at the source (ft).
    wind_speed_mph:
        20-ft wind speed (mph).
    fuel_type:
        Source fuel model (affects firebrand generation rate).

    Returns
    -------
    dict with keys: max_distance_m, probable_distance_m, p90_distance_m.
    """
    # Convert to SI for Albini equations
    fl_m = flame_length_ft * 0.3048
    U_mps = wind_speed_mph * 0.44704

    # Albini (1979) crown fire spotting: L_max = 0.000162 × U²·² × L_flame^0.6
    # (empirical regression, valid for U 5-20 m/s, FL 2-20m)
    L_max_m = 0.000162 * (U_mps ** 2.2) * (fl_m ** 0.6) * 1000

    # Mean transport distance (lognormal median)
    mu, sigma = 5.5, 1.2
    probable_m = math.exp(mu)
    p90_m = math.exp(mu + 1.28 * sigma)

    # Scale by wind speed and flame length
    scale = (U_mps / 10) * (fl_m / 5)
    probable_m *= scale
    p90_m *= scale

    return {
        "max_distance_m": min(L_max_m, 5000),
        "probable_distance_m": probable_m,
        "p90_distance_m": p90_m,
    }
