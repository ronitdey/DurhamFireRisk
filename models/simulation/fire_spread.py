"""
Rothermel (1972) fire spread model implementation.

Full implementation of the Rothermel surface fire equations used in
FARSITE, FlamMap, and BehavePlus. Includes Huygens wavelet propagation
across a gridded landscape for time-of-arrival simulation.

Reference:
    Rothermel, R.C. (1972). A Mathematical Model for Predicting Fire Spread
    in Wildland Fuels. USDA Forest Service Research Paper INT-115.

    Andrews, P.L. (2018). The Rothermel Surface Fire Spread Model and
    Associated Developments: A Comprehensive Explanation.
    Gen. Tech. Rep. RMRS-GTR-371.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr
from loguru import logger


@dataclass
class FuelParams:
    """Physical parameters for a single Scott-Burgan fuel model."""
    model: str
    w_o_1hr: float   # 1-hr fuel load (tons/acre)
    w_o_10hr: float  # 10-hr fuel load
    w_o_100hr: float # 100-hr fuel load
    w_o_lh: float    # Live herbaceous load
    w_o_lw: float    # Live woody load
    delta: float     # Fuel bed depth (ft)
    M_x: float       # Moisture of extinction (fraction)
    sigma: float     # Surface-to-volume ratio (ft²/ft³)
    h: float = 8000.0  # Low heat content (BTU/lb)
    S_T: float = 0.0555  # Total mineral content
    S_e: float = 0.010   # Effective mineral content
    rho_p: float = 32.0  # Oven-dry particle density (lb/ft³)


@dataclass
class WeatherParams:
    """Dynamic fire-weather inputs for a simulation timestep."""
    M_1hr: float    # 1-hr fuel moisture (fraction)
    M_10hr: float   # 10-hr fuel moisture
    M_100hr: float  # 100-hr fuel moisture
    M_lh: float     # Live herbaceous moisture
    M_lw: float     # Live woody moisture
    wind_speed_mph: float   # 20-ft wind speed (mph)
    wind_dir_deg: float     # Wind direction (degrees clockwise from north)
    slope_deg: float = 0.0  # Local terrain slope (degrees)
    aspect_deg: float = 0.0 # Local terrain aspect


@dataclass
class SpreadResult:
    """Output from a single-cell Rothermel computation."""
    R: float          # Rate of spread (ft/min)
    I_B: float        # Byram's fireline intensity (BTU/ft/s)
    FL: float         # Flame length (ft)
    HPA: float        # Heat per unit area (BTU/ft²)
    I_R: float        # Reaction intensity (BTU/ft²/min)
    phi_w: float      # Wind coefficient
    phi_s: float      # Slope coefficient


class RothermelFireSpread:
    """
    Implements the Rothermel (1972) surface fire spread equations.

    Typical usage:
        model = RothermelFireSpread()
        result = model.compute(fuel_params, weather_params)
    """

    def compute(self, fuel: FuelParams, weather: WeatherParams) -> SpreadResult:
        """
        Compute rate of spread and fire behavior outputs for a single
        fuel/weather combination.

        Returns SpreadResult with R (ft/min), I_B (BTU/ft/s), FL (ft).
        """
        # Convert fuel loads from tons/acre to lb/ft²
        w_1  = fuel.w_o_1hr   * 2000 / 43560
        w_10 = fuel.w_o_10hr  * 2000 / 43560
        w_100 = fuel.w_o_100hr * 2000 / 43560
        w_lh = fuel.w_o_lh   * 2000 / 43560
        w_lw = fuel.w_o_lw   * 2000 / 43560

        # Non-burnable check
        if fuel.delta <= 0 or fuel.sigma <= 0 or (w_1 + w_10 + w_100 + w_lh + w_lw) < 1e-6:
            return SpreadResult(R=0, I_B=0, FL=0, HPA=0, I_R=0, phi_w=0, phi_s=0)

        # ── Step 1: Reaction Intensity ────────────────────────────────────
        I_R = self.compute_reaction_intensity(fuel, weather, w_1, w_10, w_100, w_lh, w_lw)

        # ── Step 2: Propagating flux ratio ────────────────────────────────
        # Combined surface-to-volume ratio (weighted by load)
        total_dead = w_1 + w_10 + w_100
        sigma_wt = sigma_weighted = fuel.sigma  # simplified; full: Σ(w_i * sigma_i) / Σw_i

        # Bulk density
        w_total = total_dead + w_lh + w_lw
        rho_b = w_total / fuel.delta  # lb/ft³

        # Packing ratio
        beta = rho_b / fuel.rho_p
        beta_op = 3.348 * sigma_wt ** -0.8189  # Optimum packing ratio

        # Propagating flux ratio xi
        xi = math.exp((0.792 + 0.681 * sigma_wt ** 0.5) * (beta + 0.1)) / (192 + 0.2595 * sigma_wt)

        # ── Step 3: Wind and slope coefficients ───────────────────────────
        phi_w = self._wind_coefficient(weather, sigma_wt, beta, beta_op)
        phi_s = self._slope_coefficient(weather.slope_deg, beta)

        # Effective heating number
        epsilon = math.exp(-138.0 / sigma_wt)

        # Heat of pre-ignition
        Q_ig = 250 + 1116 * weather.M_1hr

        # ── Step 4: Rate of spread (Rothermel eq. 52) ────────────────────
        denominator = rho_b * epsilon * Q_ig
        if denominator < 1e-10:
            return SpreadResult(R=0, I_B=0, FL=0, HPA=0, I_R=I_R, phi_w=phi_w, phi_s=phi_s)

        R = (I_R * xi * (1 + phi_w + phi_s)) / denominator  # ft/min

        # ── Step 5: Derived fire behavior outputs ────────────────────────
        # Heat per unit area (BTU/ft²)
        HPA = I_R * fuel.delta / R if R > 0 else 0

        # Byram's fireline intensity (BTU/ft/s)
        I_B = (fuel.h * w_total * R) / 60.0

        # Flame length (Byram 1959): FL = 0.45 * I_B^0.46
        FL = 0.45 * (I_B ** 0.46) if I_B > 0 else 0

        return SpreadResult(R=R, I_B=I_B, FL=FL, HPA=HPA, I_R=I_R, phi_w=phi_w, phi_s=phi_s)

    def compute_reaction_intensity(
        self,
        fuel: FuelParams,
        weather: WeatherParams,
        w_1: float, w_10: float, w_100: float, w_lh: float, w_lw: float,
    ) -> float:
        """
        Compute reaction intensity I'_R (BTU/ft²/min).

        I'_R = Γ' × w_n × h × η_M × η_S

        where:
            Γ'  = optimum reaction velocity (1/min)
            w_n = net fuel load (lb/ft²) after mineral exclusion
            η_M = moisture damping coefficient
            η_S = mineral damping coefficient
        """
        w_total_dead = w_1 + w_10 + w_100
        w_total = w_total_dead + w_lh + w_lw

        if w_total < 1e-8:
            return 0.0

        # Net fuel load (after mineral exclusion)
        w_n = w_total * (1 - fuel.S_T)

        # Optimum reaction velocity Γ'
        sigma = fuel.sigma
        rho_b = w_total / max(fuel.delta, 0.01)
        beta = rho_b / fuel.rho_p
        beta_op = 3.348 * sigma ** -0.8189
        A = 133.0 * sigma ** -0.7913
        Gamma_max = sigma ** 1.5 / (495 + 0.0594 * sigma ** 1.5)
        Gamma_prime = Gamma_max * (beta / beta_op) ** A * math.exp(A * (1 - beta / beta_op))

        # Weighted moisture of live fuel
        M_live = (w_lh * weather.M_lh + w_lw * weather.M_lw) / max(w_lh + w_lw, 1e-8)

        # Effective moisture for damping (weighted dead)
        M_eff_dead = (
            (w_1 * weather.M_1hr + w_10 * weather.M_10hr + w_100 * weather.M_100hr)
            / max(w_total_dead, 1e-8)
        )
        M_eff = (
            (w_total_dead * M_eff_dead + (w_lh + w_lw) * M_live)
            / max(w_total, 1e-8)
        )

        # Moisture damping coefficient η_M
        r_M = min(M_eff / fuel.M_x, 1.0)
        eta_M = max(0, 1 - 2.59 * r_M + 5.11 * r_M ** 2 - 3.52 * r_M ** 3)

        # Mineral damping coefficient η_S
        eta_S = max(0.174 * fuel.S_e ** -0.19, 0.0)

        I_R = Gamma_prime * w_n * fuel.h * eta_M * eta_S
        return max(0.0, I_R)

    def _wind_coefficient(
        self, weather: WeatherParams, sigma: float, beta: float, beta_op: float
    ) -> float:
        """
        Wind coefficient φ_w.
        U must be midflame wind speed (ft/min); convert from 20-ft mph.
        """
        # Midflame wind speed: apply wind adjustment factor 0.4 (standard)
        U = weather.wind_speed_mph * 88 * 0.4  # mph → ft/min, then WAF
        U = max(U, 0.0)

        C = 7.47 * math.exp(-0.133 * sigma ** 0.55)
        B = 0.02526 * sigma ** 0.54
        E = 0.715 * math.exp(-3.59e-4 * sigma)
        phi_w = C * U ** B * (beta / beta_op) ** (-E)
        return max(0.0, phi_w)

    def _slope_coefficient(self, slope_deg: float, beta: float) -> float:
        """
        Slope coefficient φ_s = 5.275 × β^(-0.3) × tan²(slope).
        Slope must be in degrees.
        """
        tan_slope = math.tan(math.radians(max(0, slope_deg)))
        phi_s = 5.275 * (beta ** -0.3) * (tan_slope ** 2)
        return max(0.0, phi_s)


class FireSpreadSimulator:
    """
    Huygens wavelet propagation across a gridded landscape.

    Advances the fire perimeter using elliptical wavelet expansion
    driven by the Rothermel spread equations at each cell.
    """

    def __init__(
        self,
        fuel_params_grid: dict[str, np.ndarray],
        fuel_model_codes: np.ndarray,
        slope_grid: np.ndarray,
        aspect_grid: np.ndarray,
        resolution_m: float = 10.0,
        wind_speed_mph: float = 20.0,
        wind_dir_deg: float = 225.0,
        fuel_moisture: dict | None = None,
    ):
        self.fuel_params = fuel_params_grid
        self.fuel_codes = fuel_model_codes
        self.slope = slope_grid
        self.aspect = aspect_grid
        self.res = resolution_m
        self.wind_speed = wind_speed_mph
        self.wind_dir = wind_dir_deg
        self.fuel_moisture = fuel_moisture or {
            "M_1hr": 0.06, "M_10hr": 0.08, "M_100hr": 0.10,
            "M_lh": 0.80, "M_lw": 1.00,
        }
        self._rothermel = RothermelFireSpread()

    def simulate_spread(
        self,
        ignition_row: int,
        ignition_col: int,
        max_time_minutes: int = 60,
    ) -> xr.Dataset:
        """
        Run a Huygens wavelet fire spread simulation.

        Parameters
        ----------
        ignition_row, ignition_col:
            Array coordinates of the ignition point.
        max_time_minutes:
            Simulation duration in minutes.

        Returns
        -------
        xr.Dataset with:
            time_of_arrival: Minutes from ignition to cell burning (NaN=not reached)
            fireline_intensity: Byram's intensity at arrival (BTU/ft/s)
            flame_length: Flame length at arrival (ft)
        """
        rows, cols = self.slope.shape
        toa = np.full((rows, cols), np.nan, dtype="float32")
        intensity = np.full((rows, cols), 0.0, dtype="float32")
        flame_len = np.full((rows, cols), 0.0, dtype="float32")

        toa[ignition_row, ignition_col] = 0.0
        active: list[tuple[int, int]] = [(ignition_row, ignition_col)]

        # 8-neighbor offsets and distances
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        diag_dist = self.res * math.sqrt(2)
        neighbor_dists = [
            (diag_dist if abs(dr) + abs(dc) == 2 else self.res)
            for dr, dc in neighbors
        ]

        t = 0.0
        while active and t < max_time_minutes:
            t += 1.0
            new_active: list[tuple[int, int]] = []

            for (r, c) in active:
                fuel, weather = self._get_cell_params(r, c)
                result = self._rothermel.compute(fuel, weather)

                if result.R <= 0:
                    continue

                # Spread to neighbors
                for (dr, dc), dist in zip(neighbors, neighbor_dists):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if not np.isnan(toa[nr, nc]):
                        continue

                    # Time to reach neighbor = distance / rate of spread
                    r_ft_per_min = result.R
                    r_m_per_min = r_ft_per_min * 0.3048

                    # Apply wind direction: spread faster in downwind direction
                    bearing = math.degrees(math.atan2(dc, -dr)) % 360
                    angle_diff = abs(bearing - self.wind_dir)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    wind_factor = math.cos(math.radians(angle_diff))
                    effective_rate = r_m_per_min * max(0.2, 1 + 0.5 * wind_factor)

                    travel_time = dist / max(effective_rate, 0.001)
                    arrival = toa[r, c] + travel_time

                    if np.isnan(toa[nr, nc]) or arrival < toa[nr, nc]:
                        toa[nr, nc] = arrival
                        intensity[nr, nc] = result.I_B
                        flame_len[nr, nc] = result.FL
                        new_active.append((nr, nc))

            active = [pt for pt in new_active if (toa[pt[0], pt[1]] or 9999) <= t + 5]

        n_burned = (~np.isnan(toa)).sum()
        area_ha = n_burned * (self.res ** 2) / 10000
        logger.info(
            f"Simulation complete: {n_burned} cells burned ({area_ha:.1f} ha) "
            f"in {max_time_minutes} minutes."
        )

        return xr.Dataset({
            "time_of_arrival": (["y", "x"], toa),
            "fireline_intensity": (["y", "x"], intensity),
            "flame_length": (["y", "x"], flame_len),
        })

    def run_monte_carlo(
        self,
        ignition_row: int,
        ignition_col: int,
        n_simulations: int = 100,
        max_time_minutes: int = 60,
        wind_speed_range: tuple[float, float] = (10.0, 35.0),
        moisture_range: tuple[float, float] = (0.04, 0.12),
    ) -> xr.Dataset:
        """
        Run N stochastic simulations with randomized wind speed and fuel moisture.
        Returns probability-of-exposure raster (fraction of runs reaching each cell).
        """
        rows, cols = self.slope.shape
        hit_count = np.zeros((rows, cols), dtype="float32")
        toa_stack: list[np.ndarray] = []

        rng = np.random.default_rng(seed=42)

        for i in range(n_simulations):
            wind = float(rng.uniform(*wind_speed_range))
            moist = float(rng.uniform(*moisture_range))

            self.wind_speed = wind
            self.fuel_moisture["M_1hr"] = moist
            self.fuel_moisture["M_10hr"] = moist * 1.3

            result = self.simulate_spread(ignition_row, ignition_col, max_time_minutes)
            toa_i = result["time_of_arrival"].values
            reached = ~np.isnan(toa_i)
            hit_count += reached.astype(float)
            toa_stack.append(toa_i)

        prob_exposure = hit_count / n_simulations
        toa_arr = np.stack(toa_stack, axis=0)

        return xr.Dataset({
            "probability_of_exposure": (["y", "x"], prob_exposure),
            "time_of_arrival_p50": (["y", "x"], np.nanpercentile(toa_arr, 50, axis=0).astype("float32")),
            "time_of_arrival_p90": (["y", "x"], np.nanpercentile(toa_arr, 90, axis=0).astype("float32")),
        })

    def _get_cell_params(self, r: int, c: int) -> tuple[FuelParams, WeatherParams]:
        """Assemble FuelParams and WeatherParams for a single grid cell."""
        from features.vegetation.fuel_classifier import SCOTT_BURGAN_PARAMS, _FBFM40_CODE_TO_MODEL

        code = int(self.fuel_codes[r, c]) if self.fuel_codes is not None else 0
        model_str = _FBFM40_CODE_TO_MODEL.get(code, "NB9")
        p = SCOTT_BURGAN_PARAMS.get(model_str, SCOTT_BURGAN_PARAMS["NB9"])

        fuel = FuelParams(
            model=model_str,
            w_o_1hr=p["w_o_1hr"], w_o_10hr=p["w_o_10hr"], w_o_100hr=p["w_o_100hr"],
            w_o_lh=p["w_o_lh"], w_o_lw=p["w_o_lw"],
            delta=p["delta"], M_x=p["M_x"], sigma=p["sigma"], h=p["h"],
        )

        slope = float(self.slope[r, c]) if self.slope is not None else 0.0
        aspect = float(self.aspect[r, c]) if self.aspect is not None else 0.0
        weather = WeatherParams(
            M_1hr=self.fuel_moisture["M_1hr"],
            M_10hr=self.fuel_moisture["M_10hr"],
            M_100hr=self.fuel_moisture["M_100hr"],
            M_lh=self.fuel_moisture["M_lh"],
            M_lw=self.fuel_moisture["M_lw"],
            wind_speed_mph=self.wind_speed,
            wind_dir_deg=self.wind_dir,
            slope_deg=slope,
            aspect_deg=aspect,
        )
        return fuel, weather
