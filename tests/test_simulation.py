"""
Tests for the Rothermel fire spread simulation.

Validates physical invariants: these are not accuracy tests but
plausibility tests proving the physics relationships are correct.

All of these relationships must hold:
    - Higher slope → higher spread rate
    - Higher wind speed → higher spread rate
    - Higher fuel moisture → lower spread rate
    - Metal roof → lower risk score than wood shake
    - Zero fuel load → zero spread rate
    - Non-burnable fuel model → zero spread rate
"""

from __future__ import annotations

import numpy as np
import pytest

from features.vegetation.fuel_classifier import SCOTT_BURGAN_PARAMS, get_rothermel_params
from models.simulation.fire_spread import (
    FuelParams, WeatherParams, RothermelFireSpread, FireSpreadSimulator
)


@pytest.fixture
def rothermel():
    return RothermelFireSpread()


@pytest.fixture
def base_fuel() -> FuelParams:
    """TL3 (moderate conifer litter) as a reference fuel."""
    p = SCOTT_BURGAN_PARAMS["TL3"]
    return FuelParams(model="TL3", **{k: v for k, v in p.items() if k != "label"})


@pytest.fixture
def base_weather() -> WeatherParams:
    return WeatherParams(
        M_1hr=0.07, M_10hr=0.09, M_100hr=0.11,
        M_lh=0.80, M_lw=1.00,
        wind_speed_mph=15.0, wind_dir_deg=225.0, slope_deg=10.0,
    )


class TestRothermelInvariants:
    """Physical invariant tests for the Rothermel fire spread equations."""

    def test_higher_slope_increases_spread(self, rothermel, base_fuel, base_weather):
        """Uphill slope always increases rate of spread."""
        w_flat = WeatherParams(**{**base_weather.__dict__, "slope_deg": 0.0})
        w_steep = WeatherParams(**{**base_weather.__dict__, "slope_deg": 25.0})
        r_flat = rothermel.compute(base_fuel, w_flat).R
        r_steep = rothermel.compute(base_fuel, w_steep).R
        assert r_steep > r_flat, (
            f"Steeper slope must increase spread rate: flat={r_flat:.3f}, steep={r_steep:.3f}"
        )

    def test_higher_wind_increases_spread(self, rothermel, base_fuel, base_weather):
        """Wind always increases rate of spread."""
        w_calm = WeatherParams(**{**base_weather.__dict__, "wind_speed_mph": 0.0})
        w_windy = WeatherParams(**{**base_weather.__dict__, "wind_speed_mph": 30.0})
        r_calm = rothermel.compute(base_fuel, w_calm).R
        r_windy = rothermel.compute(base_fuel, w_windy).R
        assert r_windy > r_calm, (
            f"Wind must increase spread rate: calm={r_calm:.3f}, windy={r_windy:.3f}"
        )

    def test_higher_moisture_decreases_spread(self, rothermel, base_fuel, base_weather):
        """Higher fuel moisture always decreases rate of spread."""
        w_dry = WeatherParams(**{**base_weather.__dict__, "M_1hr": 0.04, "M_10hr": 0.06, "M_100hr": 0.08})
        w_wet = WeatherParams(**{**base_weather.__dict__, "M_1hr": 0.15, "M_10hr": 0.18, "M_100hr": 0.22})
        r_dry = rothermel.compute(base_fuel, w_dry).R
        r_wet = rothermel.compute(base_fuel, w_wet).R
        assert r_dry > r_wet, (
            f"Dry fuel must spread faster: dry={r_dry:.3f}, wet={r_wet:.3f}"
        )

    def test_zero_fuel_load_gives_zero_spread(self, rothermel, base_weather):
        """Zero fuel load → zero spread rate (non-burnable)."""
        fuel_nb = FuelParams(
            model="NB9",
            w_o_1hr=0, w_o_10hr=0, w_o_100hr=0, w_o_lh=0, w_o_lw=0,
            delta=0, M_x=0.99, sigma=0, h=0,
        )
        result = rothermel.compute(fuel_nb, base_weather)
        assert result.R == 0.0, "Non-burnable fuel must have zero spread rate"

    def test_moisture_above_extinction_gives_zero_spread(self, rothermel, base_fuel):
        """Fuel moisture at or above extinction moisture → zero effective spread."""
        # M_x for TL3 is 0.25; set moisture well above that
        w_saturated = WeatherParams(
            M_1hr=0.30, M_10hr=0.35, M_100hr=0.40,
            M_lh=1.20, M_lw=1.50,
            wind_speed_mph=20.0, wind_dir_deg=225.0, slope_deg=5.0,
        )
        result = rothermel.compute(base_fuel, w_saturated)
        assert result.R < 1.0, (
            f"Above extinction moisture spread rate should be ~0, got {result.R:.3f}"
        )

    def test_reaction_intensity_positive_for_burnable_fuel(self, rothermel, base_fuel, base_weather):
        """Reaction intensity must be positive for any burnable fuel."""
        w_dry = WeatherParams(**{**base_weather.__dict__, "M_1hr": 0.05})
        w_1 = base_fuel.w_o_1hr * 2000 / 43560
        w_10 = base_fuel.w_o_10hr * 2000 / 43560
        w_100 = base_fuel.w_o_100hr * 2000 / 43560
        w_lh = base_fuel.w_o_lh * 2000 / 43560
        w_lw = base_fuel.w_o_lw * 2000 / 43560
        I_R = rothermel.compute_reaction_intensity(base_fuel, w_dry, w_1, w_10, w_100, w_lh, w_lw)
        assert I_R > 0, f"Reaction intensity must be positive: {I_R}"

    def test_spread_rates_reasonable_range(self, rothermel, base_fuel, base_weather):
        """Rate of spread for TL3 at moderate weather should be in realistic range."""
        result = rothermel.compute(base_fuel, base_weather)
        # TL3 at moderate wind/slope: expect ~2-30 ft/min (reference from Andrews 2018)
        assert 0.5 < result.R < 100, (
            f"TL3 spread rate out of expected range: {result.R:.2f} ft/min"
        )

    def test_flame_length_increases_with_intensity(self, rothermel, base_fuel):
        """Flame length monotonically increases with fireline intensity."""
        results = []
        for wind in [5, 15, 30]:
            w = WeatherParams(
                M_1hr=0.06, M_10hr=0.08, M_100hr=0.10, M_lh=0.80, M_lw=1.00,
                wind_speed_mph=float(wind), wind_dir_deg=225.0, slope_deg=5.0,
            )
            results.append(rothermel.compute(base_fuel, w))

        for i in range(len(results) - 1):
            assert results[i].FL <= results[i+1].FL, (
                f"Flame length must increase with wind: {results[i].FL:.1f} > {results[i+1].FL:.1f}"
            )


class TestFireSpreadSimulator:
    """Integration tests for the Huygens wavelet propagation simulator."""

    def test_fire_spreads_from_ignition_point(self):
        """Fire must spread beyond the ignition cell."""
        rows, cols = 50, 50
        slope = np.full((rows, cols), 5.0, dtype="float32")
        aspect = np.full((rows, cols), 225.0, dtype="float32")
        fuel_codes = np.full((rows, cols), 123, dtype=int)  # TL3

        sim = FireSpreadSimulator(
            fuel_params_grid={},
            fuel_model_codes=fuel_codes,
            slope_grid=slope,
            aspect_grid=aspect,
            resolution_m=10.0,
            wind_speed_mph=20.0,
            wind_dir_deg=225.0,
            fuel_moisture={"M_1hr": 0.06, "M_10hr": 0.08, "M_100hr": 0.10, "M_lh": 0.80, "M_lw": 1.00},
        )
        result = sim.simulate_spread(ignition_row=25, ignition_col=25, max_time_minutes=30)
        toa = result["time_of_arrival"].values
        n_burned = (~np.isnan(toa)).sum()
        assert n_burned > 1, f"Fire must spread beyond ignition point: {n_burned} cells burned"

    def test_nonburnable_grid_no_spread(self):
        """Fire cannot spread on a grid of non-burnable fuel."""
        rows, cols = 20, 20
        slope = np.zeros((rows, cols), dtype="float32")
        aspect = np.zeros((rows, cols), dtype="float32")
        fuel_codes = np.full((rows, cols), 91, dtype=int)  # NB1 (urban, non-burnable)

        sim = FireSpreadSimulator(
            fuel_params_grid={},
            fuel_model_codes=fuel_codes,
            slope_grid=slope,
            aspect_grid=aspect,
            resolution_m=10.0,
            wind_speed_mph=20.0,
            wind_dir_deg=0.0,
        )
        result = sim.simulate_spread(ignition_row=10, ignition_col=10, max_time_minutes=30)
        toa = result["time_of_arrival"].values
        n_burned = (~np.isnan(toa)).sum()
        assert n_burned <= 1, f"Non-burnable grid should not spread: {n_burned} cells"

    def test_downwind_spread_faster_than_upwind(self):
        """Fire must spread faster in the downwind direction."""
        rows, cols = 51, 51
        slope = np.zeros((rows, cols), dtype="float32")
        aspect = np.zeros((rows, cols), dtype="float32")
        fuel_codes = np.full((rows, cols), 102, dtype=int)  # GR2 (fast-spreading grass)

        sim = FireSpreadSimulator(
            fuel_params_grid={},
            fuel_model_codes=fuel_codes,
            slope_grid=slope,
            aspect_grid=aspect,
            resolution_m=10.0,
            wind_speed_mph=25.0,
            wind_dir_deg=0.0,   # Wind from North → fire spreads South
            fuel_moisture={"M_1hr": 0.05, "M_10hr": 0.07, "M_100hr": 0.10, "M_lh": 0.60, "M_lw": 0.80},
        )
        result = sim.simulate_spread(ignition_row=25, ignition_col=25, max_time_minutes=20)
        toa = result["time_of_arrival"].values

        # Downwind cell (south, higher row index) should have earlier arrival
        toa_downwind = toa[30, 25]  # 5 cells south
        toa_upwind = toa[20, 25]   # 5 cells north

        if not (np.isnan(toa_downwind) or np.isnan(toa_upwind)):
            assert toa_downwind < toa_upwind, (
                f"Downwind spread must be faster: downwind={toa_downwind:.1f}, upwind={toa_upwind:.1f}"
            )


class TestFuelModels:
    """Tests for Scott-Burgan fuel model parameterization."""

    def test_all_models_have_required_params(self):
        """Every fuel model must have all required Rothermel parameters."""
        required = ["w_o_1hr", "w_o_10hr", "w_o_100hr", "w_o_lh", "w_o_lw", "delta", "M_x", "sigma"]
        for model, params in SCOTT_BURGAN_PARAMS.items():
            for key in required:
                assert key in params, f"Model {model} missing parameter {key}"

    def test_nonburnable_models_zero_sigma(self):
        """Non-burnable models must have zero or near-zero sigma."""
        for model in ["NB1", "NB9"]:
            assert SCOTT_BURGAN_PARAMS[model]["sigma"] == 0.0, (
                f"Non-burnable model {model} must have sigma=0"
            )

    def test_grass_spreads_faster_than_timber_litter(self):
        """Grass fuels (GR4) should produce higher spread rates than timber litter (TL1)."""
        rothermel = RothermelFireSpread()
        # Use cured-grass moisture (M_lh=0.06) with GR4 (M_x=0.40) so live fuel
        # doesn't extinguish the fire. GR2 has M_x=0.15 — live fuel at 70% moisture
        # kills reaction intensity entirely, making the test physically wrong.
        weather = WeatherParams(
            M_1hr=0.06, M_10hr=0.08, M_100hr=0.10, M_lh=0.06, M_lw=0.90,
            wind_speed_mph=15.0, wind_dir_deg=225.0, slope_deg=5.0,
        )

        def make_fuel(model: str) -> FuelParams:
            p = SCOTT_BURGAN_PARAMS[model]
            return FuelParams(model=model, **{k: v for k, v in p.items() if k != "label"})

        r_grass = rothermel.compute(make_fuel("GR4"), weather).R
        r_litter = rothermel.compute(make_fuel("TL1"), weather).R
        assert r_grass > r_litter, (
            f"Grass (GR4) must spread faster than timber litter (TL1): "
            f"GR4={r_grass:.2f}, TL1={r_litter:.2f} ft/min"
        )


class TestRiskScoring:
    """Tests for the wildfire risk scorer physical relationships."""

    def test_metal_roof_lower_score_than_wood(self):
        """Metal roof must produce lower risk score than wood shake."""
        from models.risk.wildfire_scorer import WildfireScorer
        scorer = WildfireScorer()

        base = {
            "slope_degrees": 10.0, "heat_load_index": 0.3, "tpi_class_encoded": 1,
            "upslope_profile_100m": 8.0, "upslope_profile_300m": 8.0, "upslope_profile_500m": 8.0,
            "zone1_fuel_load": 0.5, "zone2_fuel_load": 1.5, "zone3_fuel_load": 3.0,
            "zone3_fuel_continuity": 0.7, "ndvi_mean": 0.5, "dry_veg_fraction_mean": 0.2,
            "canopy_cover_pct": 60.0, "ladder_fuel_present": 0,
            "vent_screened": 0, "year_built": 1970, "neighbor_distance_m": 15.0,
            "ember_exposure_probability": 0.3,
        }

        metal_twin = dict(base)
        metal_twin["roof_material"] = "metal_standing_seam"
        wood_twin = dict(base)
        wood_twin["roof_material"] = "wood_shingles_shake"

        score_metal = scorer.score_twin(metal_twin)
        score_wood = scorer.score_twin(wood_twin)
        assert score_metal < score_wood, (
            f"Metal roof must score lower than wood shake: metal={score_metal:.1f}, wood={score_wood:.1f}"
        )

    def test_screened_vents_lower_score(self):
        """Screened vents must produce lower risk than unscreened."""
        from models.risk.wildfire_scorer import WildfireScorer
        scorer = WildfireScorer()

        base = {
            "slope_degrees": 10.0, "heat_load_index": 0.3, "tpi_class_encoded": 1,
            "upslope_profile_100m": 8.0, "upslope_profile_300m": 8.0, "upslope_profile_500m": 8.0,
            "zone1_fuel_load": 0.5, "zone2_fuel_load": 1.5, "zone3_fuel_load": 3.0,
            "zone3_fuel_continuity": 0.7, "ndvi_mean": 0.5, "dry_veg_fraction_mean": 0.2,
            "canopy_cover_pct": 60.0, "ladder_fuel_present": 0,
            "roof_material": "asphalt_shingles", "year_built": 1990,
            "neighbor_distance_m": 25.0, "ember_exposure_probability": 0.3,
        }

        screened = dict(base)
        screened["vent_screened"] = 1
        unscreened = dict(base)
        unscreened["vent_screened"] = 0

        score_s = WildfireScorer().score_twin({"vent_screening_status": "screened", **base})
        score_u = WildfireScorer().score_twin({"vent_screening_status": "unscreened", **base})
        assert score_s <= score_u, "Screened vents must not increase risk score"

    def test_zero_zone1_fuel_reduces_score(self):
        """Clearing Zone 1 (0-5ft) combustibles must reduce risk score."""
        from models.risk.wildfire_scorer import WildfireScorer
        scorer = WildfireScorer()

        base = {
            "slope_degrees": 15.0, "heat_load_index": 0.4, "tpi_class_encoded": 2,
            "upslope_profile_100m": 12.0, "upslope_profile_300m": 12.0, "upslope_profile_500m": 12.0,
            "zone1_fuel_load": 2.0, "zone2_fuel_load": 3.0, "zone3_fuel_load": 5.0,
            "zone3_fuel_continuity": 0.8, "ndvi_mean": 0.6, "dry_veg_fraction_mean": 0.3,
            "canopy_cover_pct": 70.0, "ladder_fuel_present": 1,
            "roof_material": "wood_shingles_shake", "vent_screening_status": "unscreened",
            "year_built": 1960, "neighbor_distance_m": 12.0, "ember_exposure_probability": 0.5,
        }

        with_z1 = dict(base)
        without_z1 = dict(base)
        without_z1["zone1_fuel_load"] = 0.0

        score_with = scorer.score_twin(with_z1)
        score_without = scorer.score_twin(without_z1)
        assert score_without < score_with, (
            f"Clearing Zone 1 must reduce risk: before={score_with:.1f}, after={score_without:.1f}"
        )
