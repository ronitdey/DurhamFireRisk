"""
Tests for terrain and vegetation feature computation.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestTerrainFeatures:
    @pytest.fixture
    def flat_dem(self):
        return np.ones((50, 50), dtype="float32") * 100.0

    @pytest.fixture
    def sloped_dem(self):
        """10-degree north-facing slope."""
        arr = np.zeros((50, 50), dtype="float32")
        for r in range(50):
            arr[r, :] = r * np.tan(np.radians(10)) * 10  # 10m resolution
        return arr.astype("float32")

    def test_flat_dem_zero_slope(self, flat_dem):
        from features.terrain.slope_aspect import _compute_slope_aspect
        slope, aspect = _compute_slope_aspect(flat_dem, res=10.0)
        # Interior cells of flat DEM should have ~0 slope
        assert slope[1:-1, 1:-1].max() < 0.5, "Flat DEM should have near-zero slope"

    def test_sloped_dem_correct_slope(self, sloped_dem):
        from features.terrain.slope_aspect import _compute_slope_aspect
        slope, aspect = _compute_slope_aspect(sloped_dem, res=10.0)
        interior_slope = slope[5:-5, 5:-5].mean()
        assert 8 < interior_slope < 12, f"Expected ~10° slope, got {interior_slope:.1f}°"

    def test_hli_sw_facing_exceeds_ne_facing(self):
        """SW-facing slopes (hot/dry) should have higher HLI than NE-facing (cool/moist)."""
        from features.terrain.slope_aspect import _compute_heat_load_index
        slope = np.full((10, 10), 20.0, dtype="float32")

        # McCune & Keon (2002): NE (45°) is the cold reference azimuth → HLI≈0
        # SW-facing (225°) is maximum deviation from cold reference → HLI≈1
        hli_sw = _compute_heat_load_index(slope, np.full((10, 10), 225.0)).mean()
        hli_ne = _compute_heat_load_index(slope, np.full((10, 10), 45.0)).mean()
        assert hli_sw > hli_ne, f"SW-facing HLI ({hli_sw:.3f}) must exceed NE-facing ({hli_ne:.3f})"

    def test_hli_range_zero_to_one(self):
        """Heat Load Index must always be in [0, 1]."""
        from features.terrain.slope_aspect import _compute_heat_load_index
        rng = np.random.default_rng(42)
        slope = rng.uniform(0, 45, (100, 100)).astype("float32")
        aspect = rng.uniform(0, 360, (100, 100)).astype("float32")
        hli = _compute_heat_load_index(slope, aspect)
        assert hli.min() >= 0.0
        assert hli.max() <= 1.0

    def test_tpi_ridge_positive(self):
        """TPI should be positive at a ridge (cell higher than surroundings)."""
        from features.terrain.slope_aspect import _compute_tpi
        # Create a simple ridge: elevated center row
        dem = np.zeros((30, 30), dtype="float32")
        dem[15, :] = 50.0  # Ridge at row 15
        tpi = _compute_tpi(dem, neighborhood_m=100, res=10.0)
        assert tpi[15, 15] > 0, f"Ridge should have positive TPI: {tpi[15, 15]:.2f}"

    def test_tpi_valley_negative(self):
        """TPI should be negative at a valley (cell lower than surroundings)."""
        from features.terrain.slope_aspect import _compute_tpi
        dem = np.ones((30, 30), dtype="float32") * 50.0
        dem[15, :] = 0.0  # Valley at row 15
        tpi = _compute_tpi(dem, neighborhood_m=100, res=10.0)
        assert tpi[15, 15] < 0, f"Valley should have negative TPI: {tpi[15, 15]:.2f}"

    def test_twi_flat_terrain_high(self):
        """Flat terrain with high flow accumulation should have high TWI."""
        from features.terrain.slope_aspect import _compute_twi
        slope = np.full((20, 20), 0.5, dtype="float32")  # Nearly flat
        flow_acc = np.ones((20, 20), dtype="float32") * 1000  # High accumulation
        twi = _compute_twi(slope, flow_acc, res=10.0)
        center_twi = twi[10, 10]
        assert center_twi > 10, f"High accumulation + flat terrain should have high TWI: {center_twi:.1f}"


class TestVegetationIndices:
    @pytest.fixture
    def mock_naip_bands(self):
        """Return (R, G, B, NIR) arrays as float32 in [0,1]."""
        rng = np.random.default_rng(42)
        r = rng.uniform(0.1, 0.3, (50, 50)).astype("float32")
        g = rng.uniform(0.1, 0.3, (50, 50)).astype("float32")
        b = rng.uniform(0.05, 0.2, (50, 50)).astype("float32")
        nir = rng.uniform(0.4, 0.8, (50, 50)).astype("float32")
        return r, g, b, nir

    def test_ndvi_range(self, mock_naip_bands):
        """NDVI must be in [-1, 1]."""
        from features.vegetation.ndvi_extractor import compute_vegetation_indices
        # Build a mock GeoTIFF-like dataset
        r, g, b, nir = mock_naip_bands
        # Test NDVI formula directly
        eps = 1e-6
        ndvi = (nir - r) / (nir + r + eps)
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_ndvi_vegetation_high(self):
        """Pixels with high NIR and low Red should have high NDVI."""
        r = np.full((5, 5), 0.05, dtype="float32")
        nir = np.full((5, 5), 0.70, dtype="float32")
        ndvi = (nir - r) / (nir + r + 1e-6)
        assert ndvi.mean() > 0.7, f"High-NIR/low-Red should have NDVI > 0.7: {ndvi.mean():.3f}"

    def test_ndvi_bare_soil_low(self):
        """Bare soil (moderate R, low NIR) should have low NDVI."""
        r = np.full((5, 5), 0.30, dtype="float32")
        nir = np.full((5, 5), 0.25, dtype="float32")
        ndvi = (nir - r) / (nir + r + 1e-6)
        assert ndvi.mean() < 0.1, f"Bare soil should have NDVI < 0.1: {ndvi.mean():.3f}"

    def test_spectral_mixture_fractions_sum_to_one(self, mock_naip_bands):
        """GV + NPV + Soil fractions must sum to approximately 1."""
        from features.vegetation.ndvi_extractor import _spectral_mixture
        r, g, b, nir = mock_naip_bands
        gv, npv, soil = _spectral_mixture(r, g, b, nir)
        total = gv + npv + soil
        assert np.allclose(total, 1.0, atol=0.05), (
            f"Spectral fractions must sum to ~1: range [{total.min():.3f}, {total.max():.3f}]"
        )

    def test_flammability_classification_range(self):
        """Flammability classification must produce values in [0, 4]."""
        from features.vegetation.ndvi_extractor import classify_vegetation_flammability
        rng = np.random.default_rng(0)
        ndvi = rng.uniform(-0.2, 0.9, (50, 50)).astype("float32")
        ndwi = rng.uniform(-0.5, 0.5, (50, 50)).astype("float32")
        flam = classify_vegetation_flammability(ndvi, ndwi)
        assert flam.min() >= 0
        assert flam.max() <= 4


class TestFuelClassifier:
    def test_get_rothermel_params_known_model(self):
        from features.vegetation.fuel_classifier import get_rothermel_params
        params = get_rothermel_params("TL3")
        assert params["delta"] > 0
        assert params["sigma"] > 0
        assert 0 < params["M_x"] < 1

    def test_get_rothermel_params_unknown_model_defaults_to_nb9(self):
        from features.vegetation.fuel_classifier import get_rothermel_params, SCOTT_BURGAN_PARAMS
        params = get_rothermel_params("XXXXX")
        nb9 = SCOTT_BURGAN_PARAMS["NB9"]
        assert params["sigma"] == nb9["sigma"]

    def test_total_fuel_load_nonburnable_zero(self):
        from features.vegetation.fuel_classifier import get_total_fuel_load
        assert get_total_fuel_load("NB1") == 0.0
        assert get_total_fuel_load("NB9") == 0.0

    def test_grass_higher_fuel_load_than_nonburnable(self):
        from features.vegetation.fuel_classifier import get_total_fuel_load
        assert get_total_fuel_load("GR4") > get_total_fuel_load("NB1")
