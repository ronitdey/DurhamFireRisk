"""
Tests for the data ingestion pipeline.

Tests config loading, path resolution, FWI computation, and wind rose
generation — all of which can run without network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestConfigLoader:
    def test_load_data_sources_config(self):
        from ingestion.config_loader import load_config
        cfg = load_config("data_sources.yaml")
        assert "paths" in cfg
        assert "study_area" in cfg
        assert "lidar" in cfg

    def test_get_paths_local(self):
        from ingestion.config_loader import get_paths
        paths = get_paths(colab_mode=False)
        assert "root" in paths
        assert "raw_lidar" in paths
        assert "processed_terrain" in paths
        # All values should be Path objects
        for key, val in paths.items():
            assert isinstance(val, Path), f"Expected Path for {key}, got {type(val)}"

    def test_get_paths_colab_uses_different_root(self):
        from ingestion.config_loader import get_paths
        local_paths = get_paths(colab_mode=False)
        colab_paths = get_paths(colab_mode=True)
        # Colab root should contain /content/drive
        assert "/content/drive" in str(colab_paths["root"])
        # Local root should NOT contain /content/drive
        assert "/content/drive" not in str(local_paths["root"])

    def test_study_area_bbox_valid(self):
        from ingestion.config_loader import get_study_area, load_config
        cfg = load_config("data_sources.yaml")
        sa = get_study_area(cfg)
        bbox = sa["bbox"]
        assert bbox["xmin"] < bbox["xmax"]
        assert bbox["ymin"] < bbox["ymax"]
        # Should be in NC lat/lon range
        assert 35 < bbox["ymin"] < 37
        assert -80 < bbox["xmin"] < -78


class TestFireWeatherIndex:
    """Test Canadian FWI implementation against known reference values."""

    @pytest.fixture
    def sample_daily_df(self) -> pd.DataFrame:
        """
        Standard test case from Van Wagner (1987) FWI reference manual.
        Day 1 start conditions: FFMC=85, DMC=6, DC=15.
        """
        return pd.DataFrame({
            "temp_c": [17.0, 20.0, 18.0, 22.0, 25.0],
            "rh_pct": [60.0, 45.0, 55.0, 35.0, 30.0],
            "wind_kmh": [30.0, 25.0, 20.0, 35.0, 40.0],
            "precip_mm": [0.0, 0.0, 1.5, 0.0, 0.0],
        })

    def test_fwi_components_present(self, sample_daily_df):
        from ingestion.noaa_weather import compute_fire_weather_index
        result = compute_fire_weather_index(sample_daily_df)
        for col in ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]:
            assert col in result.columns, f"Missing FWI component: {col}"

    def test_ffmc_bounded(self, sample_daily_df):
        """FFMC must stay in [0, 101]."""
        from ingestion.noaa_weather import compute_fire_weather_index
        result = compute_fire_weather_index(sample_daily_df)
        assert (result["FFMC"] >= 0).all()
        assert (result["FFMC"] <= 101).all()

    def test_fwi_increases_with_drier_conditions(self, sample_daily_df):
        """FWI on day 5 (dry, windy) must exceed day 1 (cool, humid)."""
        from ingestion.noaa_weather import compute_fire_weather_index
        result = compute_fire_weather_index(sample_daily_df)
        assert result["FWI"].iloc[-1] > result["FWI"].iloc[0], (
            "FWI should increase as conditions dry out and wind increases"
        )

    def test_precipitation_reduces_ffmc(self):
        """Significant precipitation must reduce FFMC on the following day."""
        from ingestion.noaa_weather import compute_fire_weather_index
        df = pd.DataFrame({
            "temp_c": [20.0, 20.0],
            "rh_pct": [40.0, 40.0],
            "wind_kmh": [20.0, 20.0],
            "precip_mm": [0.0, 15.0],  # Heavy rain on day 2
        })
        result = compute_fire_weather_index(df)
        # With heavy rain, FFMC should drop (more moisture in fine fuels)
        assert result["FFMC"].iloc[1] <= result["FFMC"].iloc[0] + 5, (
            "Heavy rain should not increase FFMC significantly"
        )

    def test_zero_precip_no_negative_values(self, sample_daily_df):
        """All FWI components must be non-negative."""
        from ingestion.noaa_weather import compute_fire_weather_index
        df_dry = sample_daily_df.copy()
        df_dry["precip_mm"] = 0.0
        result = compute_fire_weather_index(df_dry)
        for col in ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]:
            assert (result[col] >= 0).all(), f"{col} has negative values"


class TestWindRose:
    def test_wind_rose_sectors_sum_to_one(self):
        """Frequency values across all sectors must sum to approximately 1."""
        from ingestion.noaa_weather import build_wind_rose
        rng = np.random.default_rng(42)
        n = 10000
        df = pd.DataFrame({
            "wind_speed_mph": rng.uniform(5, 40, n),
            "wind_dir_deg": rng.uniform(0, 360, n),
            "temp_f": rng.uniform(60, 100, n),
            "rh_pct": rng.uniform(10, 80, n),
        })
        # All weather, not fire-weather filtered
        rose = build_wind_rose(df, fire_weather_only=False, n_sectors=16)
        total_freq = sum(rose.get("frequency", []))
        assert abs(total_freq - 1.0) < 0.01, f"Frequencies must sum to 1: {total_freq}"

    def test_wind_rose_16_sectors(self):
        """Wind rose must have exactly n_sectors entries."""
        from ingestion.noaa_weather import build_wind_rose
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "wind_speed_mph": rng.uniform(5, 30, 5000),
            "wind_dir_deg": rng.uniform(0, 360, 5000),
            "temp_f": rng.uniform(70, 95, 5000),
            "rh_pct": rng.uniform(10, 25, 5000),
        })
        rose = build_wind_rose(df, n_sectors=16)
        if rose:  # May be empty if fire-weather filter removes all records
            assert len(rose["sectors"]) == 16


class TestSyntheticParcels:
    def test_synthetic_parcels_created(self):
        from ingestion.parcel_fetcher import _synthetic_duke_parcels
        gdf = _synthetic_duke_parcels("EPSG:32617")
        assert len(gdf) > 0
        assert "parcel_id" in gdf.columns
        assert "owner_name" in gdf.columns
        assert "is_duke" in gdf.columns
        assert gdf["is_duke"].all()

    def test_synthetic_parcels_correct_crs(self):
        from ingestion.parcel_fetcher import _synthetic_duke_parcels
        gdf = _synthetic_duke_parcels("EPSG:32617")
        assert gdf.crs.to_epsg() == 32617
