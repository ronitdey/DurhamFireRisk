"""
Tests for risk scorer, SHAP explainer, and mitigation scenario runner.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestWildfireScorer:
    @pytest.fixture
    def base_twin(self) -> dict:
        return {
            "slope_degrees": 10.0,
            "heat_load_index": 0.3,
            "tpi_class": "mid_slope",
            "upslope_profile_100m": 8.0,
            "upslope_profile_300m": 8.0,
            "upslope_profile_500m": 8.0,
            "zone1_fuel_load": 0.5,
            "zone2_fuel_load": 1.5,
            "zone3_fuel_load": 3.0,
            "zone3_fuel_continuity": 0.7,
            "ndvi_mean": 0.5,
            "dry_veg_fraction_mean": 0.2,
            "canopy_cover_pct": 60.0,
            "ladder_fuel_present": False,
            "roof_material": "asphalt_shingles",
            "vent_screening_status": "unscreened",
            "year_built": 1990,
            "neighbor_distance_m": 25.0,
            "ember_exposure_probability": 0.3,
        }

    def test_score_in_range(self, base_twin):
        from models.risk.wildfire_scorer import WildfireScorer
        scorer = WildfireScorer()
        score = scorer.score_twin(base_twin)
        assert 0 <= score <= 100, f"Score out of range: {score}"

    def test_score_dataframe(self, base_twin):
        from models.risk.wildfire_scorer import WildfireScorer, WILDFIRE_FEATURES
        import pandas as pd
        scorer = WildfireScorer()
        df = pd.DataFrame([base_twin, base_twin])
        # Populate all expected features
        from models.risk.wildfire_scorer import ROOF_MATERIAL_MAP, TPI_CLASS_MAP
        df["tpi_class_encoded"] = df.get("tpi_class", pd.Series(["mid_slope", "mid_slope"])).apply(
            lambda x: TPI_CLASS_MAP.get(x, 1)
        )
        df["roof_material_encoded"] = df.get("roof_material", pd.Series(["asphalt_shingles"] * 2)).apply(
            lambda x: ROOF_MATERIAL_MAP.get(x, 2)
        )
        df["vent_screened"] = 0
        df["ladder_fuel_present"] = 0
        for col in WILDFIRE_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        scores = scorer.score_dataframe(df)
        assert len(scores) == 2
        assert all(0 <= s <= 100 for s in scores)

    def test_high_risk_property_scores_above_50(self):
        from models.risk.wildfire_scorer import WildfireScorer
        scorer = WildfireScorer()
        high_risk = {
            "slope_degrees": 30.0, "heat_load_index": 0.8, "tpi_class": "ridge",
            "upslope_profile_100m": 25.0, "upslope_profile_300m": 25.0, "upslope_profile_500m": 20.0,
            "zone1_fuel_load": 3.0, "zone2_fuel_load": 5.0, "zone3_fuel_load": 8.0,
            "zone3_fuel_continuity": 0.9, "ndvi_mean": 0.7, "dry_veg_fraction_mean": 0.4,
            "canopy_cover_pct": 90.0, "ladder_fuel_present": True,
            "roof_material": "wood_shingles_shake", "vent_screening_status": "unscreened",
            "year_built": 1940, "neighbor_distance_m": 8.0, "ember_exposure_probability": 0.7,
        }
        score = scorer.score_twin(high_risk)
        assert score > 50, f"High-risk property should score > 50: {score}"

    def test_low_risk_property_scores_below_50(self):
        from models.risk.wildfire_scorer import WildfireScorer
        scorer = WildfireScorer()
        low_risk = {
            "slope_degrees": 2.0, "heat_load_index": 0.1, "tpi_class": "valley",
            "upslope_profile_100m": 2.0, "upslope_profile_300m": 2.0, "upslope_profile_500m": 2.0,
            "zone1_fuel_load": 0.0, "zone2_fuel_load": 0.1, "zone3_fuel_load": 0.5,
            "zone3_fuel_continuity": 0.2, "ndvi_mean": 0.2, "dry_veg_fraction_mean": 0.05,
            "canopy_cover_pct": 10.0, "ladder_fuel_present": False,
            "roof_material": "metal_standing_seam", "vent_screening_status": "screened",
            "year_built": 2020, "neighbor_distance_m": 100.0, "ember_exposure_probability": 0.0,
        }
        score = scorer.score_twin(low_risk)
        assert score < 50, f"Low-risk property should score < 50: {score}"


class TestMitigationRunner:
    @pytest.fixture
    def sample_twin(self):
        from twin.property_twin import PropertyTwin
        return PropertyTwin(
            parcel_id="TEST_001",
            address="123 Test St, Durham NC",
            slope_degrees=15.0,
            heat_load_index=0.5,
            tpi_class="mid_slope",
            upslope_profile_100m=12.0,
            upslope_profile_300m=12.0,
            upslope_profile_500m=10.0,
            zone1_fuel_load=2.0,
            zone2_fuel_load=3.5,
            zone3_fuel_load=5.0,
            zone3_fuel_continuity=0.8,
            ndvi_mean=0.6,
            dry_veg_fraction_mean=0.3,
            canopy_cover_pct=70.0,
            ladder_fuel_present=True,
            roof_material="wood_shingles_shake",
            vent_screening_status="unscreened",
            year_built=1965,
            neighbor_distance_m=12.0,
            ember_exposure_probability=0.5,
        )

    def test_rank_all_mitigations_nonempty(self, sample_twin):
        from models.risk.wildfire_scorer import WildfireScorer
        from twin.scenario_runner import MitigationScenarioRunner
        scorer = WildfireScorer()
        runner = MitigationScenarioRunner(scorer=scorer.score_twin)
        results = runner.rank_all_mitigations(sample_twin)
        assert len(results) > 0, "Should find at least one beneficial mitigation"

    def test_all_mitigations_reduce_or_keep_score(self, sample_twin):
        from models.risk.wildfire_scorer import WildfireScorer
        from twin.scenario_runner import MitigationScenarioRunner
        scorer = WildfireScorer()
        runner = MitigationScenarioRunner(scorer=scorer.score_twin)
        results = runner.rank_all_mitigations(sample_twin)
        for r in results:
            assert r.risk_reduction_pts >= 0, (
                f"Action '{r.action.key}' increased risk: {r.risk_reduction_pts:.2f}"
            )

    def test_combined_actions_reduce_more_than_single(self, sample_twin):
        from models.risk.wildfire_scorer import WildfireScorer
        from twin.scenario_runner import MitigationScenarioRunner, MITIGATION_ACTIONS
        scorer = WildfireScorer()
        runner = MitigationScenarioRunner(scorer=scorer.score_twin)

        single = runner.run_counterfactual(sample_twin, ["replace_wood_roof_with_metal"])
        combined = runner.run_counterfactual(
            sample_twin, ["replace_wood_roof_with_metal", "screen_all_vents", "clear_zone1_combustibles"]
        )
        assert combined.mitigated_risk_score <= single.mitigated_risk_score, (
            "Combined actions must reduce score at least as much as single action"
        )

    def test_result_has_required_fields(self, sample_twin):
        from models.risk.wildfire_scorer import WildfireScorer
        from twin.scenario_runner import MitigationScenarioRunner
        scorer = WildfireScorer()
        runner = MitigationScenarioRunner(scorer=scorer.score_twin)
        result = runner.run_counterfactual(sample_twin, ["screen_all_vents"])
        d = result.to_dict()
        for key in ["original_risk_score", "mitigated_risk_score", "risk_reduction_pct", "actions"]:
            assert key in d


class TestPropertyTwin:
    def test_serialization_roundtrip(self):
        """PropertyTwin must serialize to dict and deserialize correctly."""
        from twin.property_twin import PropertyTwin
        twin = PropertyTwin(
            parcel_id="ROUNDTRIP_001",
            address="Test Address",
            slope_degrees=12.5,
            roof_material="metal_standing_seam",
            wildfire_risk_score=42.3,
        )
        d = twin.to_dict()
        twin2 = PropertyTwin.from_dict(d)
        assert twin2.parcel_id == twin.parcel_id
        assert abs(twin2.slope_degrees - twin.slope_degrees) < 0.01
        assert twin2.roof_material == twin.roof_material
        assert abs(twin2.wildfire_risk_score - twin.wildfire_risk_score) < 0.01

    def test_risk_category_labels(self):
        from twin.property_twin import PropertyTwin
        t = PropertyTwin(parcel_id="X")
        t.composite_risk_score = 20.0
        assert t.risk_category() == "LOW"
        t.composite_risk_score = 45.0
        assert t.risk_category() == "MODERATE"
        t.composite_risk_score = 65.0
        assert t.risk_category() == "HIGH"
        t.composite_risk_score = 80.0
        assert t.risk_category() == "VERY HIGH"

    def test_feature_vector_length(self):
        from twin.property_twin import PropertyTwin
        from models.risk.wildfire_scorer import WILDFIRE_FEATURES
        twin = PropertyTwin(parcel_id="FV_001", roof_material="asphalt_shingles")
        fv = twin.to_feature_vector(WILDFIRE_FEATURES)
        assert len(fv) == len(WILDFIRE_FEATURES)
