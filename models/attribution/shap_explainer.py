"""
SHAP-based risk attribution.

Decomposes a property's wildfire risk score into contributions from each
input feature, separating controllable (structure/vegetation) from
uncontrollable (terrain/location) factors. This is the core of Stand's
"what to change, and what does each change buy you" value proposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")


# Feature metadata: which features are controllable by the homeowner?
FEATURE_METADATA: dict[str, dict] = {
    "slope_degrees":           {"category": "terrain",    "controllable": False, "label": "Terrain slope"},
    "heat_load_index":         {"category": "terrain",    "controllable": False, "label": "Heat load index"},
    "tpi_class_encoded":       {"category": "terrain",    "controllable": False, "label": "Topographic position"},
    "upslope_profile_100m":    {"category": "terrain",    "controllable": False, "label": "100m upslope gradient"},
    "upslope_profile_300m":    {"category": "terrain",    "controllable": False, "label": "300m upslope gradient"},
    "upslope_profile_500m":    {"category": "terrain",    "controllable": False, "label": "500m upslope gradient"},
    "zone1_fuel_load":         {"category": "vegetation", "controllable": True,  "label": "Zone 1 fuel load (0-5ft)"},
    "zone2_fuel_load":         {"category": "vegetation", "controllable": True,  "label": "Zone 2 fuel load (5-30ft)"},
    "zone3_fuel_load":         {"category": "vegetation", "controllable": True,  "label": "Zone 3 fuel load (30-100ft)"},
    "zone3_fuel_continuity":   {"category": "vegetation", "controllable": True,  "label": "Zone 3 fuel continuity"},
    "ndvi_mean":               {"category": "vegetation", "controllable": False, "label": "Mean NDVI"},
    "dry_veg_fraction_mean":   {"category": "vegetation", "controllable": False, "label": "Dry vegetation fraction"},
    "canopy_cover_pct":        {"category": "vegetation", "controllable": True,  "label": "Canopy cover"},
    "ladder_fuel_present":     {"category": "vegetation", "controllable": True,  "label": "Ladder fuels present"},
    "roof_material_encoded":   {"category": "structure",  "controllable": True,  "label": "Roof material"},
    "vent_screened":           {"category": "structure",  "controllable": True,  "label": "Vent screening"},
    "year_built":              {"category": "structure",  "controllable": False, "label": "Year built"},
    "neighbor_distance_m":     {"category": "exposure",   "controllable": False, "label": "Neighbor distance"},
    "ember_exposure_probability": {"category": "exposure", "controllable": False, "label": "Ember exposure probability"},
}


@dataclass
class FeatureAttribution:
    """SHAP attribution for a single feature."""
    feature: str
    label: str
    category: str
    controllable: bool
    shap_value: float          # Contribution to risk score (positive = increases risk)
    feature_value: float       # Actual value of this feature for this property
    baseline_value: float      # Expected value across reference population


@dataclass
class RiskExplanation:
    """Full SHAP-based explanation for a single property's risk score."""
    parcel_id: str
    risk_score: float
    base_value: float          # E[f(X)] — mean prediction across background
    attributions: list[FeatureAttribution]

    @property
    def controllable_risk_points(self) -> float:
        """Total risk points from features the homeowner can change."""
        return sum(a.shap_value for a in self.attributions if a.controllable and a.shap_value > 0)

    @property
    def uncontrollable_risk_points(self) -> float:
        """Total risk points from location/terrain (not modifiable)."""
        return sum(a.shap_value for a in self.attributions if not a.controllable and a.shap_value > 0)

    @property
    def top_risks(self) -> list[FeatureAttribution]:
        """Features sorted by absolute SHAP contribution (descending)."""
        return sorted(self.attributions, key=lambda a: abs(a.shap_value), reverse=True)

    @property
    def top_mitigations(self) -> list[FeatureAttribution]:
        """Controllable features with positive (risk-increasing) SHAP values, sorted by impact."""
        return sorted(
            [a for a in self.attributions if a.controllable and a.shap_value > 0],
            key=lambda a: a.shap_value,
            reverse=True,
        )

    def summary_text(self) -> str:
        """Generate human-readable risk attribution summary."""
        lines = [
            f"Property Risk Score: {self.risk_score:.1f}/100",
            f"",
            f"CONTROLLABLE factors ({self.controllable_risk_points:.1f} risk points):",
        ]
        for a in self.top_mitigations[:5]:
            lines.append(f"  + {a.label:<40} {a.shap_value:+.1f} pts")

        lines += ["", f"TERRAIN/LOCATION factors ({self.uncontrollable_risk_points:.1f} risk points):"]
        for a in [x for x in self.top_risks if not x.controllable][:5]:
            lines.append(f"  {'+'if a.shap_value>0 else '-'} {a.label:<40} {a.shap_value:+.1f} pts")

        return "\n".join(lines)


class WildfireRiskExplainer:
    """
    SHAP-based explainer for the wildfire risk scoring model.

    Supports both tree explainers (for XGBoost) and deep explainers
    (for the CNN-ViT backbone), configured via model_config.yaml.
    """

    def __init__(
        self,
        scorer_model: Any,
        background_data: np.ndarray,
        feature_names: list[str] | None = None,
        explainer_type: str = "tree",
    ):
        """
        Parameters
        ----------
        scorer_model:
            Trained model (XGBoost XGBRegressor or PyTorch Module).
        background_data:
            Reference dataset for computing SHAP baselines
            (shape: [n_background_samples, n_features]).
        feature_names:
            List of feature names corresponding to columns in background_data.
        explainer_type:
            "tree" for XGBoost, "deep" for neural networks.
        """
        self.feature_names = feature_names or list(FEATURE_METADATA.keys())
        self.explainer_type = explainer_type
        self._explainer = None

        if not SHAP_AVAILABLE:
            logger.warning("SHAP unavailable; using linear approximation fallback.")
            self._background_mean = background_data.mean(axis=0)
            self._baseline_pred = 0.5
            return

        if explainer_type == "tree":
            self._explainer = shap.TreeExplainer(scorer_model, background_data)
        else:
            self._explainer = shap.DeepExplainer(scorer_model, background_data)

        self._baseline_pred = self._explainer.expected_value
        if hasattr(self._baseline_pred, "__len__"):
            self._baseline_pred = float(self._baseline_pred[0])

    def explain_property(
        self,
        feature_vector: np.ndarray,
        parcel_id: str,
        risk_score: float,
    ) -> RiskExplanation:
        """
        Compute SHAP values for a single property.

        Parameters
        ----------
        feature_vector:
            1D array of feature values (in WILDFIRE_FEATURES order).
        parcel_id:
            Identifier for the property.
        risk_score:
            Pre-computed risk score (0-100) for display.

        Returns
        -------
        RiskExplanation with per-feature SHAP attributions.
        """
        if not SHAP_AVAILABLE or self._explainer is None:
            shap_vals = self._linear_approx(feature_vector)
        else:
            X = feature_vector.reshape(1, -1)
            shap_vals_raw = self._explainer.shap_values(X)
            shap_vals = np.array(shap_vals_raw).flatten() * 100  # Scale to 0-100

        attributions: list[FeatureAttribution] = []
        for i, feat_name in enumerate(self.feature_names):
            meta = FEATURE_METADATA.get(feat_name, {
                "category": "other", "controllable": False, "label": feat_name
            })
            attributions.append(FeatureAttribution(
                feature=feat_name,
                label=meta["label"],
                category=meta["category"],
                controllable=meta["controllable"],
                shap_value=float(shap_vals[i]),
                feature_value=float(feature_vector[i]),
                baseline_value=float(self._background_mean[i]) if hasattr(self, "_background_mean") else 0.0,
            ))

        return RiskExplanation(
            parcel_id=parcel_id,
            risk_score=risk_score,
            base_value=float(self._baseline_pred) * 100,
            attributions=attributions,
        )

    def explain_batch(
        self,
        feature_matrix: np.ndarray,
        parcel_ids: list[str],
        risk_scores: list[float],
    ) -> list[RiskExplanation]:
        """Compute SHAP explanations for a batch of properties."""
        explanations: list[RiskExplanation] = []
        for i, (pid, rs) in enumerate(zip(parcel_ids, risk_scores)):
            exp = self.explain_property(feature_matrix[i], pid, rs)
            explanations.append(exp)
        return explanations

    def top_campus_mitigations(
        self,
        explanations: list[RiskExplanation],
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Aggregate SHAP values across all campus properties to identify
        which mitigation actions would reduce total campus risk the most.
        """
        action_totals: dict[str, float] = {}
        for exp in explanations:
            for attr in exp.top_mitigations:
                action_totals[attr.label] = action_totals.get(attr.label, 0) + attr.shap_value

        df = pd.DataFrame(list(action_totals.items()), columns=["mitigation_action", "total_risk_points"])
        df["avg_risk_points_per_property"] = df["total_risk_points"] / max(len(explanations), 1)
        return df.sort_values("total_risk_points", ascending=False).head(top_n).reset_index(drop=True)

    def _linear_approx(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Linear approximation of SHAP values when SHAP library is unavailable.
        Uses gradient of a simple linear model fit to background data.
        """
        if not hasattr(self, "_background_mean"):
            return np.zeros(len(feature_vector))
        diff = feature_vector - self._background_mean
        weights = np.array([
            FEATURE_METADATA.get(f, {}).get("_weight", 1.0)
            for f in self.feature_names
        ])
        return diff * weights * 0.5  # Simplified attribution
