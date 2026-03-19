"""
Property-level wildfire risk scoring model.

Combines terrain, vegetation, structural, and exposure features into
a single 0–100 risk score using an XGBoost gradient-boosted tree model.

The model is trained on simulation outputs: for each property, the risk
score reflects the predicted probability of significant structural damage
under a distribution of fire weather scenarios.

Colab training:
    Set COLAB_MODE = True at top of train() call.
"""

from __future__ import annotations

# ── Colab mode flag ──────────────────────────────────────────────────────────
COLAB_MODE = False  # Set to True when running on Google Colab
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed; using fallback linear scorer.")


# Feature columns (must match PropertyTwin attribute names)
WILDFIRE_FEATURES = [
    # Terrain
    "slope_degrees",
    "heat_load_index",
    "tpi_class_encoded",      # 0=valley, 1=mid, 2=upper, 3=ridge
    "upslope_profile_100m",
    "upslope_profile_300m",
    "upslope_profile_500m",
    # Vegetation
    "zone1_fuel_load",
    "zone2_fuel_load",
    "zone3_fuel_load",
    "zone3_fuel_continuity",
    "ndvi_mean",
    "dry_veg_fraction_mean",
    "canopy_cover_pct",
    "ladder_fuel_present",    # binary 0/1
    # Structure
    "roof_material_encoded",  # 0=metal, 1=tile, 2=asphalt, 3=wood
    "vent_screened",          # binary 0/1
    "year_built",
    # Exposure
    "neighbor_distance_m",
    "ember_exposure_probability",
]

TPI_CLASS_MAP = {"valley": 0, "mid_slope": 1, "upper_slope": 2, "ridge": 3}
ROOF_MATERIAL_MAP = {
    "metal_standing_seam": 0, "metal_corrugated": 0, "concrete_clay_tile": 1,
    "asphalt_shingles": 2, "built_up_tar_gravel": 2, "membrane_flat": 2,
    "wood_shingles_shake": 3, "unknown_occluded": 2,
}


class WildfireScorer:
    """
    XGBoost-based wildfire risk scorer.

    Score range: 0 (lowest risk) to 100 (highest risk).
    Scores are calibrated so that a score of X corresponds roughly
    to an X% annual probability of significant loss under historical
    fire weather conditions for the study area.
    """

    def __init__(self, model_path: Path | None = None):
        self.model: Any = None
        self.feature_names = WILDFIRE_FEATURES
        if model_path and model_path.exists():
            self.load(model_path)

    def score_twin(self, twin_dict: dict) -> float:
        """
        Score a single property from a PropertyTwin attribute dict.

        Parameters
        ----------
        twin_dict:
            Dict of PropertyTwin fields (see twin/property_twin.py).

        Returns
        -------
        Risk score in [0, 100].
        """
        features = self._extract_features(twin_dict)
        if self.model is None:
            return self._fallback_score(features)
        X = np.array([[features[f] for f in self.feature_names]])
        raw = float(self.model.predict(X)[0])
        return float(np.clip(raw * 100, 0, 100))

    def score_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Score a DataFrame of properties (one row per property).

        Parameters
        ----------
        df:
            DataFrame with columns matching WILDFIRE_FEATURES (pre-encoded).

        Returns
        -------
        Series of risk scores [0, 100].
        """
        X = df[self.feature_names].fillna(0).values
        if self.model is None:
            scores = np.array([self._fallback_score(dict(zip(self.feature_names, row))) for row in X])
        else:
            raw = self.model.predict(X)
            scores = np.clip(raw * 100, 0, 100)
        return pd.Series(scores, index=df.index, name="wildfire_risk_score")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_config: dict,
        save_path: Path | None = None,
    ) -> dict[str, float]:
        """
        Train XGBoost risk model on simulation-derived labels.

        Training labels y should be the structural survival probability
        under Monte Carlo fire spread simulations (0 = certain loss, 1 = no loss);
        the model predicts 1 - survival = risk.

        Parameters
        ----------
        X_train, X_val:
            Feature matrices with columns in WILDFIRE_FEATURES order.
        y_train, y_val:
            Target values in [0, 1] (structural damage probability).
        model_config:
            Config dict from model_config.yaml["risk_scorer"]["wildfire"].
        save_path:
            If provided, save the trained model here.

        Returns
        -------
        dict of evaluation metrics: {"rmse": ..., "mae": ..., "r2": ...}
        """
        if not XGB_AVAILABLE:
            raise ImportError("Install xgboost: pip install xgboost")

        cfg = model_config
        self.model = xgb.XGBRegressor(
            n_estimators=cfg.get("n_estimators", 500),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            min_child_weight=cfg.get("min_child_weight", 5),
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=30,
            verbosity=1,
            device="cuda" if _cuda_available() else "cpu",
        )
        self.model.set_params(feature_names=self.feature_names)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = self.model.predict(X_val)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
            "mae": float(mean_absolute_error(y_val, y_pred)),
            "r2": float(r2_score(y_val, y_pred)),
        }
        logger.info(f"Wildfire scorer trained: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

        if save_path:
            self.save(save_path)

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if self.model is None:
            return pd.DataFrame()
        imp = self.model.get_booster().get_fscore()
        df = pd.DataFrame(list(imp.items()), columns=["feature", "importance"])
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"Wildfire scorer saved → {path}")

    def load(self, path: Path) -> None:
        if not XGB_AVAILABLE:
            return
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path))
        logger.info(f"Wildfire scorer loaded from {path}")

    def _extract_features(self, twin: dict) -> dict[str, float]:
        """Convert PropertyTwin dict to model feature dict."""
        tpi = twin.get("tpi_class", "mid_slope")
        tpi_enc = TPI_CLASS_MAP.get(str(tpi), 1) if isinstance(tpi, str) else int(tpi)

        roof = twin.get("roof_material", "unknown_occluded")
        roof_enc = ROOF_MATERIAL_MAP.get(str(roof), 2)

        vent = twin.get("vent_screening_status", "unknown")
        vent_enc = 1.0 if vent == "screened" else 0.0

        return {
            "slope_degrees": float(twin.get("slope_degrees", 0)),
            "heat_load_index": float(twin.get("heat_load_index", 0)),
            "tpi_class_encoded": float(tpi_enc),
            "upslope_profile_100m": float(twin.get("upslope_profile_100m", 0)),
            "upslope_profile_300m": float(twin.get("upslope_profile_300m", 0)),
            "upslope_profile_500m": float(twin.get("upslope_profile_500m", 0)),
            "zone1_fuel_load": float(twin.get("zone1_fuel_load", 0)),
            "zone2_fuel_load": float(twin.get("zone2_fuel_load", 0)),
            "zone3_fuel_load": float(twin.get("zone3_fuel_load", 0)),
            "zone3_fuel_continuity": float(twin.get("zone3_fuel_continuity", 0.5)),
            "ndvi_mean": float(twin.get("ndvi_mean", 0)),
            "dry_veg_fraction_mean": float(twin.get("dry_veg_fraction_mean", 0)),
            "canopy_cover_pct": float(twin.get("canopy_cover_pct", 0)),
            "ladder_fuel_present": float(twin.get("ladder_fuel_present", False)),
            "roof_material_encoded": float(roof_enc),
            "vent_screened": vent_enc,
            "year_built": float(twin.get("year_built", 1975)),
            "neighbor_distance_m": float(twin.get("neighbor_distance_m", 100)),
            "ember_exposure_probability": float(twin.get("ember_exposure_probability", 0)),
        }

    def _fallback_score(self, features: dict) -> float:
        """
        Rule-based fallback scorer used when no trained model is available.
        Implements physically-motivated additive risk scoring.
        """
        score = 0.0

        # Terrain (not modifiable) — up to 25 points
        score += min(features.get("slope_degrees", 0) * 0.5, 15)
        score += features.get("heat_load_index", 0) * 5
        score += features.get("tpi_class_encoded", 0) * 2  # ridge = +6

        # Vegetation — up to 30 points
        z1 = features.get("zone1_fuel_load", 0)
        z2 = features.get("zone2_fuel_load", 0)
        z3 = features.get("zone3_fuel_load", 0)
        score += min(z1 * 20, 12)   # Zone 1 is most critical
        score += min(z2 * 5, 10)
        score += min(z3 * 2, 5)
        score += features.get("ladder_fuel_present", 0) * 8

        # Structure (modifiable) — up to 35 points
        roof_scores = {0: 0, 1: 5, 2: 12, 3: 20}
        score += roof_scores.get(int(features.get("roof_material_encoded", 2)), 10)
        score += (1 - features.get("vent_screened", 0)) * 12

        # Exposure — up to 10 points
        nbr_dist = features.get("neighbor_distance_m", 100)
        score += max(0, 10 - nbr_dist * 0.5) if nbr_dist < 20 else 0
        score += features.get("ember_exposure_probability", 0) * 5

        return float(np.clip(score, 0, 100))


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
