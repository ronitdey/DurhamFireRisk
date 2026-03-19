"""
Property-level flood risk scoring model.

Combines HAND values, flow accumulation, TWI, and parcel elevation
into a flood risk score calibrated against FEMA floodplain delineations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

FLOOD_FEATURES = [
    "hand_m",                    # Height Above Nearest Drainage (m) — primary predictor
    "distance_to_stream_m",      # Euclidean distance to nearest stream (m)
    "cti",                       # Compound Topographic Index
    "twi",                       # Topographic Wetness Index
    "slope_degrees",             # Local slope
    "impervious_fraction",       # Fraction of impervious surface in parcel
    "upstream_area_km2",         # Upstream contributing area (km²)
    "elevation_m",               # Absolute elevation (m)
    "in_fema_100yr",             # Binary: inside FEMA 100-yr floodplain
]


class FloodScorer:
    """
    Gradient-boosted flood risk scorer.

    Score 0-100 calibrated to return period:
        0-20:   Outside 100-yr floodplain
        20-40:  In 100-yr floodplain but not 25-yr
        40-65:  In 25-yr floodplain but not 10-yr
        65-100: In 10-yr floodplain or more frequent flooding
    """

    def __init__(self, model_path: Path | None = None):
        self.model: Any = None
        if model_path and model_path.exists():
            self._load(model_path)

    def score_twin(self, twin_dict: dict) -> float:
        """Score a single property from PropertyTwin attributes."""
        features = self._extract_features(twin_dict)
        if self.model is None:
            return self._fallback_score(features)
        X = np.array([[features[f] for f in FLOOD_FEATURES]])
        raw = float(self.model.predict(X)[0])
        return float(np.clip(raw * 100, 0, 100))

    def _extract_features(self, twin: dict) -> dict[str, float]:
        return {
            "hand_m": float(twin.get("hand_m", 20.0)),
            "distance_to_stream_m": float(twin.get("distance_to_stream_m", 500.0)),
            "cti": float(twin.get("cti", 5.0)),
            "twi": float(twin.get("twi", 5.0)),
            "slope_degrees": float(twin.get("slope_degrees", 5.0)),
            "impervious_fraction": float(twin.get("impervious_fraction", 0.3)),
            "upstream_area_km2": float(twin.get("upstream_area_km2", 1.0)),
            "elevation_m": float(twin.get("elevation_m", 100.0)),
            "in_fema_100yr": float(twin.get("in_fema_100yr", False)),
        }

    def _fallback_score(self, features: dict) -> float:
        """Rule-based fallback scorer."""
        score = 0.0
        hand = features.get("hand_m", 20.0)
        score += max(0, (10 - hand) * 5)        # Low HAND = high flood risk
        score += max(0, (1000 - features.get("distance_to_stream_m", 500)) * 0.05)
        score += min(features.get("cti", 5) * 2, 20)
        score += features.get("in_fema_100yr", 0) * 30
        return float(np.clip(score, 0, 100))

    def _load(self, path: Path) -> None:
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor()
            self.model.load_model(str(path))
        except Exception as e:
            logger.warning(f"Could not load flood model: {e}")
