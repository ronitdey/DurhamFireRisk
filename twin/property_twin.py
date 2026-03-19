"""
Digital twin object for a single parcel.

A PropertyTwin is the central data object in the risk engine. Every
risk assessment, simulation, and mitigation recommendation operates
on this object. One PropertyTwin per insured property.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import Polygon, mapping, shape


@dataclass
class PropertyTwin:
    """
    Complete physics-informed digital representation of a single parcel.

    Fields are populated in stages:
        1. Parcel geometry and identity (from GIS)
        2. Terrain features (from DEM/LiDAR)
        3. Vegetation features (from NAIP + LANDFIRE)
        4. Structure features (from vision model)
        5. Exposure features (computed from spatial relationships)
        6. Risk scores and simulation results (populated by scorer)
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    parcel_id: str
    address: str = ""
    name: str = ""   # For named Duke buildings
    geometry: Optional[Polygon] = field(default=None, repr=False)
    building_footprints: list = field(default_factory=list, repr=False)
    county: str = "Durham"
    is_duke_owned: bool = False

    # ── Terrain (from DEM/LiDAR) ─────────────────────────────────────────────
    slope_degrees: float = 0.0
    aspect_degrees: float = 0.0
    northness: float = 0.0      # cos(aspect) — for ML input
    eastness: float = 0.0       # sin(aspect) — for ML input
    tpi_class: str = "mid_slope"  # ridge | upper_slope | mid_slope | valley
    tri: float = 0.0              # Terrain Ruggedness Index
    heat_load_index: float = 0.0  # 0-1; SW-facing + steep = highest
    twi: float = 5.0              # Topographic Wetness Index
    upslope_profile_100m: float = 0.0   # Mean slope uphill within 100m
    upslope_profile_300m: float = 0.0
    upslope_profile_500m: float = 0.0

    # ── Vegetation (from NAIP + LANDFIRE) ─────────────────────────────────────
    fuel_models: dict = field(default_factory=dict)  # {code: fractional coverage}
    ndvi_mean: float = 0.0
    ndvi_p90: float = 0.0
    ndvi_std: float = 0.0
    ndwi_mean: float = 0.0
    evi_mean: float = 0.0
    dry_veg_fraction_mean: float = 0.0
    canopy_cover_pct: float = 0.0
    canopy_height_mean_m: float = 0.0
    zone1_fuel_load: float = 0.0         # 0-5ft zone (tons/acre)
    zone2_fuel_load: float = 0.0         # 5-30ft zone
    zone3_fuel_load: float = 0.0         # 30-100ft zone
    zone3_fuel_continuity: float = 0.5   # 0 = gaps; 1 = continuous
    zone2_dominant_fuel: str = "unknown"
    ladder_fuel_present: bool = False

    # ── Structure (from vision model) ────────────────────────────────────────
    roof_material: str = "unknown_occluded"
    roof_material_confidence: float = 0.0
    vent_screening_status: str = "unknown"  # screened | unscreened | unknown
    structure_type: str = "unknown"
    year_built: int = 1975
    stories: int = 1
    building_sf: float = 0.0
    wall_material: str = "unknown"
    deck_material: str = "unknown"
    assessed_value: float = 0.0

    # ── Exposure ─────────────────────────────────────────────────────────────
    neighbor_distance_m: float = 100.0
    neighbor_flag_15m: bool = False
    road_access_quality: str = "unknown"
    water_supply_proximity_m: float = 500.0

    # ── Risk scores (populated after model runs) ──────────────────────────────
    wildfire_risk_score: float = 0.0
    flood_risk_score: float = 0.0
    composite_risk_score: float = 0.0
    risk_drivers: dict = field(default_factory=dict)   # feature → SHAP value
    risk_percentile: float = 0.0    # Percentile vs. all study area properties

    # ── Simulation results ────────────────────────────────────────────────────
    fire_arrival_time_p50: float = float("inf")
    fire_arrival_time_p90: float = float("inf")
    ember_exposure_probability: float = 0.0
    flood_inundation_depth_p50: float = 0.0
    flood_inundation_depth_100yr: float = 0.0

    def to_feature_vector(self, feature_names: list[str]) -> list[float]:
        """
        Convert twin fields to a flat numeric feature vector in the order
        expected by WildfireScorer.WILDFIRE_FEATURES.
        """
        from models.risk.wildfire_scorer import ROOF_MATERIAL_MAP, TPI_CLASS_MAP
        d = asdict(self)
        d["tpi_class_encoded"] = float(TPI_CLASS_MAP.get(self.tpi_class, 1))
        d["roof_material_encoded"] = float(ROOF_MATERIAL_MAP.get(self.roof_material, 2))
        d["vent_screened"] = 1.0 if self.vent_screening_status == "screened" else 0.0
        d["ladder_fuel_present"] = float(self.ladder_fuel_present)
        return [d.get(f, 0.0) for f in feature_names]

    def risk_category(self) -> str:
        """Map composite risk score to human-readable category."""
        s = self.composite_risk_score
        if s < 30:
            return "LOW"
        elif s < 55:
            return "MODERATE"
        elif s < 75:
            return "HIGH"
        else:
            return "VERY HIGH"

    def to_dict(self) -> dict:
        """Serialize twin to a JSON-serializable dict."""
        d = asdict(self)
        if self.geometry is not None:
            d["geometry"] = mapping(self.geometry)
        d["building_footprints"] = [mapping(fp) for fp in self.building_footprints if fp is not None]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PropertyTwin":
        """Deserialize a PropertyTwin from dict (e.g., loaded from JSON)."""
        geom_data = d.pop("geometry", None)
        fps_data = d.pop("building_footprints", [])
        twin = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if geom_data:
            twin.geometry = shape(geom_data)
        twin.building_footprints = [shape(fp) for fp in fps_data]
        return twin

    def save(self, path: Path) -> None:
        """Serialize twin to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)

    @classmethod
    def load(cls, path: Path) -> "PropertyTwin":
        """Load a PropertyTwin from JSON."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    def __repr__(self) -> str:
        return (
            f"PropertyTwin(parcel_id={self.parcel_id!r}, "
            f"address={self.address!r}, "
            f"risk={self.composite_risk_score:.1f} [{self.risk_category()}])"
        )


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
