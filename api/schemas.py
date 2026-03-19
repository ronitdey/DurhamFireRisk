"""
Pydantic request/response schemas for the FastAPI risk assessment API.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AssessRequest(BaseModel):
    address: str = Field(..., description="Property address or Duke building name")
    parcel_id: Optional[str] = Field(None, description="County parcel ID if known")
    include_flood: bool = Field(True, description="Include flood risk assessment")
    include_wildfire: bool = Field(True, description="Include wildfire risk assessment")
    run_simulation: bool = Field(False, description="Run Monte Carlo fire spread simulation")
    n_simulations: int = Field(100, ge=10, le=1000, description="Number of MC simulations if run_simulation=True")


class TerrainFeatures(BaseModel):
    slope_degrees: float
    aspect_degrees: float
    tpi_class: str
    heat_load_index: float
    upslope_profile_300m: float


class VegetationFeatures(BaseModel):
    ndvi_mean: float
    dry_veg_fraction_mean: float
    canopy_cover_pct: float
    zone1_fuel_load: float
    zone2_fuel_load: float
    zone3_fuel_load: float
    ladder_fuel_present: bool


class StructureFeatures(BaseModel):
    roof_material: str
    roof_material_confidence: float
    vent_screening_status: str
    year_built: int


class RiskAttribution(BaseModel):
    feature: str
    label: str
    category: str
    controllable: bool
    shap_value: float
    feature_value: float


class AssessResponse(BaseModel):
    parcel_id: str
    address: str
    wildfire_risk_score: Optional[float]
    flood_risk_score: Optional[float]
    composite_risk_score: float
    risk_category: str                      # LOW | MODERATE | HIGH | VERY HIGH
    risk_percentile: float                  # vs. Durham County
    terrain: TerrainFeatures
    vegetation: VegetationFeatures
    structure: StructureFeatures
    top_risk_drivers: list[RiskAttribution]
    top_mitigations: list[RiskAttribution]
    fire_arrival_time_p50_min: Optional[float]
    ember_exposure_probability: Optional[float]


class MitigateRequest(BaseModel):
    parcel_id: str = Field(..., description="County parcel ID")
    actions: list[str] = Field(
        ...,
        description="List of mitigation action keys from the MITIGATION_ACTIONS catalog",
        example=["replace_wood_roof_with_metal", "screen_all_vents"],
    )


class ActionImpact(BaseModel):
    action: str
    description: str
    risk_reduction_pts: float
    risk_reduction_pct: float
    fire_arrival_time_gained_min: float
    cost_estimate_usd: tuple[int, int]
    cost_per_risk_point: str
    priority_rank: int


class MitigateResponse(BaseModel):
    parcel_id: str
    original_risk_score: float
    mitigated_risk_score: float
    risk_reduction_pct: float
    action_results: list[ActionImpact]


class ExplainResponse(BaseModel):
    parcel_id: str
    risk_score: float
    base_value: float
    controllable_risk_points: float
    uncontrollable_risk_points: float
    attributions: list[RiskAttribution]
    summary: str


class SimulateRequest(BaseModel):
    ignition_lat: float = Field(..., ge=35.5, le=36.5, description="Ignition point latitude")
    ignition_lon: float = Field(..., ge=-79.5, le=-78.5, description="Ignition point longitude")
    wind_speed_mph: float = Field(20.0, ge=0, le=60)
    wind_direction_degrees: float = Field(225.0, ge=0, le=360)
    fire_weather_index: float = Field(20.0, ge=0, le=100)
    max_time_minutes: int = Field(60, ge=5, le=240)
    n_simulations: int = Field(1, ge=1, le=100)


class SimulateResponse(BaseModel):
    n_cells_burned: int
    area_burned_ha: float
    parcels_reached_30min: list[str]
    parcels_reached_60min: list[str]
    max_fireline_intensity_btu: float
    max_flame_length_ft: float


class CampusOverviewResponse(BaseModel):
    total_parcels: int
    duke_parcels: int
    mean_wildfire_risk: float
    mean_flood_risk: float
    high_risk_count: int            # score > 55
    very_high_risk_count: int       # score > 75
    highest_risk_parcels: list[dict]  # top 10
    top_campus_mitigation: str
    estimated_campus_risk_reduction_pct: float
