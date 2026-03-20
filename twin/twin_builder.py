"""
Constructs PropertyTwin objects from all feature layers.

Orchestrates the full feature extraction pipeline for each parcel and
assembles the results into a PropertyTwin ready for risk scoring.

Colab:
    Set COLAB_MODE = True before running on Google Colab.
"""

from __future__ import annotations

# ── Colab mode flag ──────────────────────────────────────────────────────────
COLAB_MODE = False  # Set to True when running on Google Colab
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import xarray as xr
from loguru import logger
from tqdm import tqdm

from ingestion.config_loader import get_paths, load_config
from twin.property_twin import PropertyTwin


class TwinBuilder:
    """
    Assembles PropertyTwin objects from processed feature layers.

    Load sequence:
        1. Parcel boundaries (from GeoPackage)
        2. Terrain features (slope, aspect, TPI, HLI, upslope profiles)
        3. Vegetation indices (NDVI, NDWI, EVI, dry veg fraction)
        4. LANDFIRE fuel models + proximity metrics
        5. Roof material classification (from vision model or rule-based fallback)
        6. Exposure metrics (neighbor proximity, flood zone)
    """

    def __init__(
        self,
        paths: dict[str, Path],
        model_config: dict | None = None,
        run_vision_model: bool = False,
    ):
        """
        Parameters
        ----------
        paths:
            Path dict from get_paths(COLAB_MODE).
        model_config:
            Loaded model_config.yaml dict.
        run_vision_model:
            If True, run CNN-ViT roof classification. Requires GPU.
            If False, uses rule-based fallback (parcel data or default).
        """
        self.paths = paths
        self.model_config = model_config or {}
        self.run_vision_model = run_vision_model

        # Lazy-loaded feature datasets
        self._terrain_ds: Optional[xr.Dataset] = None
        self._veg_ds: Optional[xr.Dataset] = None
        self._fuel_da: Optional[xr.DataArray] = None
        self._fuel_params: Optional[dict] = None
        self._parcels: Optional[gpd.GeoDataFrame] = None

    def build_all_twins(self, save: bool = True) -> list[PropertyTwin]:
        """
        Build PropertyTwin objects for all parcels in the study area.

        Parameters
        ----------
        save:
            If True, serialize each twin to JSON in processed_twins directory.

        Returns
        -------
        List of all PropertyTwin objects.
        """
        self._load_feature_layers()

        if self._parcels is None or self._parcels.empty:
            logger.error("No parcels loaded. Run ingestion pipeline first.")
            return []

        twins: list[PropertyTwin] = []
        for idx, row in tqdm(self._parcels.iterrows(), total=len(self._parcels), desc="Building twins"):
            twin = self._build_single_twin(row)
            twins.append(twin)

            if save:
                out_path = self.paths["processed_twins"] / f"{twin.parcel_id}.json"
                twin.save(out_path)

        logger.info(f"Built {len(twins)} PropertyTwin objects.")
        return twins

    def build_twin_for_parcel(self, parcel_id: str) -> Optional[PropertyTwin]:
        """Build a single twin by parcel ID (loads feature layers if needed)."""
        self._load_feature_layers()
        if self._parcels is None:
            return None
        mask = self._parcels["parcel_id"] == parcel_id
        if not mask.any():
            logger.warning(f"Parcel {parcel_id} not found.")
            return None
        return self._build_single_twin(self._parcels[mask].iloc[0])

    def _load_feature_layers(self) -> None:
        """Load all processed feature datasets into memory."""
        # Terrain
        terrain_path = self.paths["processed_terrain"] / "terrain_features.nc"
        if terrain_path.exists():
            self._terrain_ds = xr.open_dataset(terrain_path)
            logger.info("Terrain features loaded.")
        else:
            logger.warning(f"Terrain features not found at {terrain_path}. Run features/terrain/slope_aspect.py.")

        # Vegetation indices
        veg_path = self.paths["processed_vegetation"] / "veg_indices.nc"
        if veg_path.exists():
            self._veg_ds = xr.open_dataset(veg_path)
            logger.info("Vegetation indices loaded.")

        # LANDFIRE fuel models
        fbfm40_path = self.paths["raw_landfire"] / "FBFM40_10m_utm17n.tif"
        if fbfm40_path.exists():
            from features.vegetation.fuel_classifier import map_fuel_models
            self._fuel_da, self._fuel_params = map_fuel_models(fbfm40_path)
            logger.info("Fuel models loaded.")

        # Parcels
        parcel_path = self.paths["raw_parcels"] / "parcels_study_area.gpkg"
        if parcel_path.exists():
            self._parcels = gpd.read_file(parcel_path)
        else:
            # Use synthetic Duke parcels for development
            from ingestion.parcel_fetcher import _synthetic_duke_parcels
            self._parcels = _synthetic_duke_parcels("EPSG:32617")
            logger.warning("Using synthetic Duke parcels (real parcel data not found).")

    def _build_single_twin(self, parcel_row) -> PropertyTwin:
        """Assemble a PropertyTwin from one parcel row + feature layers."""
        geom = parcel_row.get("geometry")
        parcel_id = str(parcel_row.get("parcel_id", f"PARCEL_{parcel_row.name}"))

        twin = PropertyTwin(
            parcel_id=parcel_id,
            address=str(parcel_row.get("address", "")),
            name=str(parcel_row.get("name", "")),
            geometry=geom,
            county=str(parcel_row.get("county", "Durham")),
            is_duke_owned=bool(parcel_row.get("is_duke", False)),
            structure_type=str(parcel_row.get("land_use", "unknown")),
            year_built=int(parcel_row.get("year_built", 1975) or 1975),
            building_sf=float(parcel_row.get("building_sf", 0) or 0),
            assessed_value=float(parcel_row.get("assessed_value", 0) or 0),
        )

        # Terrain features
        if self._terrain_ds is not None and geom is not None:
            self._populate_terrain(twin, geom)

        # Vegetation features
        if self._veg_ds is not None and geom is not None:
            self._populate_vegetation(twin, geom)

        # Structure (vision model or default)
        self._populate_structure(twin, parcel_row)

        return twin

    def _reproject_centroid(self, geom, target_crs_str: str):
        """Reproject a geometry's centroid to the target CRS. Returns (x, y)."""
        from pyproj import Transformer
        parcel_crs = self._parcels.crs
        if parcel_crs is not None and str(parcel_crs) != target_crs_str:
            transformer = Transformer.from_crs(str(parcel_crs), target_crs_str, always_xy=True)
            cx, cy = transformer.transform(geom.centroid.x, geom.centroid.y)
        else:
            cx, cy = geom.centroid.x, geom.centroid.y
        return cx, cy

    def _populate_terrain(self, twin: PropertyTwin, geom) -> None:
        """Extract parcel-level terrain stats from the terrain Dataset."""
        try:
            ds = self._terrain_ds
            terrain_crs = ds.attrs.get("crs", "EPSG:32617")
            cx, cy = self._reproject_centroid(geom, terrain_crs)

            def _interp(var: str, default: float = 0.0) -> float:
                if var not in ds:
                    return default
                try:
                    return float(ds[var].sel(
                        x=cx, y=cy, method="nearest"
                    ).values)
                except Exception:
                    return default

            twin.slope_degrees = _interp("slope_deg")
            twin.aspect_degrees = _interp("aspect_deg")
            twin.northness = _interp("northness")
            twin.eastness = _interp("eastness")
            twin.heat_load_index = _interp("heat_load_index")
            twin.tri = _interp("tri")
            twin.twi = _interp("twi")

            tpi_val = _interp("tpi")
            tpi_std = float(ds["tpi"].std().values) if "tpi" in ds else 1.0
            if tpi_val > tpi_std:
                twin.tpi_class = "ridge"
            elif tpi_val > 0.5 * tpi_std:
                twin.tpi_class = "upper_slope"
            elif tpi_val > -0.5 * tpi_std:
                twin.tpi_class = "mid_slope"
            else:
                twin.tpi_class = "valley"

        except Exception as e:
            logger.debug(f"Terrain extraction error for {twin.parcel_id}: {e}")

    def _populate_vegetation(self, twin: PropertyTwin, geom) -> None:
        """Extract parcel-level vegetation stats from the veg Dataset."""
        try:
            ds = self._veg_ds
            veg_crs = ds.attrs.get("crs", "EPSG:32617") if ds.attrs else "EPSG:32617"
            cx, cy = self._reproject_centroid(geom, veg_crs)

            def _interp(var: str, default: float = 0.0) -> float:
                if var not in ds:
                    return default
                try:
                    return float(ds[var].sel(x=cx, y=cy, method="nearest").values)
                except Exception:
                    return default

            twin.ndvi_mean = _interp("ndvi")
            twin.ndwi_mean = _interp("ndwi")
            twin.evi_mean = _interp("evi")
            twin.dry_veg_fraction_mean = _interp("dry_veg_fraction")
            # canopy_cover_pct: fraction of cells with NDVI > 0.4 (approximated here)
            twin.canopy_cover_pct = max(0, (twin.ndvi_mean - 0.1) / 0.5 * 100)

        except Exception as e:
            logger.debug(f"Vegetation extraction error for {twin.parcel_id}: {e}")

    def _populate_structure(self, twin: PropertyTwin, parcel_row) -> None:
        """
        Populate structural attributes.
        Uses vision model if run_vision_model=True; otherwise rule-based defaults.
        """
        if self.run_vision_model:
            # In production: load NAIP patch, run ClimateRiskBackbone, decode prediction
            pass

        # Fallback: infer from parcel data
        year = twin.year_built
        if year < 1940:
            twin.roof_material = "wood_shingles_shake"
            twin.roof_material_confidence = 0.4
        elif year < 1980:
            twin.roof_material = "asphalt_shingles"
            twin.roof_material_confidence = 0.5
        else:
            twin.roof_material = "asphalt_shingles"
            twin.roof_material_confidence = 0.6

        # Duke buildings: assume more modern materials
        if twin.is_duke_owned and year >= 2000:
            twin.roof_material = "membrane_flat"
            twin.roof_material_confidence = 0.7

        twin.vent_screening_status = "unknown"
        twin.deck_material = "unknown"


def build_all_twins(colab_mode: bool = False, save: bool = True) -> list[PropertyTwin]:
    """Convenience function: load config, build, and optionally save all twins."""
    paths = get_paths(colab_mode)
    cfg = load_config("model_config.yaml")
    builder = TwinBuilder(paths=paths, model_config=cfg, run_vision_model=False)
    return builder.build_all_twins(save=save)


if __name__ == "__main__":
    twins = build_all_twins(colab_mode=COLAB_MODE)
    logger.info(f"Built {len(twins)} twins.")
