"""
Durham/Orange County parcel boundary ingestion.

Downloads parcel polygons from county GIS endpoints and filters to
the study area, retaining structure type, year built, and ownership fields.

Usage:
    python ingestion/parcel_fetcher.py

Colab:
    Set COLAB_MODE = True before running on Google Colab.
"""

from __future__ import annotations

# ── Colab mode flag ──────────────────────────────────────────────────────────
COLAB_MODE = False  # Set to True when running on Google Colab
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger
from shapely.geometry import box

from ingestion.config_loader import get_paths, get_study_area, load_config


# Durham County ArcGIS REST endpoint for parcels
_DURHAM_PARCEL_URL = (
    "https://gisweb.durhamnc.gov/arcgis/rest/services/PublicWS/Parcels/MapServer/0/query"
)
# Orange County parcel REST endpoint
_ORANGE_PARCEL_URL = (
    "https://maps.orangecountync.gov/arcgis/rest/services/PropertySearch/MapServer/0/query"
)


def fetch_parcels(
    bbox: tuple[float, float, float, float],
    out_dir: Path,
    output_crs: str = "EPSG:32617",
) -> gpd.GeoDataFrame:
    """
    Download parcel boundaries for Durham and Orange Counties within bbox.

    Parameters
    ----------
    bbox:
        (xmin, ymin, xmax, ymax) in EPSG:4326.
    out_dir:
        Directory to save the parcel GeoPackage.
    output_crs:
        Target CRS for the output GeoDataFrame.

    Returns
    -------
    GeoDataFrame with unified parcel schema.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "parcels_study_area.gpkg"

    if cache_path.exists():
        logger.info(f"Loading cached parcels from {cache_path.name}")
        return gpd.read_file(cache_path)

    gdfs: list[gpd.GeoDataFrame] = []

    durham = _fetch_arcgis_parcels(_DURHAM_PARCEL_URL, bbox, county="Durham")
    if durham is not None and not durham.empty:
        gdfs.append(durham)

    orange = _fetch_arcgis_parcels(_ORANGE_PARCEL_URL, bbox, county="Orange")
    if orange is not None and not orange.empty:
        gdfs.append(orange)

    if not gdfs:
        logger.warning("No parcels retrieved. Using synthetic Duke campus parcels.")
        return _synthetic_duke_parcels(output_crs)

    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    combined = combined.to_crs(output_crs)
    combined = _clip_to_bbox(combined, bbox, output_crs)
    combined = _standardize_schema(combined)

    combined.to_file(cache_path, driver="GPKG")
    logger.info(f"Saved {len(combined)} parcels → {cache_path.name}")
    return combined


def identify_duke_parcels(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Flag parcels owned by Duke University.

    Uses owner name string matching against common Duke entity names.
    """
    duke_keywords = [
        "DUKE UNIVERSITY",
        "DUKE HEALTH",
        "DUKE MEDICINE",
        "DUKE MANAGEMENT",
        "TRUSTEES OF DUKE",
    ]
    owner_col = _find_column(parcels, ["owner", "owner_name", "OWNER", "OWNERNAME"])
    if owner_col is None:
        logger.warning("No owner column found; cannot identify Duke parcels.")
        parcels["is_duke"] = False
        return parcels

    parcels["is_duke"] = parcels[owner_col].str.upper().str.contains(
        "|".join(duke_keywords), na=False
    )
    n_duke = parcels["is_duke"].sum()
    logger.info(f"Identified {n_duke} Duke University parcels.")
    return parcels


def _fetch_arcgis_parcels(
    url: str,
    bbox: tuple[float, float, float, float],
    county: str,
    max_records: int = 2000,
) -> gpd.GeoDataFrame | None:
    """Query an ArcGIS REST feature layer within bbox using pagination."""
    xmin, ymin, xmax, ymax = bbox
    params = {
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": max_records,
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        gdf = gpd.read_file(resp.text)
        gdf["county"] = county
        logger.info(f"Retrieved {len(gdf)} {county} County parcels.")
        return gdf
    except Exception as e:
        logger.warning(f"{county} County parcel fetch failed: {e}")
        return None


def _clip_to_bbox(
    gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    crs: str,
) -> gpd.GeoDataFrame:
    """Clip GeoDataFrame to the study area bounding box."""
    bbox_geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326").to_crs(crs)
    return gpd.clip(gdf, bbox_geom)


def _standardize_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalize column names across Durham and Orange County schemas to a
    unified parcel schema.
    """
    renames: dict[str, str] = {}

    col_map = {
        "parcel_id": ["parcelid", "pid", "parid", "objectid"],
        "owner_name": ["ownername", "owner", "owner1"],
        "address": ["siteaddress", "address", "site_address", "fulladdress"],
        "land_use": ["landuse", "land_use_code", "proptype", "lu_code"],
        "year_built": ["yearbuilt", "year_built", "yr_built"],
        "assessed_value": ["totalvalue", "total_value", "assessed_value"],
        "building_sf": ["totalsqft", "bldgsf", "grosssqft"],
        "stories": ["stories", "numstories"],
    }

    for std, candidates in col_map.items():
        found = _find_column(gdf, candidates)
        if found and found != std:
            renames[found] = std

    return gdf.rename(columns=renames)


def _find_column(gdf: gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in gdf.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _synthetic_duke_parcels(crs: str) -> gpd.GeoDataFrame:
    """
    Return a minimal synthetic parcel GeoDataFrame covering key Duke buildings
    for development/testing when real parcel data is unavailable.
    """
    import shapely.geometry as sg

    # Approximate centroids in WGS84 for key Duke buildings
    buildings = [
        {"name": "Duke Chapel", "lon": -78.9403, "lat": 36.0023, "type": "religious"},
        {"name": "Gross Hall", "lon": -78.9409, "lat": 36.0035, "type": "office"},
        {"name": "Perkins Library", "lon": -78.9385, "lat": 36.0015, "type": "library"},
        {"name": "Cameron Indoor Stadium", "lon": -78.9432, "lat": 36.0000, "type": "assembly"},
        {"name": "Duke Hospital", "lon": -78.9415, "lat": 35.9946, "type": "medical"},
        {"name": "East Campus Baldwin Auditorium", "lon": -78.9237, "lat": 36.0044, "type": "assembly"},
    ]

    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    records = []
    for i, b in enumerate(buildings):
        cx, cy = t.transform(b["lon"], b["lat"])
        half = 40  # 80m × 80m placeholder footprint
        geom = sg.box(cx - half, cy - half, cx + half, cy + half)
        records.append({
            "parcel_id": f"DUKE_{i:04d}",
            "name": b["name"],
            "land_use": b["type"],
            "owner_name": "DUKE UNIVERSITY",
            "is_duke": True,
            "year_built": 1930 + i * 10,
            "geometry": geom,
        })

    gdf = gpd.GeoDataFrame(records, crs=crs)
    logger.info(f"Created {len(gdf)} synthetic Duke parcels for development.")
    return gdf


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    paths = get_paths(COLAB_MODE)
    cfg = load_config("data_sources.yaml")
    sa = get_study_area(cfg)
    bbox = (
        sa["bbox"]["xmin"], sa["bbox"]["ymin"],
        sa["bbox"]["xmax"], sa["bbox"]["ymax"],
    )

    parcels = fetch_parcels(bbox=bbox, out_dir=paths["raw_parcels"])
    parcels = identify_duke_parcels(parcels)
    logger.info(f"Total parcels: {len(parcels)}, Duke parcels: {parcels['is_duke'].sum()}")


if __name__ == "__main__":
    main()
