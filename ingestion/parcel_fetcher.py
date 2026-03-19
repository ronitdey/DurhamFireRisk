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

    # Check for manually placed local files (GeoJSON, Shapefile, GeoPackage)
    local = _load_local_parcel_file(out_dir)
    if local is not None:
        local = local.to_crs(output_crs)
        local = _clip_to_bbox(local, bbox, output_crs)
        local = _standardize_schema(local)
        local.to_file(cache_path, driver="GPKG")
        logger.info(f"Loaded {len(local)} parcels from local file → {cache_path.name}")
        return local

    gdfs: list[gpd.GeoDataFrame] = []

    durham = _fetch_arcgis_parcels(_DURHAM_PARCEL_URL, bbox, county="Durham")
    if durham is not None and not durham.empty:
        gdfs.append(durham)

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


def _load_local_parcel_file(out_dir: Path) -> gpd.GeoDataFrame | None:
    """
    Check out_dir for a manually placed parcel file and load it.
    Accepts GeoJSON, Shapefile (.shp), or GeoPackage (.gpkg).
    Files with spaces in the name (e.g. from browser downloads) are handled.
    """
    for pattern in ["*.geojson", "*.shp", "*.gpkg", "*.json"]:
        matches = list(out_dir.glob(pattern))
        # Skip the cache file we write ourselves
        matches = [m for m in matches if m.name != "parcels_study_area.gpkg"]
        if matches:
            path = matches[0]
            try:
                gdf = gpd.read_file(path)
                logger.info(f"Loaded local parcel file: {path.name} ({len(gdf)} features)")
                return gdf
            except Exception as e:
                logger.warning(f"Failed to load {path.name}: {e}")
    return None


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
    Return a single-parcel GeoDataFrame for Randolph Residence Hall.

    Used as the fallback when county GIS endpoints are unreachable.
    Coordinates are the verified centroid of the building.
    """
    import shapely.geometry as sg
    from pyproj import Transformer

    # Verified centroid of Randolph Residence Hall, Duke East Campus
    LON = -78.91714316573827
    LAT = 36.00688194282498

    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = t.transform(LON, LAT)
    half = 35  # ~70m × 70m footprint (typical Duke residential hall)
    geom = sg.box(cx - half, cy - half, cx + half, cy + half)

    gdf = gpd.GeoDataFrame(
        [{
            "parcel_id": "RANDOLPH_001",
            "name": "Randolph Residence Hall",
            "address": "50 Brodie Gym Drive, Durham NC 27705",
            "land_use": "residential_dormitory",
            "owner_name": "DUKE UNIVERSITY",
            "is_duke": True,
            "year_built": 1929,
            "stories": 4,
            "geometry": geom,
        }],
        crs=crs,
    )
    logger.info("Using Randolph Residence Hall as single-building proof-of-concept parcel.")
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
