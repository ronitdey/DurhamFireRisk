"""
Interactive campus risk map using Folium/Leaflet.js.

Generates a standalone HTML map with toggleable layers:
    - Parcel risk choropleth (green → yellow → red)
    - LANDFIRE fuel load overlay
    - Fire spread simulation isochrones (animated)
    - Clickable SHAP waterfall on parcel selection
    - Top mitigation recommendations per parcel

Usage:
    python visualization/risk_map.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import folium
import geopandas as gpd
import numpy as np
from branca.colormap import LinearColormap
from loguru import logger


def build_risk_map(
    twins_gdf: gpd.GeoDataFrame,
    output_path: Path,
    center_lat: float = 36.001,
    center_lon: float = -78.940,
    zoom: int = 14,
) -> folium.Map:
    """
    Build an interactive Folium map of property-level wildfire risk.

    Parameters
    ----------
    twins_gdf:
        GeoDataFrame with one row per property, including:
            geometry, parcel_id, wildfire_risk_score, composite_risk_score,
            roof_material, ladder_fuel_present, zone1_fuel_load, etc.
        Must be in EPSG:4326 for Folium (WGS84).
    output_path:
        Path to save the standalone HTML file.
    center_lat, center_lon:
        Map center (Duke Chapel coordinates by default).
    zoom:
        Initial zoom level.

    Returns
    -------
    Folium Map object (also saved to output_path).
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
    )

    # Base tile layers
    folium.TileLayer("CartoDB positron", name="Light Base").add_to(m)
    folium.TileLayer("Esri.WorldImagery", name="Satellite", attr="Esri").add_to(m)

    # Risk colormap
    colormap = LinearColormap(
        colors=["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"],
        vmin=0, vmax=100,
        caption="Wildfire Risk Score (0=Low, 100=Very High)",
    )
    colormap.add_to(m)

    # ── Layer 1: Risk choropleth ───────────────────────────────────────────
    risk_layer = folium.FeatureGroup(name="Wildfire Risk Scores", show=True)

    for _, row in twins_gdf.iterrows():
        geom = row.get("geometry")
        if geom is None:
            continue

        score = float(row.get("wildfire_risk_score", 0) or 0)
        color = colormap(score)
        parcel_id = str(row.get("parcel_id", ""))
        name = str(row.get("name", row.get("address", parcel_id)))

        # Popup HTML
        popup_html = _build_popup(row, score)

        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda feat, c=color, s=score: {
                "fillColor": c,
                "color": "#2c3e50" if s > 55 else "#7f8c8d",
                "weight": 2 if s > 75 else 1,
                "fillOpacity": 0.75,
            },
            tooltip=folium.Tooltip(f"{name}: {score:.0f}/100"),
            popup=folium.Popup(popup_html, max_width=350),
        ).add_to(risk_layer)

    risk_layer.add_to(m)

    # ── Layer 2: Ladder fuel flags ─────────────────────────────────────────
    ladder_layer = folium.FeatureGroup(name="Ladder Fuels", show=False)
    for _, row in twins_gdf.iterrows():
        if row.get("ladder_fuel_present"):
            geom = row.get("geometry")
            if geom:
                cx, cy = geom.centroid.x, geom.centroid.y
                folium.CircleMarker(
                    location=[cy, cx],
                    radius=6,
                    color="#e67e22",
                    fill=True,
                    fill_color="#e67e22",
                    fill_opacity=0.8,
                    tooltip="Ladder fuels detected",
                ).add_to(ladder_layer)
    ladder_layer.add_to(m)

    # ── Layer 3: High-risk neighbor flags ─────────────────────────────────
    neighbor_layer = folium.FeatureGroup(name="Structures < 15m Apart", show=False)
    for _, row in twins_gdf.iterrows():
        if row.get("neighbor_flag_15m"):
            geom = row.get("geometry")
            if geom:
                cx, cy = geom.centroid.x, geom.centroid.y
                folium.CircleMarker(
                    location=[cy, cx],
                    radius=5,
                    color="#c0392b",
                    fill=True,
                    fill_color="#c0392b",
                    fill_opacity=0.9,
                    tooltip=f"Adjacent structure {row.get('neighbor_distance_m', '?'):.0f}m away",
                ).add_to(neighbor_layer)
    neighbor_layer.add_to(m)

    # ── Layer 4: Duke campus boundary ─────────────────────────────────────
    duke_layer = folium.FeatureGroup(name="Duke-Owned Properties", show=True)
    for _, row in twins_gdf[twins_gdf.get("is_duke_owned", False)].iterrows():
        geom = row.get("geometry")
        if geom:
            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda feat: {
                    "fillColor": "none",
                    "color": "#003087",  # Duke Blue
                    "weight": 3,
                    "dashArray": "5, 5",
                },
            ).add_to(duke_layer)
    duke_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    _add_legend(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info(f"Risk map saved → {output_path}")
    return m


def _build_popup(row, score: float) -> str:
    """Build HTML popup for a parcel feature."""
    name = str(row.get("name", row.get("address", row.get("parcel_id", "Unknown"))))
    roof = str(row.get("roof_material", "unknown")).replace("_", " ").title()
    ladder = "Yes ⚠️" if row.get("ladder_fuel_present") else "No ✓"
    z1 = row.get("zone1_fuel_load", 0) or 0
    nbr = row.get("neighbor_distance_m", 100) or 100

    risk_color = "#2ecc71" if score < 30 else "#f39c12" if score < 55 else "#e74c3c"
    risk_label = "LOW" if score < 30 else "MODERATE" if score < 55 else "HIGH" if score < 75 else "VERY HIGH"

    return f"""
    <div style="font-family: Arial, sans-serif; min-width: 280px;">
        <h4 style="margin:0 0 8px; color:#2c3e50;">{name}</h4>
        <div style="background:{risk_color}; color:white; padding:4px 10px;
                    border-radius:12px; display:inline-block; font-weight:bold; margin-bottom:8px;">
            {score:.0f}/100 — {risk_label}
        </div>
        <table style="width:100%; font-size:12px;">
            <tr><td><b>Roof Material</b></td><td>{roof}</td></tr>
            <tr><td><b>Ladder Fuels</b></td><td>{ladder}</td></tr>
            <tr><td><b>Zone 1 Fuel Load</b></td><td>{z1:.2f} tons/acre</td></tr>
            <tr><td><b>Nearest Neighbor</b></td><td>{nbr:.0f}m</td></tr>
        </table>
        <div style="margin-top:8px; font-size:11px; color:#7f8c8d;">
            Click <a href="/explain/{row.get('parcel_id', '')}">here</a> for full SHAP attribution.
        </div>
    </div>
    """


def _add_legend(m: folium.Map) -> None:
    """Add a color legend to the map."""
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-family: Arial;">
        <b>Wildfire Risk</b><br>
        <span style="background:#2ecc71;padding:2px 8px;border-radius:3px;">LOW (0-30)</span><br>
        <span style="background:#f39c12;padding:2px 8px;border-radius:3px;color:white;">MODERATE (30-55)</span><br>
        <span style="background:#e74c3c;padding:2px 8px;border-radius:3px;color:white;">HIGH (55-75)</span><br>
        <span style="background:#8e44ad;padding:2px 8px;border-radius:3px;color:white;">VERY HIGH (75+)</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def twins_to_geodataframe(twins: list, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Convert a list of PropertyTwin objects to a GeoDataFrame for visualization.

    Parameters
    ----------
    twins:
        List of PropertyTwin objects (must have geometry set).
    crs:
        Target CRS (Folium requires EPSG:4326).
    """
    from shapely.geometry import box

    records = []
    for twin in twins:
        geom = twin.geometry
        if geom is None:
            # Placeholder geometry for twins without spatial data
            geom = box(-78.94, 35.99, -78.93, 36.00)
        records.append({
            "parcel_id": twin.parcel_id,
            "name": twin.name,
            "address": twin.address,
            "wildfire_risk_score": twin.wildfire_risk_score,
            "composite_risk_score": twin.composite_risk_score,
            "roof_material": twin.roof_material,
            "ladder_fuel_present": twin.ladder_fuel_present,
            "neighbor_flag_15m": twin.neighbor_flag_15m,
            "neighbor_distance_m": twin.neighbor_distance_m,
            "zone1_fuel_load": twin.zone1_fuel_load,
            "is_duke_owned": twin.is_duke_owned,
            "geometry": geom,
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:32617")
    return gdf.to_crs(crs)


if __name__ == "__main__":
    from ingestion.config_loader import get_paths
    from twin.property_twin import PropertyTwin

    paths = get_paths(colab_mode=False)
    twin_dir = paths["processed_twins"]
    twins = [PropertyTwin.load(f) for f in twin_dir.glob("*.json")] if twin_dir.exists() else []

    if not twins:
        logger.warning("No twins found. Building synthetic demo twins.")
        from ingestion.parcel_fetcher import _synthetic_duke_parcels
        parcels = _synthetic_duke_parcels("EPSG:32617")
        twins = [
            PropertyTwin(
                parcel_id=str(row["parcel_id"]),
                name=str(row["name"]),
                geometry=row["geometry"].centroid.buffer(30),
                is_duke_owned=True,
                wildfire_risk_score=float(np.random.uniform(25, 85)),
                composite_risk_score=float(np.random.uniform(25, 85)),
            )
            for _, row in parcels.iterrows()
        ]

    gdf = twins_to_geodataframe(twins)
    out = paths["processed"] / "campus_risk_map.html"
    build_risk_map(gdf, out)
    logger.info(f"Open {out} in a browser to view the map.")
