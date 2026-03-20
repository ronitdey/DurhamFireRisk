"""
Interactive campus risk map using Folium/Leaflet.js.

Generates a standalone HTML map with:
    - ESRI satellite imagery base layer
    - Fire spread isochrones from Rothermel simulation
    - Fireline intensity heatmap
    - Building-level risk choropleth with detailed popups
    - Defensible space analysis rings (30 ft / 100 ft)
    - Toggleable layer controls and professional legend

Usage:
    python visualization/risk_map.py

Colab:
    from visualization.risk_map import build_risk_map, twins_to_geodataframe
    m = build_risk_map(twins_gdf, output_path, paths=result["paths"])
    m  # renders inline
"""

from __future__ import annotations

import json
from pathlib import Path

import folium
import folium.plugins as plugins
import geopandas as gpd
import numpy as np
from branca.colormap import LinearColormap
from loguru import logger


# Fire arrival time contour styling: (minutes, color, weight, dash_array)
_ISOCHRONE_STYLE = [
    (5,  "#ff3333", 3, None),
    (10, "#ff7733", 3, None),
    (20, "#ffcc00", 2, None),
    (30, "#33aaff", 2, "8,6"),
    (60, "#cc77ff", 2, "8,6"),
    (90, "#aaaaaa", 1, "4,4"),
]


def build_risk_map(
    twins_gdf: gpd.GeoDataFrame,
    output_path: Path,
    paths: dict | None = None,
    center_lat: float = 36.0069,
    center_lon: float = -78.9171,
    zoom: int = 17,
) -> folium.Map:
    """
    Build an interactive Folium risk map with satellite imagery,
    fire spread isochrones, and per-building risk overlays.

    Parameters
    ----------
    twins_gdf:
        GeoDataFrame from twins_to_geodataframe(), in EPSG:4326.
    output_path:
        Where to save the standalone HTML file.
    paths:
        Path dict from get_paths() — needed for fire simulation overlay.
    center_lat, center_lon:
        Fallback map center (used when twins_gdf is empty).
    zoom:
        Initial zoom level.

    Returns
    -------
    Folium Map object (also saved to output_path).
    """
    # Auto-center on data
    if not twins_gdf.empty:
        bounds = twins_gdf.total_bounds
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None)

    # ── Base tiles ────────────────────────────────────────────────────────────
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite",
        overlay=False,
    ).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark", overlay=False).add_to(m)
    folium.TileLayer("CartoDB positron", name="Light", overlay=False).add_to(m)

    # Risk colormap
    colormap = LinearColormap(
        colors=["#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"],
        vmin=0, vmax=100,
        caption="Wildfire Risk Score",
    )
    colormap.add_to(m)

    # ── Overlay layers ────────────────────────────────────────────────────────
    if paths:
        _add_fire_spread_layer(m, paths)
    _add_building_risk_layer(m, twins_gdf, colormap)
    _add_defensible_space_layer(m, twins_gdf)

    # ── Controls & plugins ────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MiniMap(tile_layer="CartoDB positron", toggle_display=True).add_to(m)
    _add_legend(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info(f"Risk map saved → {output_path}")
    return m


# ── Layer builders ────────────────────────────────────────────────────────────

def _add_fire_spread_layer(m: folium.Map, paths: dict) -> None:
    """Add fire spread isochrones and intensity heatmap from simulation."""
    import xarray as xr
    from pyproj import Transformer

    sim_path = paths.get("processed_terrain", Path()) / "fire_simulation.nc"
    if not sim_path.exists():
        logger.info("No fire simulation data — skipping isochrones.")
        return

    try:
        ds = xr.open_dataset(sim_path)
        toa = ds["time_of_arrival"].values.copy()
        xs = ds.x.values.copy()
        ys = ds.y.values.copy()
        intensity = (
            ds["fireline_intensity"].values.copy()
            if "fireline_intensity" in ds
            else None
        )
        crs = ds.attrs.get("crs", "EPSG:32617")
        ds.close()
    except Exception as e:
        logger.warning(f"Failed to load simulation: {e}")
        return

    # Sanity-check: xs/ys must be real coordinates, not bare indices
    if xs.max() < 1000:
        logger.warning("Simulation has no georeferenced coords — skipping overlay.")
        return

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # ── Isochrone contours ────────────────────────────────────────────────
    iso_layer = folium.FeatureGroup(name="Fire Arrival Isochrones", show=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    toa_filled = np.nan_to_num(toa, nan=9999.0)
    valid_toa = toa[~np.isnan(toa)]
    max_toa = float(valid_toa.max()) if valid_toa.size > 0 else 0
    n_burned = int((~np.isnan(toa)).sum())
    logger.info(
        f"Simulation: {n_burned}/{toa.size} burned cells, "
        f"max arrival={max_toa:.0f}min, x=[{xs.min():.0f},{xs.max():.0f}], "
        f"y=[{ys.min():.0f},{ys.max():.0f}]"
    )

    n_lines = 0
    fig, ax = plt.subplots()
    for time_min, color, weight, dash in _ISOCHRONE_STYLE:
        if max_toa < 1 or time_min > max_toa * 1.5:
            continue
        try:
            cs = ax.contour(xs, ys, toa_filled, levels=[time_min])
            # matplotlib 3.8+ removed cs.collections; use cs.allsegs instead
            segments = cs.allsegs[0] if hasattr(cs, "allsegs") else []
            if not segments and hasattr(cs, "collections"):
                # Fallback for older matplotlib
                for collection in cs.collections:
                    for path in collection.get_paths():
                        segments.append(path.vertices)
            for verts in segments:
                if len(verts) < 3:
                    continue
                lons, lats = transformer.transform(verts[:, 0], verts[:, 1])
                coords = list(zip(lats.tolist(), lons.tolist()))
                kw = {
                    "color": color,
                    "weight": weight,
                    "opacity": 0.85,
                    "tooltip": f"Fire arrives in {time_min} min",
                }
                if dash:
                    kw["dash_array"] = dash
                folium.PolyLine(coords, **kw).add_to(iso_layer)
                n_lines += 1
        except Exception as e:
            logger.warning(f"Contour extraction failed at {time_min}min: {e}")
        ax.clear()
    plt.close(fig)
    iso_layer.add_to(m)
    logger.info(f"Added {n_lines} isochrone polylines.")

    # ── Fireline intensity heatmap ────────────────────────────────────────
    if intensity is not None:
        heat_layer = folium.FeatureGroup(name="Fireline Intensity Heatmap", show=False)
        step = max(1, min(toa.shape) // 80)
        heat_data = []
        for r in range(0, toa.shape[0], step):
            for c in range(0, toa.shape[1], step):
                if not np.isnan(toa[r, c]) and intensity[r, c] > 0:
                    lon, lat = transformer.transform(float(xs[c]), float(ys[r]))
                    heat_data.append([lat, lon, float(intensity[r, c])])
        if heat_data:
            plugins.HeatMap(
                heat_data,
                min_opacity=0.3,
                radius=12,
                blur=10,
                gradient={
                    0.2: "#ffffb2", 0.4: "#fecc5c",
                    0.6: "#fd8d3c", 0.8: "#e31a1c", 1.0: "#800026",
                },
            ).add_to(heat_layer)
        heat_layer.add_to(m)

    logger.info(f"Fire spread overlay added ({max_toa:.0f} min max arrival).")


def _add_building_risk_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    colormap: LinearColormap,
) -> None:
    """Add building polygons colored by risk score with detailed popups."""
    risk_layer = folium.FeatureGroup(name="Building Risk Scores", show=True)

    for _, row in gdf.iterrows():
        geom = row.get("geometry")
        if geom is None:
            continue

        score = _sf(row.get("wildfire_risk_score"))
        color = colormap(score)
        name = str(
            row.get("name")
            or row.get("address")
            or row.get("parcel_id", "")
        )

        popup_html = _build_popup(row, score)

        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda feat, c=color: {
                "fillColor": c,
                "color": "#ffffff",
                "weight": 2,
                "fillOpacity": 0.7,
            },
            tooltip=folium.Tooltip(
                f"<b>{name}</b><br>Risk: {score:.0f}/100", sticky=True,
            ),
            popup=folium.Popup(popup_html, max_width=340),
        ).add_to(risk_layer)

    risk_layer.add_to(m)


def _add_defensible_space_layer(m: folium.Map, gdf: gpd.GeoDataFrame) -> None:
    """Show 30 ft and 100 ft defensible space rings around each building."""
    ds_layer = folium.FeatureGroup(name="Defensible Space Zones", show=False)

    for _, row in gdf.iterrows():
        geom = row.get("geometry")
        if geom is None:
            continue
        cx, cy = geom.centroid.x, geom.centroid.y

        # 100 ft = 30.5 m
        folium.Circle(
            location=[cy, cx],
            radius=30.5,
            color="#f39c12",
            fill=False,
            weight=2,
            dash_array="10,5",
            tooltip="100 ft defensible space",
        ).add_to(ds_layer)

        # 30 ft = 9.1 m
        folium.Circle(
            location=[cy, cx],
            radius=9.1,
            color="#e74c3c",
            fill=False,
            weight=2,
            tooltip="30 ft immediate zone",
        ).add_to(ds_layer)

    ds_layer.add_to(m)


def _add_duke_boundary_layer(m: folium.Map, gdf: gpd.GeoDataFrame) -> None:
    """Dashed Duke Blue outline around Duke-owned parcels."""
    if "is_duke_owned" not in gdf.columns:
        return
    duke = gdf[gdf["is_duke_owned"] == True]  # noqa: E712
    if duke.empty:
        return

    duke_layer = folium.FeatureGroup(name="Duke-Owned Parcels", show=True)
    for _, row in duke.iterrows():
        geom = row.get("geometry")
        if geom is None:
            continue
        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda feat: {
                "fillColor": "none",
                "color": "#003087",
                "weight": 3,
                "dashArray": "6,4",
            },
        ).add_to(duke_layer)
    duke_layer.add_to(m)


# ── Popup & risk breakdown ────────────────────────────────────────────────────

def _sf(val, default: float = 0.0) -> float:
    """Safe float cast handling None / NaN."""
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _risk_breakdown(row) -> dict:
    """Decompose score using the same logic as WildfireScorer._fallback_score."""
    from models.risk.wildfire_scorer import ROOF_MATERIAL_MAP, TPI_CLASS_MAP

    slope = _sf(row.get("slope_degrees"))
    hli = _sf(row.get("heat_load_index"))
    tpi = TPI_CLASS_MAP.get(str(row.get("tpi_class", "mid_slope")), 1)
    terrain = min(slope * 0.5, 15) + hli * 5 + tpi * 2

    z1 = _sf(row.get("zone1_fuel_load"))
    z2 = _sf(row.get("zone2_fuel_load"))
    z3 = _sf(row.get("zone3_fuel_load"))
    ladder = float(row.get("ladder_fuel_present", False) or False)
    vegetation = min(z1 * 20, 12) + min(z2 * 5, 10) + min(z3 * 2, 5) + ladder * 8

    roof_enc = ROOF_MATERIAL_MAP.get(str(row.get("roof_material", "unknown_occluded")), 2)
    roof_pts = {0: 0, 1: 5, 2: 12, 3: 20}.get(roof_enc, 10)
    vent = 1.0 if row.get("vent_screening_status") == "screened" else 0.0
    structure = roof_pts + (1 - vent) * 12

    nbr = _sf(row.get("neighbor_distance_m"), 100)
    ember = _sf(row.get("ember_exposure_probability"))
    exposure = (max(0, 10 - nbr * 0.5) if nbr < 20 else 0) + ember * 5

    return {
        "terrain": round(terrain, 1),
        "vegetation": round(vegetation, 1),
        "structure": round(structure, 1),
        "exposure": round(exposure, 1),
    }


def _build_popup(row, score: float) -> str:
    """Rich HTML popup with risk badge, breakdown bars, and simulation results."""
    bd = _risk_breakdown(row)
    name = str(row.get("name") or row.get("address") or row.get("parcel_id", "Unknown"))
    address = str(row.get("address", ""))

    if score < 30:
        label, bg = "LOW", "#27ae60"
    elif score < 55:
        label, bg = "MODERATE", "#f39c12"
    elif score < 75:
        label, bg = "HIGH", "#e74c3c"
    else:
        label, bg = "VERY HIGH", "#8e44ad"

    roof = str(row.get("roof_material", "unknown")).replace("_", " ").title()
    year = int(row.get("year_built", 0) or 0)
    vent = str(row.get("vent_screening_status", "unknown")).title()
    slope = _sf(row.get("slope_degrees"))

    fire_time = _sf(row.get("fire_arrival_time_p50"), float("inf"))
    ember = _sf(row.get("ember_exposure_probability"))

    fire_html = ""
    if fire_time < float("inf"):
        fire_html = (
            '<div style="background:#fef5e7;padding:8px 10px;border-radius:6px;'
            'margin:10px 0;border-left:3px solid #e67e22;">'
            '<b style="font-size:12px;">Fire Simulation (Worst Case)</b><br>'
            f'<span style="font-size:11px;">Arrival: <b>{fire_time:.0f} min</b>'
            f' &nbsp;|&nbsp; Ember exposure: <b>{ember:.0%}</b></span></div>'
        )

    def bar(lbl, val, mx, clr):
        pct = min(val / mx * 100, 100) if mx > 0 else 0
        return (
            '<div style="display:flex;align-items:center;margin:3px 0;">'
            f'<span style="width:72px;font-size:11px;color:#555;">{lbl}</span>'
            '<div style="flex:1;background:#ecf0f1;border-radius:3px;height:12px;overflow:hidden;">'
            f'<div style="width:{pct:.0f}%;background:{clr};height:100%;border-radius:3px;"></div>'
            f'</div><span style="width:48px;text-align:right;font-size:10px;color:#777;">'
            f'{val:.0f}/{mx}</span></div>'
        )

    bars = (
        bar("Terrain", bd["terrain"], 25, "#3498db")
        + bar("Vegetation", bd["vegetation"], 30, "#2ecc71")
        + bar("Structure", bd["structure"], 35, "#e74c3c")
        + bar("Exposure", bd["exposure"], 10, "#9b59b6")
    )

    return (
        '<div style="font-family:\'Helvetica Neue\',Arial,sans-serif;width:300px;">'
        f'<h4 style="margin:0 0 2px;color:#2c3e50;font-size:14px;">{name}</h4>'
        f'<p style="margin:0 0 8px;color:#95a5a6;font-size:11px;">{address}</p>'
        f'<div style="background:{bg};color:white;padding:5px 14px;'
        'border-radius:16px;display:inline-block;font-weight:600;'
        f'font-size:14px;margin-bottom:10px;">{score:.0f}/100 — {label}</div>'
        f'<div style="margin:8px 0;">{bars}</div>'
        f'{fire_html}'
        '<table style="width:100%;font-size:11px;border-collapse:collapse;color:#555;">'
        f'<tr><td style="padding:2px 0;"><b>Roof</b></td><td>{roof}</td></tr>'
        f'<tr><td style="padding:2px 0;"><b>Year Built</b></td>'
        f'<td>{year if year > 0 else "Unknown"}</td></tr>'
        f'<tr><td style="padding:2px 0;"><b>Vents</b></td><td>{vent}</td></tr>'
        f'<tr><td style="padding:2px 0;"><b>Slope</b></td><td>{slope:.1f}&deg;</td></tr>'
        '</table></div>'
    )


# ── Legend ────────────────────────────────────────────────────────────────────

def _add_legend(m: folium.Map) -> None:
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(255,255,255,0.95);padding:14px 18px;border-radius:10px;
                box-shadow:0 2px 12px rgba(0,0,0,0.15);font-family:'Helvetica Neue',Arial;
                max-width:200px;font-size:12px;">
        <b style="font-size:13px;">Risk Levels</b>
        <div style="margin:6px 0 8px;">
            <span style="display:inline-block;width:12px;height:12px;background:#27ae60;
                         border-radius:2px;vertical-align:middle;"></span>
            <span style="vertical-align:middle;margin-left:4px;">Low (0–30)</span><br>
            <span style="display:inline-block;width:12px;height:12px;background:#f39c12;
                         border-radius:2px;vertical-align:middle;"></span>
            <span style="vertical-align:middle;margin-left:4px;">Moderate (30–55)</span><br>
            <span style="display:inline-block;width:12px;height:12px;background:#e74c3c;
                         border-radius:2px;vertical-align:middle;"></span>
            <span style="vertical-align:middle;margin-left:4px;">High (55–75)</span><br>
            <span style="display:inline-block;width:12px;height:12px;background:#8e44ad;
                         border-radius:2px;vertical-align:middle;"></span>
            <span style="vertical-align:middle;margin-left:4px;">Very High (75+)</span>
        </div>
        <hr style="border:none;border-top:1px solid #ddd;margin:6px 0;">
        <b style="font-size:13px;">Fire Isochrones</b>
        <div style="margin:6px 0 8px;">
            <span style="display:inline-block;width:18px;border-top:3px solid #ff3333;
                         vertical-align:middle;"></span>
            <span style="vertical-align:middle;margin-left:4px;">5 min</span>&emsp;
            <span style="display:inline-block;width:18px;border-top:3px solid #ff7733;
                         vertical-align:middle;"></span>
            <span style="vertical-align:middle;margin-left:4px;">10 min</span><br>
            <span style="display:inline-block;width:18px;border-top:2px solid #ffcc00;
                         vertical-align:middle;"></span>
            <span style="vertical-align:middle;margin-left:4px;">20 min</span>&emsp;
            <span style="display:inline-block;width:18px;border-top:2px dashed #33aaff;
                         vertical-align:middle;"></span>
            <span style="vertical-align:middle;margin-left:4px;">30 min</span>
        </div>
        <hr style="border:none;border-top:1px solid #ddd;margin:6px 0;">
        <b style="font-size:13px;">Defensible Space</b>
        <div style="margin:6px 0;">
            <span style="display:inline-block;width:12px;height:12px;border:2px solid #e74c3c;
                         border-radius:50%;vertical-align:middle;"></span>
            <span style="vertical-align:middle;margin-left:4px;">30 ft zone</span><br>
            <span style="display:inline-block;width:12px;height:12px;border:2px dashed #f39c12;
                         border-radius:50%;vertical-align:middle;"></span>
            <span style="vertical-align:middle;margin-left:4px;">100 ft zone</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# ── GeoDataFrame conversion ──────────────────────────────────────────────────

def twins_to_geodataframe(
    twins: list,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
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
            geom = box(-78.94, 35.99, -78.93, 36.00)
        records.append({
            "parcel_id": twin.parcel_id,
            "name": twin.name,
            "address": twin.address,
            "geometry": geom,
            # Risk
            "wildfire_risk_score": twin.wildfire_risk_score,
            "composite_risk_score": twin.composite_risk_score,
            # Terrain
            "slope_degrees": twin.slope_degrees,
            "aspect_degrees": twin.aspect_degrees,
            "heat_load_index": twin.heat_load_index,
            "tpi_class": twin.tpi_class,
            "twi": twin.twi,
            # Vegetation
            "zone1_fuel_load": twin.zone1_fuel_load,
            "zone2_fuel_load": twin.zone2_fuel_load,
            "zone3_fuel_load": twin.zone3_fuel_load,
            "ladder_fuel_present": twin.ladder_fuel_present,
            "ndvi_mean": twin.ndvi_mean,
            "canopy_cover_pct": twin.canopy_cover_pct,
            # Structure
            "roof_material": twin.roof_material,
            "roof_material_confidence": twin.roof_material_confidence,
            "year_built": twin.year_built,
            "structure_type": twin.structure_type,
            "vent_screening_status": twin.vent_screening_status,
            "building_sf": twin.building_sf,
            # Exposure
            "neighbor_distance_m": twin.neighbor_distance_m,
            "neighbor_flag_15m": twin.neighbor_flag_15m,
            "ember_exposure_probability": twin.ember_exposure_probability,
            # Simulation
            "fire_arrival_time_p50": twin.fire_arrival_time_p50,
            "fire_arrival_time_p90": twin.fire_arrival_time_p90,
            # Ownership
            "is_duke_owned": twin.is_duke_owned,
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:32617")
    return gdf.to_crs(crs)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion.config_loader import get_paths
    from twin.property_twin import PropertyTwin

    paths = get_paths(colab_mode=False)
    twin_dir = paths["processed_twins"]
    twins = (
        [PropertyTwin.load(f) for f in twin_dir.glob("*.json")]
        if twin_dir.exists()
        else []
    )

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
    build_risk_map(gdf, out, paths=paths)
    logger.info(f"Open {out} in a browser to view the map.")
