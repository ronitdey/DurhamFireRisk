"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import { BuildingProperties } from "@/lib/types";
import { riskColorExpression, fireColorExpression } from "@/lib/colors";

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || "";
const EAST_CAMPUS_CENTER: [number, number] = [-78.9158, 36.0072];

interface MapProps {
  onBuildingClick: (props: BuildingProperties | null) => void;
}

export default function Map({ onBuildingClick }: MapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [layers, setLayers] = useState({
    buildings: true,
    isochrones: true,
    intensity: false,
  });

  const initMap = useCallback(() => {
    if (!mapContainer.current || map.current) return;

    mapboxgl.accessToken = MAPBOX_TOKEN;
    console.log("[Map] Token:", MAPBOX_TOKEN ? `${MAPBOX_TOKEN.slice(0, 10)}...` : "MISSING");
    console.log("[Map] Container size:", mapContainer.current.clientWidth, "x", mapContainer.current.clientHeight);

    const m = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/satellite-streets-v12",
      center: EAST_CAMPUS_CENTER,
      zoom: 16.5,
      pitch: 45,
      bearing: -20,
      antialias: true,
    });

    m.addControl(new mapboxgl.NavigationControl(), "top-right");

    m.on("error", (e) => console.error("[Map] Error:", e.error?.message || e));

    m.on("load", () => {
      const canvas = mapContainer.current?.querySelector("canvas");
      console.log("[Map] Style loaded successfully");
      console.log("[Map] Canvas:", canvas ? `${canvas.width}x${canvas.height}` : "NOT FOUND");
      console.log("[Map] WebGL:", m.getCanvas().getContext("webgl2") ? "webgl2" : m.getCanvas().getContext("webgl") ? "webgl" : "NONE");
      // Buildings layer - 3D extruded polygons
      m.addSource("buildings", {
        type: "geojson",
        data: "/data/buildings.geojson",
      });

      m.addLayer({
        id: "buildings-3d",
        type: "fill-extrusion",
        source: "buildings",
        paint: {
          "fill-extrusion-color": riskColorExpression,
          "fill-extrusion-height": [
            "*",
            ["coalesce", ["get", "stories"], 1],
            4,
          ],
          "fill-extrusion-base": 0,
          "fill-extrusion-opacity": 0.85,
        },
      });

      // Building outlines (flat)
      m.addLayer({
        id: "buildings-outline",
        type: "line",
        source: "buildings",
        paint: {
          "line-color": "#ffffff",
          "line-width": 1.5,
          "line-opacity": 0.6,
        },
      });

      // Fire isochrones
      m.addSource("isochrones", {
        type: "geojson",
        data: "/data/fire_isochrones.geojson",
      });

      m.addLayer({
        id: "isochrones-line",
        type: "line",
        source: "isochrones",
        paint: {
          "line-color": fireColorExpression,
          "line-width": 2.5,
          "line-opacity": 0.9,
        },
      });

      // Fire intensity
      m.addSource("intensity", {
        type: "geojson",
        data: "/data/fire_intensity.geojson",
      });

      m.addLayer({
        id: "intensity-heat",
        type: "heatmap",
        source: "intensity",
        paint: {
          "heatmap-weight": [
            "interpolate",
            ["linear"],
            ["get", "intensity"],
            0, 0,
            500, 0.5,
            2000, 1,
          ],
          "heatmap-intensity": 1.2,
          "heatmap-radius": 20,
          "heatmap-opacity": 0.7,
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            0, "rgba(0,0,0,0)",
            0.2, "rgba(255,255,178,0.6)",
            0.4, "rgba(254,204,92,0.7)",
            0.6, "rgba(253,141,60,0.8)",
            0.8, "rgba(227,26,28,0.85)",
            1, "rgba(128,0,38,0.9)",
          ],
        },
        layout: {
          visibility: "none",
        },
      });

      // Click handler for buildings
      m.on("click", "buildings-3d", (e) => {
        if (e.features && e.features.length > 0) {
          const props = e.features[0].properties as Record<string, unknown>;
          // Parse stringified booleans/nulls from GeoJSON
          const parsed: BuildingProperties = {
            ...props,
            name: props.name === "null" ? null : (props.name as string),
            address: props.address === "null" ? null : (props.address as string),
            yearBuilt: props.yearBuilt === "null" ? null : Number(props.yearBuilt),
            fireArrivalMin: props.fireArrivalMin === "null" ? null : Number(props.fireArrivalMin),
            ladderFuel: props.ladderFuel === "true" || props.ladderFuel === true,
            isDuke: props.isDuke === "true" || props.isDuke === true,
            riskScore: Number(props.riskScore),
            terrain: Number(props.terrain),
            terrainMax: Number(props.terrainMax),
            vegetation: Number(props.vegetation),
            vegetationMax: Number(props.vegetationMax),
            structure: Number(props.structure),
            structureMax: Number(props.structureMax),
            exposure: Number(props.exposure),
            exposureMax: Number(props.exposureMax),
            slope: Number(props.slope),
            aspect: Number(props.aspect),
            heatLoadIndex: Number(props.heatLoadIndex),
            stories: Number(props.stories),
            buildingSqFt: Number(props.buildingSqFt),
            canopyCover: Number(props.canopyCover),
            ndviMean: Number(props.ndviMean),
            emberProbability: Number(props.emberProbability),
          } as BuildingProperties;
          onBuildingClick(parsed);

          // Fly to building
          if (e.lngLat) {
            m.flyTo({
              center: e.lngLat,
              zoom: 18,
              pitch: 55,
              duration: 1200,
            });
          }
        }
      });

      // Cursor feedback
      m.on("mouseenter", "buildings-3d", () => {
        m.getCanvas().style.cursor = "pointer";
      });
      m.on("mouseleave", "buildings-3d", () => {
        m.getCanvas().style.cursor = "";
      });

      // Hover tooltip
      const popup = new mapboxgl.Popup({
        closeButton: false,
        closeOnClick: false,
        className: "building-tooltip",
      });

      m.on("mousemove", "buildings-3d", (e) => {
        if (e.features && e.features.length > 0) {
          const p = e.features[0].properties!;
          const name = p.name && p.name !== "null" ? p.name : p.id;
          popup
            .setLngLat(e.lngLat)
            .setHTML(
              `<div class="font-medium">${name}</div>
               <div class="text-sm opacity-80">Risk: ${Number(p.riskScore).toFixed(0)}/100</div>`
            )
            .addTo(m);
        }
      });
      m.on("mouseleave", "buildings-3d", () => popup.remove());

      setLoaded(true);
    });

    map.current = m;

    return () => m.remove();
  }, [onBuildingClick]);

  useEffect(() => {
    initMap();
  }, [initMap]);

  // Toggle layer visibility
  useEffect(() => {
    if (!map.current || !loaded) return;
    const m = map.current;

    const setVis = (id: string, visible: boolean) => {
      if (m.getLayer(id)) {
        m.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
      }
    };

    setVis("buildings-3d", layers.buildings);
    setVis("buildings-outline", layers.buildings);
    setVis("isochrones-line", layers.isochrones);
    setVis("intensity-heat", layers.intensity);
  }, [layers, loaded]);

  const toggleLayer = (key: keyof typeof layers) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <>
      <div ref={mapContainer} className="absolute inset-0" style={{ width: "100%", height: "100%" }} />

      {/* Layer controls */}
      <div className="absolute top-4 left-4 z-10">
        <div className="glass-panel px-4 py-3 space-y-1">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-2">
            Layers
          </h2>
          {([
            ["buildings", "Buildings"],
            ["isochrones", "Fire Isochrones"],
            ["intensity", "Intensity Heatmap"],
          ] as const).map(([key, label]) => (
            <label
              key={key}
              className="flex items-center gap-2 cursor-pointer text-sm text-white/90 hover:text-white transition-colors"
            >
              <input
                type="checkbox"
                checked={layers[key]}
                onChange={() => toggleLayer(key)}
                className="rounded border-white/20 bg-white/10 text-emerald-500 focus:ring-emerald-500/50 focus:ring-offset-0"
              />
              {label}
            </label>
          ))}
        </div>
      </div>
    </>
  );
}
