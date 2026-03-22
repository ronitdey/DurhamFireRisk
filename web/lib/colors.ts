/** Risk score → color mapping */
export function riskColor(score: number): string {
  if (score < 30) return "#22c55e";
  if (score < 55) return "#f59e0b";
  if (score < 75) return "#ef4444";
  return "#a855f7";
}

/** Risk score → RGBA for Mapbox fill-extrusion */
export function riskColorRGBA(score: number, alpha = 0.85): string {
  if (score < 30) return `rgba(34, 197, 94, ${alpha})`;
  if (score < 55) return `rgba(245, 158, 11, ${alpha})`;
  if (score < 75) return `rgba(239, 68, 68, ${alpha})`;
  return `rgba(168, 85, 247, ${alpha})`;
}

/** Mapbox data-driven color expression for risk scores */
export const riskColorExpression: mapboxgl.Expression = [
  "interpolate",
  ["linear"],
  ["get", "riskScore"],
  0, "#22c55e",
  30, "#22c55e",
  31, "#f59e0b",
  55, "#f59e0b",
  56, "#ef4444",
  75, "#ef4444",
  76, "#a855f7",
  100, "#a855f7",
];

/** Fire isochrone time → color */
export function fireTimeColor(minutes: number): string {
  if (minutes <= 1) return "#ff0000";
  if (minutes <= 2) return "#ff4400";
  if (minutes <= 3) return "#ff7700";
  if (minutes <= 5) return "#ffaa00";
  if (minutes <= 10) return "#ffcc00";
  if (minutes <= 20) return "#ffee00";
  return "#ffff88";
}

export const fireColorExpression: mapboxgl.Expression = [
  "interpolate",
  ["linear"],
  ["get", "minutes"],
  0, "#ff0000",
  1, "#ff2200",
  2, "#ff5500",
  3, "#ff8800",
  5, "#ffbb00",
  10, "#ffdd00",
  30, "#ffff88",
];
