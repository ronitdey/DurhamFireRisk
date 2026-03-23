"use client";

import { BuildingProperties } from "@/lib/types";
import { riskColor } from "@/lib/colors";

interface BuildingPanelProps {
  building: BuildingProperties | null;
  onClose: () => void;
}

// ── Risk narrative generator ────────────────────────────────────────────────

function generateSummary(b: BuildingProperties): string {
  const parts: string[] = [];
  const name = b.name || "This building";

  // Dominant risk driver
  const drivers = [
    { label: "structural vulnerabilities", val: b.structure / b.structureMax },
    { label: "surrounding vegetation", val: b.vegetation / b.vegetationMax },
    { label: "terrain characteristics", val: b.terrain / b.terrainMax },
    { label: "exposure to neighboring structures", val: b.exposure / b.exposureMax },
  ].sort((a, d) => d.val - a.val);

  const top = drivers[0];

  // Opening
  if (b.riskScore < 30) {
    parts.push(`${name} has relatively low wildfire risk.`);
  } else if (b.riskScore < 55) {
    parts.push(`${name} faces moderate wildfire risk, primarily driven by ${top.label}.`);
  } else if (b.riskScore < 75) {
    parts.push(`${name} has elevated wildfire risk. The main concern is ${top.label}.`);
  } else {
    parts.push(`${name} is at serious wildfire risk, with ${top.label} as the primary driver.`);
  }

  // Structure specifics
  const roof = b.roofMaterial.toLowerCase();
  if (roof.includes("wood") || roof.includes("shake")) {
    parts.push("Its wood roof is highly combustible and vulnerable to ember ignition.");
  } else if (roof.includes("asphalt")) {
    parts.push("Asphalt shingles provide moderate fire resistance but can ignite under sustained ember exposure.");
  } else if (roof.includes("metal")) {
    parts.push("Its metal roof offers strong fire resistance.");
  } else if (roof.includes("slate") || roof.includes("tile")) {
    parts.push("Its slate/tile roof is highly fire-resistant, though embers can enter through gaps.");
  } else if (roof.includes("membrane") || roof.includes("flat")) {
    parts.push("Its flat membrane roof has good fire resistance but is susceptible to pooling debris ignition.");
  } else if (roof.includes("built") || roof.includes("tar")) {
    parts.push("Its built-up roof provides moderate fire resistance.");
  }

  if (b.ventScreening.toLowerCase() === "unscreened") {
    parts.push("Unscreened vents could allow embers to enter the attic.");
  } else if (b.ventScreening.toLowerCase() === "partial") {
    parts.push("Partially screened vents reduce but don't eliminate ember intrusion risk.");
  }

  // Terrain
  if (b.slope > 10) {
    parts.push(`A steep ${b.slope.toFixed(0)}° slope accelerates fire spread toward the structure.`);
  } else if (b.slope > 5) {
    parts.push(`Moderate slope (${b.slope.toFixed(0)}°) contributes to fire spread.`);
  }

  if (b.tpiClass === "ridge" || b.tpiClass === "upper_slope") {
    parts.push("Its elevated position on a ridge increases flame and ember exposure.");
  }

  // Vegetation
  if (b.ladderFuel) {
    parts.push("Ladder fuels nearby could carry ground fire into the canopy.");
  }

  if (b.canopyCover > 40) {
    parts.push(`Dense canopy cover (${b.canopyCover.toFixed(0)}%) creates continuous fuel overhead.`);
  }

  // Fire simulation
  if (b.fireArrivalMin !== null && b.fireArrivalMin < 5) {
    parts.push(`In worst-case conditions, fire would reach this building in under ${Math.ceil(b.fireArrivalMin)} minutes.`);
  } else if (b.fireArrivalMin !== null) {
    parts.push(`Simulated fire arrival is ${b.fireArrivalMin.toFixed(0)} minutes under worst-case conditions.`);
  }

  if (b.emberProbability > 0.7) {
    parts.push("Ember exposure probability is high — firebrands from upwind fuel sources pose a significant ignition risk.");
  }

  return parts.join(" ");
}

// ── Metric explanation tooltips ─────────────────────────────────────────────

const METRIC_INFO: Record<string, { desc: string; why: string }> = {
  Terrain: {
    desc: "Slope steepness, heat load, and topographic position",
    why: "Fire spreads faster uphill and ridge positions receive more radiant heat and ember exposure",
  },
  Vegetation: {
    desc: "Fuel load in 30/100/200ft zones and ladder fuel presence",
    why: "Near-structure vegetation is the primary ignition pathway — ladder fuels carry surface fire into the canopy",
  },
  Structure: {
    desc: "Roof material combustibility and vent screening",
    why: "Roof ignition from embers is the #1 cause of structure loss in wildfires. Unscreened vents allow ember entry",
  },
  Exposure: {
    desc: "Neighbor proximity and ember spotting probability",
    why: "Structures within 20m of each other create fire spread chains. Wind-carried embers can ignite buildings from hundreds of meters away",
  },
};

// ── Components ──────────────────────────────────────────────────────────────

function RiskBar({
  label,
  value,
  max,
  color,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
}) {
  const pct = Math.min((value / max) * 100, 100);
  const info = METRIC_INFO[label];

  return (
    <div className="space-y-1 group">
      <div className="flex justify-between text-xs">
        <span className="text-white/60 flex items-center gap-1">
          {label}
          {info && (
            <span className="relative">
              <svg
                width="12"
                height="12"
                viewBox="0 0 16 16"
                fill="none"
                className="text-white/30 hover:text-white/60 transition-colors cursor-help"
              >
                <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
                <path d="M8 7v4M8 5.5v0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
              <span className="absolute left-4 -top-1 w-56 p-2.5 rounded-lg bg-[#0a0f1e]/95 border border-white/10 text-[11px] text-white/80 leading-relaxed hidden group-hover:block z-50 shadow-xl backdrop-blur-sm pointer-events-none">
                <span className="font-medium text-white/90 block mb-1">{info.desc}</span>
                {info.why}
              </span>
            </span>
          )}
        </span>
        <span className="text-white/80 font-mono">
          {value.toFixed(1)}/{max}
        </span>
      </div>
      <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between py-1.5 border-b border-white/5 last:border-0">
      <span className="text-xs text-white/50">{label}</span>
      <span className="text-xs text-white/90 font-medium">{value}</span>
    </div>
  );
}

export default function BuildingPanel({ building, onClose }: BuildingPanelProps) {
  if (!building) return null;

  const b = building;
  const color = riskColor(b.riskScore);
  const displayName = b.name || b.address || b.id;
  const summary = generateSummary(b);

  return (
    <div className="absolute top-0 right-0 h-full z-20 flex items-start pt-4 pr-4">
      <div className="glass-panel w-80 max-h-[calc(100vh-2rem)] overflow-y-auto animate-slide-in">
        {/* Header */}
        <div className="p-5 border-b border-white/10">
          <div className="flex items-start justify-between">
            <div className="flex-1 min-w-0">
              <h2 className="text-lg font-semibold text-white truncate">
                {displayName}
              </h2>
              {b.address && b.name && (
                <p className="text-xs text-white/40 mt-0.5 truncate">{b.address}</p>
              )}
            </div>
            <button
              onClick={onClose}
              className="ml-2 p-1 rounded-lg hover:bg-white/10 transition-colors text-white/40 hover:text-white"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </button>
          </div>

          {/* Risk badge */}
          <div className="mt-3 flex items-center gap-3">
            <div
              className="text-3xl font-bold tabular-nums"
              style={{ color }}
            >
              {b.riskScore.toFixed(0)}
            </div>
            <div>
              <div
                className="text-xs font-semibold px-2 py-0.5 rounded-full"
                style={{
                  backgroundColor: color + "20",
                  color,
                }}
              >
                {b.riskCategory}
              </div>
              <p className="text-[10px] text-white/40 mt-0.5">out of 100</p>
            </div>
          </div>
        </div>

        {/* Risk Summary */}
        <div className="p-5 border-b border-white/10">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-2">
            Risk Assessment
          </h3>
          <p className="text-xs text-white/70 leading-relaxed">
            {summary}
          </p>
        </div>

        {/* Risk breakdown */}
        <div className="p-5 border-b border-white/10 space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-white/50">
            Risk Breakdown
          </h3>
          <RiskBar label="Terrain" value={b.terrain} max={b.terrainMax} color="#3b82f6" />
          <RiskBar label="Vegetation" value={b.vegetation} max={b.vegetationMax} color="#22c55e" />
          <RiskBar label="Structure" value={b.structure} max={b.structureMax} color="#ef4444" />
          <RiskBar label="Exposure" value={b.exposure} max={b.exposureMax} color="#a855f7" />
        </div>

        {/* Fire simulation */}
        {(b.fireArrivalMin !== null || b.emberProbability > 0) && (
          <div className="p-5 border-b border-white/10">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-3">
              Fire Simulation
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {b.fireArrivalMin !== null && (
                <div className="bg-orange-500/10 rounded-lg p-3 border border-orange-500/20">
                  <p className="text-[10px] text-orange-400/70 uppercase tracking-wider">
                    Arrival Time
                  </p>
                  <p className="text-lg font-bold text-orange-400 mt-0.5">
                    {b.fireArrivalMin.toFixed(0)}
                    <span className="text-xs font-normal ml-0.5">min</span>
                  </p>
                </div>
              )}
              <div className="bg-red-500/10 rounded-lg p-3 border border-red-500/20">
                <p className="text-[10px] text-red-400/70 uppercase tracking-wider">
                  Ember Exposure
                </p>
                <p className="text-lg font-bold text-red-400 mt-0.5">
                  {(b.emberProbability * 100).toFixed(0)}
                  <span className="text-xs font-normal">%</span>
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Building details */}
        <div className="p-5 border-b border-white/10">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-2">
            Structure
          </h3>
          <InfoRow label="Roof Material" value={b.roofMaterial} />
          <InfoRow label="Vent Screening" value={b.ventScreening} />
          <InfoRow label="Stories" value={String(b.stories)} />
          {b.yearBuilt && <InfoRow label="Year Built" value={String(b.yearBuilt)} />}
          {b.buildingSqFt > 0 && (
            <InfoRow label="Building Area" value={`${b.buildingSqFt.toLocaleString()} sq ft`} />
          )}
        </div>

        {/* Terrain & Vegetation */}
        <div className="p-5 pb-6">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-2">
            Environment
          </h3>
          <InfoRow label="Slope" value={`${b.slope.toFixed(1)}°`} />
          <InfoRow label="TPI Class" value={b.tpiClass.replace("_", " ")} />
          <InfoRow label="Heat Load Index" value={b.heatLoadIndex.toFixed(3)} />
          <InfoRow label="Canopy Cover" value={`${b.canopyCover.toFixed(0)}%`} />
          <InfoRow label="NDVI" value={b.ndviMean.toFixed(3)} />
          <InfoRow label="Ladder Fuel" value={b.ladderFuel ? "Present" : "None"} />
        </div>

        {/* Methodology footer */}
        <div className="px-5 pb-5">
          <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-3">
            <p className="text-[10px] text-white/30 leading-relaxed">
              Scores derived from Rothermel (1972) fire spread physics, 1m LiDAR terrain analysis, and structure-level vulnerability assessment. Fire simulation uses worst-case 35mph SW wind scenario.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
