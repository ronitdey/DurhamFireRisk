"use client";

import { BuildingProperties } from "@/lib/types";
import { riskColor } from "@/lib/colors";

interface BuildingPanelProps {
  building: BuildingProperties | null;
  onClose: () => void;
}

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
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-white/60">{label}</span>
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
        <div className="p-5">
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
      </div>
    </div>
  );
}
