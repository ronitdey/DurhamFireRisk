"use client";

import { useEffect, useState } from "react";
import { CampusStats } from "@/lib/types";

function Stat({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="text-center">
      <p className="text-lg font-bold text-white tabular-nums">
        {value}
        {unit && <span className="text-xs font-normal text-white/40 ml-0.5">{unit}</span>}
      </p>
      <p className="text-[10px] uppercase tracking-wider text-white/40">{label}</p>
    </div>
  );
}

export default function StatsBar() {
  const [stats, setStats] = useState<CampusStats | null>(null);

  useEffect(() => {
    fetch("/data/stats.json")
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {});
  }, []);

  if (!stats) return null;

  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10">
      <div className="glass-panel px-6 py-3 flex items-center gap-6">
        <div className="pr-6 border-r border-white/10">
          <h1 className="text-sm font-semibold text-white tracking-tight">
            Durham Fire Risk
          </h1>
          <p className="text-[10px] text-white/40">Duke East Campus</p>
        </div>
        <Stat label="Buildings" value={String(stats.buildingCount)} />
        <Stat label="Mean Risk" value={stats.meanRisk.toFixed(1)} />
        <Stat label="High Risk" value={String(stats.highRiskCount)} />
        <Stat label="Max Score" value={stats.maxRisk.toFixed(0)} unit="/100" />
      </div>
    </div>
  );
}
