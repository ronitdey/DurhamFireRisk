"use client";

import { useState } from "react";

export default function Legend() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="absolute bottom-4 left-4 z-10">
      <div className="glass-panel px-4 py-3">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-white/50 hover:text-white/80 transition-colors w-full"
        >
          <span>Legend</span>
          <svg
            width="10"
            height="10"
            viewBox="0 0 10 10"
            className={`ml-auto transition-transform ${collapsed ? "rotate-180" : ""}`}
          >
            <path d="M2 6l3-3 3 3" stroke="currentColor" strokeWidth="1.5" fill="none" />
          </svg>
        </button>

        {!collapsed && (
          <div className="mt-3 space-y-3">
            {/* Risk levels */}
            <div className="space-y-1.5">
              <p className="text-[10px] uppercase tracking-wider text-white/40">
                Risk Level
              </p>
              {[
                ["#22c55e", "Low (0–30)"],
                ["#f59e0b", "Moderate (30–55)"],
                ["#ef4444", "High (55–75)"],
                ["#a855f7", "Very High (75+)"],
              ].map(([color, label]) => (
                <div key={label} className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-sm"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-xs text-white/70">{label}</span>
                </div>
              ))}
            </div>

            {/* Fire isochrones */}
            <div className="space-y-1.5 pt-2 border-t border-white/10">
              <p className="text-[10px] uppercase tracking-wider text-white/40">
                Fire Arrival
              </p>
              <div className="h-2 rounded-full bg-gradient-to-r from-red-600 via-orange-500 to-yellow-300" />
              <div className="flex justify-between text-[10px] text-white/40">
                <span>0 min</span>
                <span>Max</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
