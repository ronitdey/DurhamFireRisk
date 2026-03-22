"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import BuildingPanel from "@/components/BuildingPanel";
import Legend from "@/components/Legend";
import StatsBar from "@/components/StatsBar";
import { BuildingProperties } from "@/lib/types";

// Mapbox GL requires browser APIs — avoid SSR
const Map = dynamic(() => import("@/components/Map"), { ssr: false });

export default function Home() {
  const [selectedBuilding, setSelectedBuilding] =
    useState<BuildingProperties | null>(null);

  return (
    <main className="relative w-screen h-screen">
      <Map onBuildingClick={setSelectedBuilding} />
      <StatsBar />
      <Legend />
      <BuildingPanel
        building={selectedBuilding}
        onClose={() => setSelectedBuilding(null)}
      />
    </main>
  );
}
