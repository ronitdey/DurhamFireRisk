export interface BuildingProperties {
  id: string;
  name: string | null;
  address: string | null;
  riskScore: number;
  riskCategory: "Low" | "Moderate" | "High" | "Very High";
  terrain: number;
  terrainMax: number;
  vegetation: number;
  vegetationMax: number;
  structure: number;
  structureMax: number;
  exposure: number;
  exposureMax: number;
  slope: number;
  aspect: number;
  heatLoadIndex: number;
  tpiClass: string;
  roofMaterial: string;
  ventScreening: string;
  yearBuilt: number | null;
  stories: number;
  buildingSqFt: number;
  canopyCover: number;
  ndviMean: number;
  ladderFuel: boolean;
  fireArrivalMin: number | null;
  emberProbability: number;
  isDuke: boolean;
}

export interface IsochoneProperties {
  minutes: number;
  label: string;
}

export interface IntensityProperties {
  intensity: number;
  arrivalMin: number;
}

export interface CampusStats {
  buildingCount: number;
  meanRisk: number;
  maxRisk: number;
  minRisk: number;
  highRiskCount: number;
  moderateRiskCount: number;
  lowRiskCount: number;
  fireMaxArrival: number;
  fireBurnedCells: number;
}
