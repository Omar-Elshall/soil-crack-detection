import { API } from "./config";

export interface MissionMeta {
  id: string;
  start_time: string;
  end_time: string | null;
  status: "active" | "complete";
  model: string;
  total_detections: number;
  max_coverage_pct: number;
  mean_coverage_pct: number;
  flight_duration_s: number;
  bbox: { min_lat: number; max_lat: number; min_lon: number; max_lon: number } | null;
}

export async function startMission(model = "EfficientCrackNet"): Promise<{ mission_id: string }> {
  const r = await fetch(`${API.data}/missions/start?model=${encodeURIComponent(model)}`, { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function stopMission(id: string): Promise<{ ok: boolean; meta: MissionMeta }> {
  const r = await fetch(`${API.data}/missions/${id}/stop`, { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function listMissions(): Promise<MissionMeta[]> {
  const r = await fetch(`${API.data}/missions`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getMission(id: string): Promise<MissionMeta> {
  const r = await fetch(`${API.data}/missions/${id}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export interface DetectionRow {
  timestamp: string;
  lat: number; lon: number; alt_m: number;
  north_m: number; east_m: number; heading_deg: number;
  crack_ratio_pct: number; mask_filename: string;
}

export async function getDetections(id: string): Promise<DetectionRow[]> {
  const r = await fetch(`${API.data}/missions/${id}/detections`);
  if (!r.ok) return [];
  return r.json();
}

export function csvUrl(id: string)     { return `${API.data}/missions/${id}/export/csv`; }
export function geojsonUrl(id: string) { return `${API.data}/missions/${id}/export/geojson`; }
export function pdfUrl(id: string)     { return `${API.data}/missions/${id}/export/pdf`; }
export function maskUrl(id: string, filename: string) {
  return `${API.data}/missions/${id}/masks/${filename}`;
}
