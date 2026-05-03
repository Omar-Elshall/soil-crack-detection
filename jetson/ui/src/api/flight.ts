import { getActive } from "./mavlinkSource";

// All command POSTs go to whichever MAVLink source is currently active —
// radio relay (preferred) or Jetson WiFi (fallback). Resolved per-call so
// commands fire over the right path even if the source changed mid-session.
async function cmd(action: string, body?: object): Promise<{ ok: boolean; message: string }> {
  const base = getActive().base;
  const r = await fetch(`${base}/command/${action}`, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  return r.json();
}

export const arm      = ()                                          => cmd("arm");
export const disarm   = ()                                          => cmd("disarm");
export const guided   = ()                                          => cmd("guided");
export const takeoff  = (altitude_m = 0.3)                         => cmd("takeoff", { altitude_m });
export const land     = ()                                          => cmd("land");
export const rtl      = ()                                          => cmd("rtl");
export const gotoNED  = (north_m: number, east_m: number, alt_m: number) =>
  cmd("goto", { north_m, east_m, alt_m });

export interface Waypoint { lat: number; lon: number; alt: number }
export const uploadMission  = (waypoints: Waypoint[], takeoff_alt = 4.0) =>
  cmd("upload-mission", { waypoints, takeoff_alt });
export const startMission   = () => cmd("start-mission");

// Sequenced demo flight: GUIDED -> ARM -> takeoff -> hover -> LAND -> disarm.
// Backend handles timing; UI just awaits the final response.
export const demoFlight     = (altitude_m = 1.0, hover_seconds = 30.0) =>
  fetch(`${getActive().base}/command/demo-flight?altitude_m=${altitude_m}&hover_seconds=${hover_seconds}`, {
    method: "POST",
  }).then((r) => r.json());
