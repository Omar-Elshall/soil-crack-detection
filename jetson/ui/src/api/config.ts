// Derive the service host from the page URL so the UI works from any laptop
// on the same WiFi without rebuilding.
const HOST = typeof window !== "undefined" ? window.location.hostname : "localhost";

// MAVLink has TWO possible sources:
//   PRIMARY = the laptop's local relay reading from the SiK telemetry radio
//             (works at hundreds of meters, regardless of WiFi)
//   FALLBACK = the Jetson's MAVLink service over WiFi
//
// At startup we ping PRIMARY/status; if it responds we use the relay. Otherwise
// we fall back. We re-check every 30 s so plugging the radio in mid-demo
// transparently switches the UI to the radio path. See mavlinkSource.ts for
// the active state + reconnection signal.
export const MAVLINK_PRIMARY  = "http://localhost:18002";
export const MAVLINK_FALLBACK = `http://${HOST}:8002`;

export const API = {
  inference: import.meta.env.VITE_INFERENCE_URL ?? `http://${HOST}:8001`,
  mavlink:   import.meta.env.VITE_MAVLINK_URL   ?? MAVLINK_FALLBACK,  // overridden at runtime by mavlinkSource.ts
  data:      import.meta.env.VITE_DATA_URL       ?? `http://${HOST}:8003`,
};

export const WS = {
  detections: (API.inference.replace(/^http/, "ws")) + "/ws/detections",
  telemetry:  (API.mavlink.replace(/^http/, "ws"))   + "/ws/telemetry",
};
