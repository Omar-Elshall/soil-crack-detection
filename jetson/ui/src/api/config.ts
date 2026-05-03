// Derive the service host from the page URL so the UI works from any laptop
// on the same WiFi without rebuilding (was hard-coded to localhost, which
// broke when the page is served from the Jetson but loaded on a laptop).
const HOST = typeof window !== "undefined" ? window.location.hostname : "localhost";

export const API = {
  inference: import.meta.env.VITE_INFERENCE_URL ?? `http://${HOST}:8001`,
  mavlink:   import.meta.env.VITE_MAVLINK_URL   ?? `http://${HOST}:8002`,
  data:      import.meta.env.VITE_DATA_URL       ?? `http://${HOST}:8003`,
};

export const WS = {
  detections: (API.inference.replace(/^http/, "ws")) + "/ws/detections",
  telemetry:  (API.mavlink.replace(/^http/, "ws"))   + "/ws/telemetry",
};
