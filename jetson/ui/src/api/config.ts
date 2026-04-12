export const API = {
  inference: import.meta.env.VITE_INFERENCE_URL ?? "http://localhost:8001",
  mavlink:   import.meta.env.VITE_MAVLINK_URL   ?? "http://localhost:8002",
  data:      import.meta.env.VITE_DATA_URL       ?? "http://localhost:8003",
};

export const WS = {
  detections: (API.inference.replace(/^http/, "ws")) + "/ws/detections",
  telemetry:  (API.mavlink.replace(/^http/, "ws"))   + "/ws/telemetry",
};
