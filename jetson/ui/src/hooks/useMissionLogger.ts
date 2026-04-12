import { useEffect, useRef } from "react";
import { API } from "../api/config";
import type { Detection } from "./useDetections";
import type { Telemetry } from "./useTelemetry";

/**
 * Bridges inference detections → data service while a mission is active.
 * Fires at most once per `intervalMs` (default 1 Hz) to avoid hammering the service.
 */
export function useMissionLogger(
  activeMissionId: string | null,
  latest: Detection | null,
  telem: Telemetry,
  intervalMs = 1000,
) {
  // Keep telemetry in a ref so the effect doesn't re-run on every telem update —
  // only on new detection frames.
  const telemRef = useRef(telem);
  useEffect(() => { telemRef.current = telem; }, [telem]);

  const lastLogRef = useRef<number>(0);

  useEffect(() => {
    if (!activeMissionId || !latest) return;

    const now = Date.now();
    if (now - lastLogRef.current < intervalMs) return;
    lastLogRef.current = now;

    const t = telemRef.current;
    fetch(`${API.data}/missions/${activeMissionId}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat: t.lat,
        lon: t.lon,
        alt_m: t.alt_m,
        north_m: t.north_m,
        east_m: t.east_m,
        heading_deg: t.heading_deg,
        crack_ratio_pct: latest.crack_ratio_pct,
      }),
    }).catch(() => { /* silently ignore — offline or service down */ });
  }, [latest, activeMissionId, intervalMs]);
}
