import { useEffect, useRef, useState } from "react";
import { wsUrl, subscribe, getActive, type MavlinkKind } from "../api/mavlinkSource";

export interface StatusMessage {
  text: string;
  severity: string;       // "ERROR" | "WARNING" | "NOTICE" | "INFO" etc.
  severity_level: number; // 0=emergency … 7=debug
  ts: number;             // unix timestamp
}

export interface Telemetry {
  connected: boolean;
  source: MavlinkKind;     // "radio" | "wifi" — which path the data came from
  lat: number;
  lon: number;
  alt_m: number;
  roll_deg: number;
  pitch_deg: number;
  yaw_deg: number;
  heading_deg: number;
  battery_pct: number;
  battery_v: number;
  mode: string;
  armed: boolean;
  gps_fix: number;
  north_m: number;
  east_m: number;
  satellites: number;
  status_messages: StatusMessage[];
}

const DEFAULT: Telemetry = {
  connected: false, source: "wifi",
  lat: 0, lon: 0, alt_m: 0,
  roll_deg: 0, pitch_deg: 0, yaw_deg: 0, heading_deg: 0,
  battery_pct: 0, battery_v: 0, mode: "—", armed: false,
  gps_fix: 0, north_m: 0, east_m: 0, satellites: 0,
  status_messages: [],
};

export function useTelemetry() {
  const [telem, setTelem] = useState<Telemetry>(DEFAULT);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let alive = true;
    let retryTimeout: ReturnType<typeof setTimeout>;
    let currentSourceKind = getActive().kind;

    function connect() {
      if (!alive) return;
      try { wsRef.current?.close(); } catch {}
      const url = wsUrl();
      currentSourceKind = getActive().kind;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (e) => {
        if (!alive) return;
        try {
          const data = JSON.parse(e.data);
          setTelem({ ...data, source: currentSourceKind });
        } catch { /* ignore */ }
      };
      ws.onclose = () => {
        if (alive) retryTimeout = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
    }

    // Re-open immediately when the active source changes (radio plugged in/out).
    const unsubscribe = subscribe(() => {
      clearTimeout(retryTimeout);
      connect();
    });

    connect();
    return () => {
      alive = false;
      clearTimeout(retryTimeout);
      unsubscribe();
      wsRef.current?.close();
    };
  }, []);

  return telem;
}
