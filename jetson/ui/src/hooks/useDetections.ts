import { useEffect, useRef, useState } from "react";
import { WS } from "../api/config";

export interface Detection {
  crack_ratio_pct: number;
  fps: number;
  timestamp_ms: number;
}

export function useDetections() {
  const [latest, setLatest] = useState<Detection | null>(null);
  const [history, setHistory] = useState<Detection[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let alive = true;
    let retryTimeout: ReturnType<typeof setTimeout>;

    function connect() {
      const ws = new WebSocket(WS.detections);
      wsRef.current = ws;

      ws.onmessage = (e) => {
        if (!alive) return;
        try {
          const d: Detection = JSON.parse(e.data);
          setLatest(d);
          setHistory((prev) => [...prev.slice(-299), d]);
        } catch { /* ignore */ }
      };
      ws.onclose = () => {
        if (alive) retryTimeout = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
    }

    connect();
    return () => {
      alive = false;
      clearTimeout(retryTimeout);
      wsRef.current?.close();
    };
  }, []);

  return { latest, history };
}
