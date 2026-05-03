import { useEffect, useState } from "react";
import { API } from "../api/config";
import { getActive, subscribe, type MavlinkKind } from "../api/mavlinkSource";

export interface ServiceHealth {
  inference: boolean;
  mavlink: boolean;
  mavlink_source: MavlinkKind;   // "radio" | "wifi" — which path the dot reflects
  data: boolean;
}

async function ping(url: string): Promise<boolean> {
  try {
    // 5 s — first hit through mDNS (soilcrack.local) can take 2-3 s on
    // some networks, and the browser sometimes re-resolves per request.
    const r = await fetch(`${url}/status`, { signal: AbortSignal.timeout(5000) });
    return r.ok;
  } catch { return false; }
}

export function useServiceHealth(intervalMs = 5000) {
  const [health, setHealth] = useState<ServiceHealth>({
    inference: false, mavlink: false, mavlink_source: "wifi", data: false,
  });

  useEffect(() => {
    let alive = true;
    async function check() {
      const src = getActive();
      const [inference, mavlink, data] = await Promise.all([
        ping(API.inference),
        ping(src.base),
        ping(API.data),
      ]);
      if (alive) setHealth({ inference, mavlink, mavlink_source: src.kind, data });
    }
    check();
    const unsub = subscribe(() => check());     // re-check immediately on source change
    const id = setInterval(check, intervalMs);
    return () => { alive = false; clearInterval(id); unsub(); };
  }, [intervalMs]);

  return health;
}
