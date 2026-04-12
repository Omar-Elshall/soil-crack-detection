import { useEffect, useState } from "react";
import { API } from "../api/config";

export interface ServiceHealth {
  inference: boolean;
  mavlink: boolean;
  data: boolean;
}

async function ping(url: string): Promise<boolean> {
  try {
    const r = await fetch(`${url}/status`, { signal: AbortSignal.timeout(2000) });
    return r.ok;
  } catch { return false; }
}

export function useServiceHealth(intervalMs = 5000) {
  const [health, setHealth] = useState<ServiceHealth>({ inference: false, mavlink: false, data: false });

  useEffect(() => {
    let alive = true;
    async function check() {
      const [inference, mavlink, data] = await Promise.all([
        ping(API.inference),
        ping(API.mavlink),
        ping(API.data),
      ]);
      if (alive) setHealth({ inference, mavlink, data });
    }
    check();
    const id = setInterval(check, intervalMs);
    return () => { alive = false; clearInterval(id); };
  }, [intervalMs]);

  return health;
}
