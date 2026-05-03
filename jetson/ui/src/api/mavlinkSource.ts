// mavlinkSource.ts — runtime selection between the laptop SiK-radio relay
// (PRIMARY) and the Jetson MAVLink service (FALLBACK).
//
// At app start we probe both via /status. PRIMARY wins if it responds in
// <1.5 s. We re-check every 30 s so plugging the radio in mid-demo
// transparently switches the UI's source.
//
// Components subscribe via subscribe(callback) to be notified when the
// active source changes — they should re-open their WebSockets and
// redo health pings against the new base.

import { MAVLINK_PRIMARY, MAVLINK_FALLBACK } from "./config";

export type MavlinkKind = "radio" | "wifi";
export interface MavlinkSource {
  base: string;
  kind: MavlinkKind;
}

const RADIO: MavlinkSource = { base: MAVLINK_PRIMARY,  kind: "radio" };
const WIFI:  MavlinkSource = { base: MAVLINK_FALLBACK, kind: "wifi"  };

let active: MavlinkSource = WIFI;     // default until first probe completes
const listeners = new Set<(s: MavlinkSource) => void>();

async function ping(base: string, timeoutMs = 1500): Promise<boolean> {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    const r = await fetch(`${base}/status`, { signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) return false;
    // The relay process can be alive while the radio is unplugged — keep
    // serving stale snapshots — so we have to look at the body. The MAVLink
    // service sets connected=false once no message has arrived in ~8 s.
    const body = await r.json();
    return body && body.connected === true;
  } catch {
    return false;
  }
}

async function probe() {
  const radioOk = await ping(MAVLINK_PRIMARY);
  const next: MavlinkSource = radioOk ? RADIO : WIFI;
  if (next.base !== active.base) {
    active = next;
    listeners.forEach((cb) => cb(active));
  }
}

export function getActive(): MavlinkSource {
  return active;
}

export function wsUrl(): string {
  return active.base.replace(/^http/, "ws") + "/ws/telemetry";
}

export function subscribe(cb: (s: MavlinkSource) => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

// Kick off probing — once at startup, then every 30 s.
if (typeof window !== "undefined") {
  probe();
  setInterval(probe, 30_000);
}
