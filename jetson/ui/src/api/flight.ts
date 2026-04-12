import { API } from "./config";

async function cmd(action: string, body?: object): Promise<{ ok: boolean; message: string }> {
  const r = await fetch(`${API.mavlink}/command/${action}`, {
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
