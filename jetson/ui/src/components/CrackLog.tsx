import { useEffect, useRef } from "react";
import type { Detection } from "../hooks/useDetections";
import type { Telemetry } from "../hooks/useTelemetry";

interface Props {
  history: Detection[];
  telem: Telemetry;
}

export function CrackLog({ history, telem }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history.length]);

  const entries = history.slice(-50).reverse(); // show last 50, newest first

  return (
    <div className="flex flex-col overflow-hidden rounded-md border border-parchment-darker bg-white/50">
      <div className="px-3 py-2 border-b border-parchment-darker flex items-center justify-between">
        <span className="text-[9px] font-mono uppercase tracking-widest text-ink-faint">
          Live Detections
        </span>
        <span className="text-[9px] font-mono text-ink-faint">{history.length} frames</span>
      </div>

      <div className="overflow-y-auto flex-1 min-h-0">
        {entries.length === 0 ? (
          <div className="px-3 py-4 text-[10px] font-mono text-ink-faint text-center">
            Waiting for detections…
          </div>
        ) : (
          <table className="w-full text-[10px] font-mono">
            <thead>
              <tr className="border-b border-parchment-darker sticky top-0 bg-parchment">
                <th className="text-left px-3 py-1.5 text-ink-faint font-medium">Time</th>
                <th className="text-right px-3 py-1.5 text-ink-faint font-medium">N (m)</th>
                <th className="text-right px-3 py-1.5 text-ink-faint font-medium">E (m)</th>
                <th className="text-right px-3 py-1.5 text-ink-faint font-medium">Cover%</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((d, i) => {
                const pct = d.crack_ratio_pct;
                const color = pct > 20 ? "text-terracotta font-medium" : pct > 5 ? "text-amber-600" : "text-ink-soft";
                return (
                  <tr
                    key={d.timestamp_ms}
                    className={`border-b border-parchment-darker/50 ${i === 0 ? "animate-pulse-once bg-terracotta/5" : ""}`}
                  >
                    <td className="px-3 py-1 text-ink-muted">
                      {new Date(d.timestamp_ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                    </td>
                    <td className="px-3 py-1 text-right text-ink-muted">{telem.north_m.toFixed(1)}</td>
                    <td className="px-3 py-1 text-right text-ink-muted">{telem.east_m.toFixed(1)}</td>
                    <td className={`px-3 py-1 text-right ${color}`}>{pct.toFixed(1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
