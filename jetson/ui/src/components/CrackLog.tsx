import type { Detection } from "../hooks/useDetections";
import type { Telemetry } from "../hooks/useTelemetry";

interface Props { history: Detection[]; telem: Telemetry }

export function CrackLog({ history, telem }: Props) {
  // Newest first — no auto-scroll needed, newest is always visible at top
  const entries = history.slice(-50).reverse();

  return (
    <div className="h-full flex flex-col overflow-hidden rounded-md border border-parchment-darker bg-surface">

      {/* Fixed header — outside scroll container so it never overlaps */}
      <div className="shrink-0 px-3 py-1.5 border-b border-parchment-darker flex items-center justify-between bg-parchment-dark">
        <span className="text-[9px] font-mono uppercase tracking-widest text-ink-faint">Live Detections</span>
        <span className="text-[9px] font-mono text-ink-faint">{history.length} frames</span>
      </div>

      {/* Fixed column headers — also outside scroll */}
      {entries.length > 0 && (
        <div className="shrink-0 grid grid-cols-4 px-3 py-1 border-b border-parchment-darker bg-parchment-dark/60">
          <span className="text-[9px] font-mono text-ink-faint">Time</span>
          <span className="text-[9px] font-mono text-ink-faint text-right">N m</span>
          <span className="text-[9px] font-mono text-ink-faint text-right">E m</span>
          <span className="text-[9px] font-mono text-ink-faint text-right">Cov%</span>
        </div>
      )}

      {/* Scrollable rows */}
      <div className="overflow-y-auto flex-1 min-h-0">
        {entries.length === 0 ? (
          <div className="px-3 py-4 text-[10px] font-mono text-ink-faint text-center">Waiting for detections…</div>
        ) : (
          entries.map((d) => {
            const pct = d.crack_ratio_pct;
            const pctCls = pct > 20 ? "text-terracotta font-bold" : pct > 5 ? "text-amber-500" : "text-ink-soft";
            return (
              <div
                key={d.timestamp_ms}
                className="grid grid-cols-4 px-3 py-1.5 border-b border-parchment-darker/40 hover:bg-parchment-dark/40 transition-colors"
              >
                <span className="text-[10px] font-mono text-ink-muted">
                  {new Date(d.timestamp_ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                </span>
                <span className="text-[10px] font-mono text-ink-muted text-right">{telem.north_m.toFixed(1)}</span>
                <span className="text-[10px] font-mono text-ink-muted text-right">{telem.east_m.toFixed(1)}</span>
                <span className={`text-[10px] font-mono text-right ${pctCls}`}>{pct.toFixed(1)}%</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
