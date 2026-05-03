import { useState, useEffect } from "react";
import { useServiceHealth } from "../hooks/useServiceHealth";
import type { StatusMessage } from "../hooks/useTelemetry";
import { X, AlertTriangle, Info } from "lucide-react";

interface Props { statusMessages?: StatusMessage[]; armed?: boolean; mode?: string }

function Dot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-1.5 h-1.5 rounded-full transition-colors ${ok ? "bg-moss" : "bg-red-400 animate-pulse"}`} />
      <span className={`text-[10px] font-mono uppercase tracking-widest ${ok ? "text-ink-muted" : "text-red-400"}`}>
        {label}
      </span>
    </div>
  );
}

export function StatusBar({ statusMessages = [], armed = false, mode }: Props) {
  const health = useServiceHealth();
  const [dismissed, setDismissed] = useState<number>(0); // dismiss messages older than ts

  // Show most recent non-debug message
  const visible = statusMessages
    .filter(m => m.severity_level <= 5 && m.ts > dismissed)
    .slice(-1)[0];

  // Auto-dismiss info/debug messages after 8s
  useEffect(() => {
    if (!visible || visible.severity_level >= 6) return;
    const timer = setTimeout(() => setDismissed(visible.ts), 8000);
    return () => clearTimeout(timer);
  }, [visible?.ts]);

  const msgColor =
    visible?.severity_level === undefined ? "" :
    visible.severity_level <= 3 ? "text-red-400 bg-red-400/8 border-red-400/20" :
    visible.severity_level === 4 ? "text-amber-400 bg-amber-400/8 border-amber-400/20" :
    "text-ink-muted bg-surface/40 border-parchment-darker";

  return (
    <div className="shrink-0">
      {/* Service health bar */}
      <div className="h-8 px-4 border-b border-parchment-darker bg-parchment-dark/60 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-[10px] font-mono text-ink-faint uppercase tracking-widest">
            {health.inference && health.mavlink && health.data ? "All systems nominal" : "Service degraded"}
          </span>
          {/* Armed status — always visible */}
          <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-sm border text-[10px] font-mono font-bold uppercase tracking-widest transition-colors ${
            armed
              ? "bg-red-500/15 border-red-500/40 text-red-400"
              : "bg-moss/10 border-moss/25 text-moss"
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${armed ? "bg-red-400 animate-pulse" : "bg-moss"}`} />
            {armed ? "ARMED" : "SAFE"}
          </div>
          {mode && (
            <span className="text-[10px] font-mono text-ink-faint uppercase tracking-widest">
              {mode}
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <Dot ok={health.inference} label="Camera" />
          <Dot ok={health.mavlink}   label={health.mavlink_source === "radio" ? "MAVLink·radio" : "MAVLink·wifi"} />
          <Dot ok={health.data}      label="Data" />
        </div>
      </div>

      {/* ArduPilot message banner */}
      {visible && (
        <div className={`px-4 py-1.5 border-b flex items-center gap-2 text-xs font-mono ${msgColor}`}>
          {visible.severity_level <= 4
            ? <AlertTriangle size={12} className="shrink-0" />
            : <Info size={12} className="shrink-0" />
          }
          <span className="font-bold uppercase text-[10px] tracking-wider shrink-0">{visible.severity}</span>
          <span className="flex-1 truncate">{visible.text}</span>
          <button onClick={() => setDismissed(visible.ts)} className="shrink-0 opacity-60 hover:opacity-100">
            <X size={12} />
          </button>
        </div>
      )}
    </div>
  );
}
