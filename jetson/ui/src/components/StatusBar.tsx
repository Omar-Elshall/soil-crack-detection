import { useServiceHealth } from "../hooks/useServiceHealth";

function Dot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-1.5 h-1.5 rounded-full transition-colors ${ok ? "bg-moss" : "bg-red-400"}`} />
      <span className={`text-[10px] font-mono uppercase tracking-widest ${ok ? "text-ink-muted" : "text-red-400"}`}>
        {label}
      </span>
    </div>
  );
}

export function StatusBar() {
  const health = useServiceHealth();
  const allOk = health.inference && health.mavlink && health.data;

  return (
    <div className="h-8 px-4 border-b border-parchment-darker bg-surface/60 flex items-center justify-between shrink-0">
      <span className="text-[10px] font-mono text-ink-faint uppercase tracking-widest">
        {allOk ? "All services online" : "Service issue detected"}
      </span>
      <div className="flex items-center gap-4">
        <Dot ok={health.inference} label="Camera" />
        <Dot ok={health.mavlink}   label="MAVLink" />
        <Dot ok={health.data}      label="Data" />
      </div>
    </div>
  );
}
