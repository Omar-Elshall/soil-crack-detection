import { Square, Play } from "lucide-react";

interface Props {
  activeMissionId: string | null;
  loading: boolean;
  onStart: () => void;
  onStop: () => void;
}

export function MissionControl({ activeMissionId, loading, onStart, onStop }: Props) {
  const active = !!activeMissionId;

  return (
    <div className={`rounded-md border p-3 flex items-center justify-between gap-3 transition-colors ${
      active ? "bg-terracotta/10 border-terracotta/30" : "bg-surface/80 border-parchment-darker"
    }`}>
      <div className="min-w-0">
        <div className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
          {active ? "Mission Active" : "Mission"}
        </div>
        <div className="text-xs font-mono text-ink mt-0.5 truncate max-w-[160px]">
          {activeMissionId ?? "No active mission"}
        </div>
      </div>

      {active ? (
        <button
          onClick={onStop}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono font-medium text-white border border-chrome-border disabled:opacity-50 transition-colors"
          style={{ background: "#0D1117" }}
        >
          <Square size={10} className="fill-current" />
          Stop
        </button>
      ) : (
        <button
          onClick={onStart}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-terracotta text-white text-xs font-mono font-medium border border-terracotta hover:bg-terracotta-dark disabled:opacity-50 transition-colors"
        >
          <Play size={10} className="fill-current" />
          Start
        </button>
      )}
    </div>
  );
}
