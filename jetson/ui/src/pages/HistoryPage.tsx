import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listMissions, type MissionMeta } from "../api/missions";
import { CheckCircle, Clock, RefreshCw, ChevronRight } from "lucide-react";

function duration(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function MissionRow({ m }: { m: MissionMeta }) {
  const navigate = useNavigate();
  const complete = m.status === "complete";
  const date = new Date(m.start_time).toLocaleString(undefined, {
    month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });

  return (
    <div
      onClick={() => navigate(`/missions/${m.id}`)}
      className="bg-white/60 border border-parchment-darker rounded-md px-4 py-3 flex items-center gap-4 cursor-pointer hover:bg-parchment-dark hover:border-ink-faint/30 transition-colors group"
    >
      <div className={`shrink-0 ${complete ? "text-moss" : "text-terracotta"}`}>
        {complete ? <CheckCircle size={16} /> : <Clock size={16} />}
      </div>

      <div className="min-w-0 flex-1">
        <div className="font-mono text-xs text-ink font-medium truncate">{m.id}</div>
        <div className="text-[10px] text-ink-muted mt-0.5">{date}</div>
      </div>

      <div className="hidden sm:flex items-center gap-4 shrink-0">
        <div className="text-center">
          <div className="font-mono text-sm text-ink">{m.total_detections}</div>
          <div className="text-[9px] text-ink-faint uppercase tracking-widest">frames</div>
        </div>
        <div className="text-center">
          <div className={`font-mono text-sm ${m.max_coverage_pct > 10 ? "text-terracotta" : "text-ink"}`}>
            {m.max_coverage_pct.toFixed(1)}%
          </div>
          <div className="text-[9px] text-ink-faint uppercase tracking-widest">max cover</div>
        </div>
        <div className="text-center">
          <div className="font-mono text-sm text-ink">{duration(m.flight_duration_s)}</div>
          <div className="text-[9px] text-ink-faint uppercase tracking-widest">duration</div>
        </div>
      </div>

      <ChevronRight size={14} className="text-ink-faint group-hover:text-ink-muted transition-colors shrink-0" />
    </div>
  );
}

export default function HistoryPage() {
  const [missions, setMissions] = useState<MissionMeta[]>([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try { setMissions(await listMissions()); }
    finally { setLoading(false); }
  }

  useEffect(() => { load(); }, []);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="px-6 pt-6 pb-4 border-b border-parchment-darker flex items-end justify-between">
        <div>
          <h1 className="font-display text-2xl text-ink italic">Mission History</h1>
          <p className="text-xs text-ink-muted mt-1 font-sans">
            {missions.length} recorded mission{missions.length !== 1 ? "s" : ""}
          </p>
        </div>
        <button
          onClick={load}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-parchment-darker bg-white/60 text-xs font-mono text-ink-soft hover:bg-parchment-dark transition-colors"
        >
          <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4">
        {loading && missions.length === 0 ? (
          <div className="text-xs font-mono text-ink-faint text-center mt-12">Loading…</div>
        ) : missions.length === 0 ? (
          <div className="text-center mt-16">
            <div className="font-display text-lg text-ink-muted italic">No missions yet</div>
            <p className="text-xs text-ink-faint mt-2">Start a mission from the Live view to begin recording.</p>
          </div>
        ) : (
          <div className="flex flex-col gap-2 max-w-3xl">
            {missions.map((m) => <MissionRow key={m.id} m={m} />)}
          </div>
        )}
      </div>
    </div>
  );
}
