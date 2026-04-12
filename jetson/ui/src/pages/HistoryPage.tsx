import { useEffect, useState } from "react";
import { listMissions, type MissionMeta, csvUrl, geojsonUrl, pdfUrl } from "../api/missions";
import { FileText, Map, Download, RefreshCw, CheckCircle, Clock } from "lucide-react";

function duration(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function MissionRow({ m }: { m: MissionMeta }) {
  const complete = m.status === "complete";
  const date = new Date(m.start_time).toLocaleString(undefined, {
    month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });

  return (
    <div className="bg-white/60 border border-parchment-darker rounded-md px-4 py-3 flex items-center gap-4">

      {/* Status icon */}
      <div className={`shrink-0 ${complete ? "text-moss" : "text-terracotta"}`}>
        {complete ? <CheckCircle size={16} /> : <Clock size={16} />}
      </div>

      {/* ID + date */}
      <div className="min-w-0 flex-1">
        <div className="font-mono text-xs text-ink font-medium truncate">{m.id}</div>
        <div className="text-[10px] text-ink-muted mt-0.5">{date}</div>
      </div>

      {/* Stats */}
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

      {/* Export buttons */}
      {complete && (
        <div className="flex items-center gap-1 shrink-0">
          <a
            href={csvUrl(m.id)}
            download
            title="Download CSV"
            className="p-1.5 rounded text-ink-muted hover:text-ink hover:bg-parchment-dark transition-colors"
          >
            <Download size={13} />
          </a>
          <a
            href={geojsonUrl(m.id)}
            download
            title="Download GeoJSON"
            className="p-1.5 rounded text-ink-muted hover:text-ink hover:bg-parchment-dark transition-colors"
          >
            <Map size={13} />
          </a>
          <a
            href={pdfUrl(m.id)}
            download
            title="Download PDF Report"
            className="p-1.5 rounded text-ink-muted hover:text-ink hover:bg-parchment-dark transition-colors"
          >
            <FileText size={13} />
          </a>
        </div>
      )}
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

      {/* Header */}
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

      {/* List */}
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
