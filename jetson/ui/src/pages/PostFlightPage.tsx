import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  getMission, getDetections, csvUrl, geojsonUrl, pdfUrl,
  type MissionMeta, type DetectionRow,
} from "../api/missions";
import { SummaryCards } from "../components/SummaryCards";
import { MissionMap, type DetectionPoint } from "../components/MissionMap";
import { CrackRatioChart } from "../components/CrackRatioChart";
import { ArrowLeft, Download, Map, BarChart2, Table, FileText } from "lucide-react";
import type { Detection } from "../hooks/useDetections";

type Tab = "map" | "analysis" | "raw";

export default function PostFlightPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [meta, setMeta] = useState<MissionMeta | null>(null);
  const [detections, setDetections] = useState<DetectionRow[]>([]);
  const [tab, setTab] = useState<Tab>("map");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    Promise.all([getMission(id), getDetections(id)])
      .then(([m, d]) => { setMeta(m); setDetections(d); })
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <span className="text-xs font-mono text-ink-faint animate-pulse">Loading mission…</span>
      </div>
    );
  }

  if (!meta || !id) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2">
        <span className="font-display text-lg text-ink-muted italic">Mission not found</span>
        <button onClick={() => navigate("/history")} className="text-xs font-mono text-terracotta hover:underline">
          ← Back to History
        </button>
      </div>
    );
  }

  // Convert detections to chart format
  const chartHistory: Detection[] = detections.map((d, i) => ({
    crack_ratio_pct: d.crack_ratio_pct,
    fps: 0,
    timestamp_ms: i,
  }));

  // Map points
  const mapPoints: DetectionPoint[] = detections
    .filter((d) => d.lat !== 0)
    .map((d) => ({
      lat: d.lat, lon: d.lon,
      crack_ratio_pct: d.crack_ratio_pct,
      timestamp: d.timestamp,
    }));

  const hasGPS = mapPoints.length > 0;
  const defaultCenter: [number, number] | undefined = hasGPS
    ? [mapPoints[0].lat, mapPoints[0].lon]
    : undefined;

  const tabStyle = (t: Tab) =>
    `flex items-center gap-1.5 px-3 py-2 text-xs font-mono border-b-2 transition-colors ${
      tab === t
        ? "border-terracotta text-terracotta"
        : "border-transparent text-ink-muted hover:text-ink hover:border-parchment-darker"
    }`;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">

      {/* Header */}
      <div className="px-6 pt-5 pb-3 border-b border-parchment-darker shrink-0">
        <button
          onClick={() => navigate("/history")}
          className="flex items-center gap-1 text-xs font-mono text-ink-muted hover:text-ink mb-3 transition-colors"
        >
          <ArrowLeft size={12} /> History
        </button>
        <div className="flex items-end justify-between gap-4">
          <div>
            <h1 className="font-display text-2xl text-ink italic">{meta.id}</h1>
            <p className="text-xs text-ink-muted mt-1 font-sans">
              {new Date(meta.start_time).toLocaleString()} · {meta.model}
            </p>
          </div>
          {/* Export buttons */}
          <div className="flex items-center gap-2 shrink-0">
            <a href={csvUrl(id)} download
              className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-parchment-darker bg-white/60 text-xs font-mono text-ink-soft hover:bg-parchment-dark transition-colors">
              <Download size={11} /> CSV
            </a>
            <a href={geojsonUrl(id)} download
              className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-parchment-darker bg-white/60 text-xs font-mono text-ink-soft hover:bg-parchment-dark transition-colors">
              <Map size={11} /> GeoJSON
            </a>
            <a href={pdfUrl(id)} download
              className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-terracotta/30 bg-terracotta/8 text-xs font-mono text-terracotta hover:bg-terracotta/15 transition-colors">
              <FileText size={11} /> PDF Report
            </a>
          </div>
        </div>
      </div>

      {/* Summary cards */}
      <div className="px-6 py-4 shrink-0 border-b border-parchment-darker">
        <SummaryCards meta={meta} />
      </div>

      {/* Tabs */}
      <div className="px-6 border-b border-parchment-darker shrink-0 flex gap-1">
        <button className={tabStyle("map")}     onClick={() => setTab("map")}>
          <Map size={12} /> Map
        </button>
        <button className={tabStyle("analysis")} onClick={() => setTab("analysis")}>
          <BarChart2 size={12} /> Analysis
        </button>
        <button className={tabStyle("raw")}     onClick={() => setTab("raw")}>
          <Table size={12} /> Raw Data
        </button>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-hidden p-4">

        {tab === "map" && (
          <div className="h-full">
            {hasGPS ? (
              <MissionMap points={mapPoints} defaultCenter={defaultCenter} />
            ) : (
              <div className="h-full flex items-center justify-center flex-col gap-2">
                <span className="font-display text-lg text-ink-muted italic">No GPS data</span>
                <p className="text-xs text-ink-faint">This mission was flown without a GPS fix.<br />NED coordinates are logged in the Raw Data tab.</p>
              </div>
            )}
          </div>
        )}

        {tab === "analysis" && (
          <div className="max-w-3xl flex flex-col gap-6">
            <div>
              <h3 className="text-xs font-mono uppercase tracking-widest text-ink-muted mb-3">
                Crack Coverage Over Time
              </h3>
              <div className="rounded-md border border-parchment-darker bg-white/60 p-4 h-40">
                <CrackRatioChart history={chartHistory} />
              </div>
            </div>

            <div>
              <h3 className="text-xs font-mono uppercase tracking-widest text-ink-muted mb-3">
                Top Detections by Coverage
              </h3>
              <div className="rounded-md border border-parchment-darker bg-white/60 overflow-hidden">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="border-b border-parchment-darker bg-parchment">
                      <th className="text-left px-4 py-2 text-ink-faint font-medium">Timestamp</th>
                      <th className="text-right px-4 py-2 text-ink-faint font-medium">Alt (m)</th>
                      <th className="text-right px-4 py-2 text-ink-faint font-medium">Heading °</th>
                      <th className="text-right px-4 py-2 text-ink-faint font-medium">Coverage %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...detections]
                      .sort((a, b) => b.crack_ratio_pct - a.crack_ratio_pct)
                      .slice(0, 10)
                      .map((d, i) => (
                        <tr key={i} className="border-b border-parchment-darker/50">
                          <td className="px-4 py-1.5 text-ink-muted">{d.timestamp.slice(0, 19)}</td>
                          <td className="px-4 py-1.5 text-right text-ink-soft">{d.alt_m.toFixed(1)}</td>
                          <td className="px-4 py-1.5 text-right text-ink-soft">{d.heading_deg.toFixed(0)}</td>
                          <td className={`px-4 py-1.5 text-right font-medium ${d.crack_ratio_pct > 10 ? "text-terracotta" : "text-ink"}`}>
                            {d.crack_ratio_pct.toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {tab === "raw" && (
          <div className="h-full overflow-auto">
            <table className="w-full text-[11px] font-mono border border-parchment-darker rounded-md overflow-hidden">
              <thead>
                <tr className="bg-parchment border-b border-parchment-darker sticky top-0">
                  {["#", "Timestamp", "Lat", "Lon", "Alt m", "N m", "E m", "Hdg °", "Cover %", "Mask"].map((h) => (
                    <th key={h} className="text-left px-3 py-2 text-ink-faint font-medium whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {detections.map((d, i) => (
                  <tr key={i} className="border-b border-parchment-darker/40 hover:bg-white/40">
                    <td className="px-3 py-1.5 text-ink-faint">{i + 1}</td>
                    <td className="px-3 py-1.5 text-ink-muted whitespace-nowrap">{d.timestamp.slice(0, 19)}</td>
                    <td className="px-3 py-1.5 text-ink-soft">{d.lat.toFixed(6)}</td>
                    <td className="px-3 py-1.5 text-ink-soft">{d.lon.toFixed(6)}</td>
                    <td className="px-3 py-1.5 text-ink-soft">{d.alt_m.toFixed(1)}</td>
                    <td className="px-3 py-1.5 text-ink-soft">{d.north_m.toFixed(2)}</td>
                    <td className="px-3 py-1.5 text-ink-soft">{d.east_m.toFixed(2)}</td>
                    <td className="px-3 py-1.5 text-ink-soft">{d.heading_deg.toFixed(0)}</td>
                    <td className={`px-3 py-1.5 font-medium ${d.crack_ratio_pct > 10 ? "text-terracotta" : "text-ink-soft"}`}>
                      {d.crack_ratio_pct.toFixed(2)}%
                    </td>
                    <td className="px-3 py-1.5 text-ink-faint truncate max-w-[100px]">
                      {d.mask_filename || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

      </div>
    </div>
  );
}
