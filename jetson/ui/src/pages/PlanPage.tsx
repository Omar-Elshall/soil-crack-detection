import { useState, useCallback } from "react";
import { MapContainer, TileLayer, useMapEvents, Polyline, Polygon, Marker } from "react-leaflet";
import L from "leaflet";
import { uploadMission, startMission, arm, guided, type Waypoint } from "../api/flight";
import { useTelemetry } from "../hooks/useTelemetry";
import { ConfirmModal } from "../components/ConfirmModal";
import {
  Grid3x3, Upload, Play, Trash2, MapPin, AlertTriangle,
  CheckCircle, Loader, FlaskConical, Info, Undo2, Check,
} from "lucide-react";
import { getActive } from "../api/mavlinkSource";

// ── Geometry helpers ────────────────────────────────────────────────────────────

// Proper intersection of two line segments (p1→p2) and (p3→p4).
// Returns true only for proper (non-endpoint) crossings.
function segmentsIntersect(
  p1: [number,number], p2: [number,number],
  p3: [number,number], p4: [number,number],
): boolean {
  const dx1 = p2[0]-p1[0], dy1 = p2[1]-p1[1];
  const dx2 = p4[0]-p3[0], dy2 = p4[1]-p3[1];
  const denom = dx1*dy2 - dy1*dx2;
  if (Math.abs(denom) < 1e-12) return false; // parallel / collinear
  const dx3 = p3[0]-p1[0], dy3 = p3[1]-p1[1];
  const t = (dx3*dy2 - dy3*dx2) / denom;
  const u = (dx3*dy1 - dy3*dx1) / denom;
  // Strictly interior intersections only (exclude shared endpoints)
  return t > 0 && t < 1 && u > 0 && u < 1;
}

// Returns true if the polygon has no self-intersecting edges.
function isSimplePolygon(verts: [number,number][]): boolean {
  const n = verts.length;
  for (let i = 0; i < n; i++) {
    const a1 = verts[i], a2 = verts[(i+1)%n];
    for (let j = i+2; j < n; j++) {
      // Skip the pair (last edge, first edge) — they share vertex 0
      if (i === 0 && j === n-1) continue;
      const b1 = verts[j], b2 = verts[(j+1)%n];
      if (segmentsIntersect(a1, a2, b1, b2)) return false;
    }
  }
  return true;
}

// ── Polygon lawnmower grid ──────────────────────────────────────────────────────

function generatePolygonGrid(
  verts: [number,number][], // [lat, lon] pairs
  laneSpacingM: number,
  altM: number,
): Waypoint[] {
  if (verts.length < 3) return [];

  // Project vertices to local metres (centroid = origin)
  const lat0 = verts.reduce((s, v) => s + v[0], 0) / verts.length;
  const lon0 = verts.reduce((s, v) => s + v[1], 0) / verts.length;
  const cosLat = Math.cos(lat0 * Math.PI / 180);

  const pts: [number,number][] = verts.map(([lat, lon]) => [
    (lon - lon0) * cosLat * 111320, // x = east (m)
    (lat - lat0) * 111320,          // y = north (m)
  ]);

  const ys  = pts.map(p => p[1]);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spacing = Math.max(0.5, laneSpacingM);
  const n = pts.length;
  const waypoints: Waypoint[] = [];

  // Convert local metres back to lat/lon
  const toLonLat = (xm: number, ym: number): [number,number] => [
    lat0 + ym / 111320,
    lon0 + xm / (cosLat * 111320),
  ];

  // Build scanline y-values, centred within the polygon extent
  const scanYs: number[] = [];
  for (let y = minY + spacing * 0.5; y < maxY; y += spacing) scanYs.push(y);
  if (scanYs.length === 0) scanYs.push((minY + maxY) / 2); // at least one pass

  scanYs.forEach((y, rowIdx) => {
    // Scanline intersection with each polygon edge
    const xs: number[] = [];
    for (let i = 0; i < n; i++) {
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[(i+1) % n];
      // Standard even–odd rule: count edge if scanline crosses it (not if y = upper vertex)
      if ((y1 < y && y <= y2) || (y2 < y && y <= y1)) {
        const t = (y - y1) / (y2 - y1);
        xs.push(x1 + t * (x2 - x1));
      }
    }
    xs.sort((a, b) => a - b);

    // Each consecutive pair of x values is an interior pass segment
    for (let k = 0; k+1 < xs.length; k += 2) {
      const [latA, lonA] = toLonLat(xs[k],   y);
      const [latB, lonB] = toLonLat(xs[k+1], y);
      // Alternate direction each row for lawnmower pattern
      if (rowIdx % 2 === 0) {
        waypoints.push({ lat: latA, lon: lonA, alt: altM });
        waypoints.push({ lat: latB, lon: lonB, alt: altM });
      } else {
        waypoints.push({ lat: latB, lon: lonB, alt: altM });
        waypoints.push({ lat: latA, lon: lonA, alt: altM });
      }
    }
  });

  return waypoints;
}

// ── Map icons ──────────────────────────────────────────────────────────────────

const droneIcon = L.divIcon({
  className: "",
  html: `<div style="width:12px;height:12px;border-radius:50%;background:#06B6D4;border:2.5px solid white;box-shadow:0 0 8px rgba(6,182,212,0.6)"></div>`,
  iconAnchor: [6, 6],
});

const startIcon = L.divIcon({
  className: "",
  html: `<div style="width:20px;height:20px;border-radius:50%;background:#22C55E;border:2px solid white;display:flex;align-items:center;justify-content:center;font-size:9px;color:white;font-weight:bold">S</div>`,
  iconAnchor: [10, 10],
});

const endIcon = L.divIcon({
  className: "",
  html: `<div style="width:20px;height:20px;border-radius:50%;background:#EF4444;border:2px solid white;display:flex;align-items:center;justify-content:center;font-size:9px;color:white;font-weight:bold">E</div>`,
  iconAnchor: [10, 10],
});

const firstVertexIcon = L.divIcon({
  className: "",
  html: `<div style="width:14px;height:14px;border-radius:50%;background:#F59E0B;border:2.5px solid white;box-shadow:0 0 6px rgba(245,158,11,0.6);display:flex;align-items:center;justify-content:center;font-size:7px;color:white;font-weight:bold">1</div>`,
  iconAnchor: [7, 7],
});

const vertexIcon = L.divIcon({
  className: "",
  html: `<div style="width:9px;height:9px;border-radius:50%;background:#06B6D4;border:2px solid white"></div>`,
  iconAnchor: [4, 4],
});

// ── Polygon click-to-draw handler ──────────────────────────────────────────────
function PolygonDrawer({ onAdd }: { onAdd: (pt: [number,number]) => void }) {
  useMapEvents({
    click(e) {
      onAdd([e.latlng.lat, e.latlng.lng]);
    },
  });
  return null;
}

// ── Status badge ───────────────────────────────────────────────────────────────
type StatusState = "idle" | "uploading" | "ok" | "error";
interface Status { type: StatusState; msg: string }

function StatusBadge({ status }: { status: Status }) {
  if (status.type === "idle") return null;
  const style =
    status.type === "ok"       ? "border-moss/30 bg-moss/10 text-moss" :
    status.type === "error"    ? "border-red-400/30 bg-red-400/10 text-red-400" :
    "border-parchment-darker bg-surface/50 text-ink-muted";
  const Icon =
    status.type === "uploading" ? Loader :
    status.type === "ok"        ? CheckCircle : AlertTriangle;
  return (
    <div className={`flex items-start gap-2 px-3 py-2 rounded-md border text-[10px] font-mono ${style}`}>
      <Icon size={12} className={`mt-0.5 shrink-0 ${status.type==="uploading" ? "animate-spin" : ""}`} />
      <span>{status.msg}</span>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────
const MISSION_CHECKLIST = [
  "All personnel are at least 10 m from the drone",
  "Propellers are securely attached and undamaged",
  "Hardware safety switch has been pressed",
  "Battery is charged and secured",
  "Survey area on the map matches the intended field",
  "Area is free of obstacles and overhead obstructions",
  "You have direct line-of-sight to the drone at all times",
];

export default function PlanPage() {
  const telem = useTelemetry();

  // Polygon state
  const [vertices, setVertices]         = useState<[number,number][]>([]);
  const [drawing, setDrawing]           = useState(false);
  const [polygonClosed, setPolygonClosed] = useState(false);
  const [polyError, setPolyError]       = useState<string | null>(null);

  // Flight parameters
  const [altM, setAltM]             = useState(4);
  const [speedMs, setSpeedMs]       = useState(2);
  const [overlapPct, setOverlapPct] = useState(60);

  // Mission state
  const [waypoints, setWaypoints]   = useState<Waypoint[]>([]);
  const [status, setStatus]         = useState<Status>({ type: "idle", msg: "" });
  const [testStatus, setTestStatus] = useState<Status>({ type: "idle", msg: "" });
  const [showMissionConfirm, setShowMissionConfirm] = useState(false);

  const footprintM   = 2 * altM * Math.tan((62 / 2) * (Math.PI / 180));
  const laneSpacingM = footprintM * (1 - overlapPct / 100);

  // ── Polygon actions ──────────────────────────────────────────────────────────
  const handleAddVertex = useCallback((pt: [number,number]) => {
    setVertices(prev => [...prev, pt]);
    setPolyError(null);
    setWaypoints([]);
  }, []);

  function handleClosePolygon() {
    if (vertices.length < 3) return;
    if (!isSimplePolygon(vertices)) {
      setPolyError("Polygon has self-intersecting edges. Undo the last vertex or clear and redraw.");
      return;
    }
    setPolygonClosed(true);
    setDrawing(false);
    setPolyError(null);
  }

  function handleUndoVertex() {
    setVertices(prev => prev.slice(0, -1));
    setPolyError(null);
    setWaypoints([]);
  }

  function handleClear() {
    setVertices([]);
    setDrawing(false);
    setPolygonClosed(false);
    setPolyError(null);
    setWaypoints([]);
    setStatus({ type: "idle", msg: "" });
  }

  function handleStartDrawing() {
    handleClear();
    setDrawing(true);
  }

  // ── Grid & mission actions ───────────────────────────────────────────────────
  function generate() {
    if (!polygonClosed || vertices.length < 3) return;
    setWaypoints(generatePolygonGrid(vertices, laneSpacingM, altM));
    setStatus({ type: "idle", msg: "" });
  }

  async function handleUpload() {
    if (!waypoints.length) return;
    setStatus({ type: "uploading", msg: "Uploading mission to drone…" });
    const res = await uploadMission(waypoints, altM);
    setStatus(res.ok
      ? { type: "ok",    msg: res.message }
      : { type: "error", msg: res.message });
  }

  async function handleStart() {
    setShowMissionConfirm(false);
    setStatus({ type: "uploading", msg: "Setting GUIDED → arming → AUTO…" });
    const g = await guided();
    if (!g.ok) { setStatus({ type: "error", msg: "GUIDED failed: " + g.message }); return; }
    const a = await arm();
    if (!a.ok) { setStatus({ type: "error", msg: "Arm failed: " + a.message }); return; }
    const s = await startMission();
    setStatus(s.ok
      ? { type: "ok",    msg: "Mission started — AUTO mode engaged" }
      : { type: "error", msg: "Failed: " + s.message });
  }

  async function handleTestFlight() {
    setTestStatus({ type: "uploading", msg: "Test flight running (~20s)…" });
    try {
      const res = await fetch(`${getActive().base}/command/test-flight`, { method: "POST" });
      const data = await res.json();
      setTestStatus(data.ok
        ? { type: "ok",    msg: data.message }
        : { type: "error", msg: data.message });
    } catch {
      setTestStatus({ type: "error", msg: "Request failed" });
    }
    setTimeout(() => setTestStatus({ type: "idle", msg: "" }), 5000);
  }

  // ── Derived stats ────────────────────────────────────────────────────────────
  const center: [number,number] = telem.lat !== 0 ? [telem.lat, telem.lon] : [24.423, 54.608];

  const estDistM = waypoints.reduce((acc, wp, i) => {
    if (i === 0) return 0;
    const prev = waypoints[i - 1];
    const dlat = (wp.lat - prev.lat) * (Math.PI / 180) * 6371000;
    const dlon = (wp.lon - prev.lon) * (Math.PI / 180) * 6371000 * Math.cos(wp.lat * Math.PI / 180);
    return acc + Math.sqrt(dlat*dlat + dlon*dlon);
  }, 0);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">

      {/* Header */}
      <div className="px-6 pt-5 pb-4 border-b border-parchment-darker shrink-0">
        <h1 className="font-display text-2xl font-bold text-ink tracking-tight">Mission Planner</h1>
        <p className="text-xs text-ink-muted mt-1 font-sans">
          Plot a survey polygon on the map, generate a lawnmower grid inside it, then upload and fly autonomously.
        </p>
      </div>

      <div className="flex-1 flex overflow-hidden">

        {/* ── Settings panel ────────────────────────────────────────────── */}
        <div className="w-72 shrink-0 flex flex-col gap-4 p-4 border-r border-parchment-darker overflow-y-auto">

          {/* GPS warning */}
          {telem.connected && telem.gps_fix < 3 && (
            <div className="flex items-start gap-2 px-3 py-2.5 rounded-md border border-amber-500/25 bg-amber-500/8 text-[10px] font-mono text-amber-500">
              <AlertTriangle size={12} className="mt-0.5 shrink-0" />
              <span>No GPS fix (fix={telem.gps_fix}). Autonomous flight requires GPS 3D fix.</span>
            </div>
          )}

          {/* COM port note */}
          <div className="flex items-start gap-2 px-3 py-2 rounded-md border border-parchment-darker bg-surface/50 text-[10px] font-mono text-ink-faint">
            <Info size={11} className="mt-0.5 shrink-0" />
            <span>Disconnect Mission Planner before uploading — both cannot hold the serial port simultaneously.</span>
          </div>

          {/* Step 1 — Draw polygon */}
          <div>
            <h3 className="text-[10px] font-mono uppercase tracking-widest text-ink-muted mb-2">
              <span className="inline-flex w-4 h-4 rounded-full bg-terracotta/20 text-terracotta items-center justify-center mr-1.5 text-[9px] font-bold">1</span>
              Survey Polygon
            </h3>

            {/* Draw / Redraw button */}
            <button
              onClick={handleStartDrawing}
              className={`w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-md border text-xs font-mono font-medium transition-colors ${
                drawing
                  ? "border-terracotta bg-terracotta/15 text-terracotta"
                  : polygonClosed
                  ? "border-moss/40 bg-moss/10 text-moss"
                  : "border-parchment-darker bg-surface/80 text-ink-soft hover:border-terracotta/30 hover:text-ink"
              }`}
            >
              <MapPin size={13} />
              {drawing
                ? `Placing vertices… (${vertices.length} placed)`
                : polygonClosed
                ? `Polygon set (${vertices.length} pts) — click to redraw`
                : "Click to draw survey polygon"}
            </button>

            {/* Drawing controls */}
            {drawing && (
              <div className="mt-2 flex flex-col gap-1.5">
                {/* Close polygon — only enabled when ≥3 vertices */}
                <button
                  onClick={handleClosePolygon}
                  disabled={vertices.length < 3}
                  className="flex items-center justify-center gap-2 px-3 py-2 rounded-md border text-xs font-mono font-bold transition-colors border-moss/50 bg-moss/15 text-moss hover:bg-moss/25 disabled:opacity-35 disabled:cursor-not-allowed"
                >
                  <Check size={13} /> Close Polygon
                </button>
                <div className="flex gap-1.5">
                  <button
                    onClick={handleUndoVertex}
                    disabled={vertices.length === 0}
                    className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md border text-[10px] font-mono border-parchment-darker text-ink-muted hover:text-ink disabled:opacity-35 disabled:cursor-not-allowed"
                  >
                    <Undo2 size={11} /> Undo
                  </button>
                  <button
                    onClick={handleClear}
                    className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md border text-[10px] font-mono border-parchment-darker text-ink-faint hover:text-ink-muted"
                  >
                    <Trash2 size={11} /> Clear
                  </button>
                </div>
              </div>
            )}

            {/* Validation error */}
            {polyError && (
              <div className="mt-2 flex items-start gap-2 px-3 py-2 rounded-md border border-red-400/30 bg-red-400/10 text-[10px] font-mono text-red-400">
                <AlertTriangle size={11} className="mt-0.5 shrink-0" />
                <span>{polyError}</span>
              </div>
            )}

            {/* Vertex summary when closed */}
            {polygonClosed && vertices.length >= 3 && (
              <div className="mt-2 text-[10px] font-mono text-ink-muted space-y-0.5 pl-1">
                {vertices.slice(0, 4).map((v, i) => (
                  <div key={i}>pt{i+1}: {v[0].toFixed(5)}, {v[1].toFixed(5)}</div>
                ))}
                {vertices.length > 4 && <div className="text-ink-faint">…+{vertices.length-4} more</div>}
              </div>
            )}
          </div>

          {/* Step 2 — Parameters */}
          <div>
            <h3 className="text-[10px] font-mono uppercase tracking-widest text-ink-muted mb-2">
              <span className="inline-flex w-4 h-4 rounded-full bg-terracotta/20 text-terracotta items-center justify-center mr-1.5 text-[9px] font-bold">2</span>
              Flight Parameters
            </h3>
            <div className="space-y-3">
              {[
                { label: "Altitude AGL", min: 2,   max: 20, step: 0.5, value: altM,       set: setAltM,       unit: "m"   },
                { label: "Speed",        min: 0.5,  max: 5,  step: 0.5, value: speedMs,    set: setSpeedMs,    unit: "m/s" },
                { label: "Side Overlap", min: 20,   max: 90, step: 5,   value: overlapPct, set: setOverlapPct, unit: "%"   },
              ].map(({ label, min, max, step, value, set, unit }) => (
                <label key={label} className="block">
                  <div className="flex justify-between text-[10px] font-mono text-ink-muted mb-1">
                    <span>{label}</span>
                    <span className="text-ink font-medium">{value}{unit}</span>
                  </div>
                  <input type="range" min={min} max={max} step={step} value={value}
                    onChange={e => set(Number(e.target.value))}
                    className="w-full accent-terracotta h-1" />
                </label>
              ))}
            </div>
          </div>

          {/* Computed stats */}
          <div className="rounded-md border border-parchment-darker bg-surface/50 p-3 space-y-1.5 text-[10px] font-mono">
            {[
              ["Camera footprint", `${footprintM.toFixed(1)} m`],
              ["Lane spacing",     `${Math.max(0.5, laneSpacingM).toFixed(1)} m`],
              ["Waypoints",        waypoints.length.toString()],
              ["Est. distance",    waypoints.length > 0 ? `${estDistM.toFixed(0)} m` : "—"],
              ["Est. duration",    waypoints.length > 0 && speedMs > 0 ? `${(estDistM/speedMs/60).toFixed(1)} min` : "—"],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-ink-muted">{k}</span>
                <span className="text-ink">{v}</span>
              </div>
            ))}
          </div>

          {/* Step 3 — Generate & upload */}
          <div>
            <h3 className="text-[10px] font-mono uppercase tracking-widest text-ink-muted mb-2">
              <span className="inline-flex w-4 h-4 rounded-full bg-terracotta/20 text-terracotta items-center justify-center mr-1.5 text-[9px] font-bold">3</span>
              Generate & Upload
            </h3>
            <div className="flex flex-col gap-2">

              <button
                onClick={generate}
                disabled={!polygonClosed}
                className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-md border text-xs font-mono font-medium transition-colors border-parchment-darker bg-surface/80 text-ink-soft hover:text-ink hover:bg-parchment-dark disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Grid3x3 size={13} /> Generate Polygon Grid
              </button>

              <button
                onClick={handleUpload}
                disabled={!waypoints.length || status.type === "uploading"}
                className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-md border text-xs font-mono font-bold transition-colors border-terracotta/50 bg-terracotta/15 text-terracotta hover:bg-terracotta/25 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Upload size={13} /> Upload Mission to Drone
              </button>

              <button
                onClick={() => setShowMissionConfirm(true)}
                disabled={status.type !== "ok" || !waypoints.length}
                className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-md border text-xs font-mono font-bold transition-colors border-moss/50 bg-moss/15 text-moss hover:bg-moss/25 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Play size={13} /> Arm & Start Mission
              </button>

              {waypoints.length > 0 && (
                <button
                  onClick={handleClear}
                  className="flex items-center justify-center gap-2 px-3 py-2 rounded-md border text-xs font-mono text-ink-faint border-parchment-darker hover:text-ink-muted transition-colors"
                >
                  <Trash2 size={12} /> Clear
                </button>
              )}
            </div>
          </div>

          <StatusBadge status={status} />

          {/* Test flight */}
          <div className="border-t border-parchment-darker pt-4">
            <h3 className="text-[10px] font-mono uppercase tracking-widest text-ink-muted mb-1">Safety Test</h3>
            <p className="text-[10px] text-ink-faint font-sans mb-2.5">
              Hover test — takes off 2 m, holds for 5 s, then lands. No GPS required. Use to verify motors and control before a real mission.
            </p>
            <button
              onClick={handleTestFlight}
              disabled={testStatus.type === "uploading"}
              className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-md border text-xs font-mono font-bold transition-colors border-sky/40 bg-sky/10 text-sky hover:bg-sky/20 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {testStatus.type === "uploading"
                ? <><Loader size={13} className="animate-spin" /> Running…</>
                : <><FlaskConical size={13} /> Run Hover Test</>
              }
            </button>
            <StatusBadge status={testStatus} />
          </div>
        </div>

        {/* ── Map ───────────────────────────────────────────────────────── */}
        <div className="flex-1 relative">
          <MapContainer
            center={center}
            zoom={17}
            className="w-full h-full"
            style={{ cursor: drawing ? "crosshair" : "grab" }}
          >
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="© OpenStreetMap" />

            {/* Click handler while drawing */}
            {drawing && <PolygonDrawer onAdd={handleAddVertex} />}

            {/* Polygon outline (dashed while drawing, solid when closed) */}
            {!polygonClosed && vertices.length >= 2 && (
              <Polygon
                positions={vertices}
                pathOptions={{ color: "rgb(6,182,212)", weight: 2, dashArray: "6 4", fillOpacity: 0.06 }}
              />
            )}
            {polygonClosed && vertices.length >= 3 && (
              <Polygon
                positions={vertices}
                pathOptions={{ color: "rgb(6,182,212)", weight: 2, fillOpacity: 0.10, fillColor: "rgb(6,182,212)" }}
              />
            )}

            {/* Vertex markers while drawing */}
            {!polygonClosed && vertices.map((v, i) => (
              <Marker
                key={i}
                position={v}
                icon={i === 0 ? firstVertexIcon : vertexIcon}
              />
            ))}

            {/* Lawnmower flight path */}
            {waypoints.length > 1 && (
              <Polyline
                positions={waypoints.map(w => [w.lat, w.lon] as [number,number])}
                pathOptions={{ color: "#F59E0B", weight: 2, opacity: 0.85 }}
              />
            )}

            {/* Start / end markers */}
            {waypoints.length > 0 && (
              <>
                <Marker position={[waypoints[0].lat, waypoints[0].lon]} icon={startIcon} />
                <Marker position={[waypoints[waypoints.length-1].lat, waypoints[waypoints.length-1].lon]} icon={endIcon} />
              </>
            )}

            {/* Live drone position */}
            {telem.lat !== 0 && (
              <Marker position={[telem.lat, telem.lon]} icon={droneIcon} />
            )}
          </MapContainer>

          {/* Drawing hint overlay */}
          {drawing && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[1000] pointer-events-none">
              <div className="px-4 py-2 rounded-full text-white text-xs font-mono bg-black/70 backdrop-blur-sm whitespace-nowrap">
                {vertices.length < 3
                  ? `Click to place vertices (${vertices.length}/3 minimum)`
                  : `${vertices.length} vertices — press Close Polygon when done`}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Mission start confirmation */}
      {showMissionConfirm && (
        <ConfirmModal
          title="Start Autonomous Mission"
          description={`The drone will arm, take off to ${altM} m, and fly ${waypoints.length} waypoints autonomously at ${speedMs} m/s. Estimated flight time: ${(estDistM/speedMs/60).toFixed(1)} min.`}
          checklist={MISSION_CHECKLIST}
          confirmLabel="ARM & START MISSION"
          confirmClass="bg-moss hover:bg-green-600 border border-moss"
          warning="The drone will fly autonomously. You are responsible for maintaining safe operations at all times."
          onConfirm={handleStart}
          onCancel={() => setShowMissionConfirm(false)}
        />
      )}
    </div>
  );
}
