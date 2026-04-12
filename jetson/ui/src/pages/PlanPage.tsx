import { useState, useCallback } from "react";
import { MapContainer, TileLayer, useMapEvents, Polyline, Marker, Rectangle } from "react-leaflet";
import L from "leaflet";
import { uploadMission, startMission, arm, guided, type Waypoint } from "../api/flight";
import { useTelemetry } from "../hooks/useTelemetry";
import { ConfirmModal } from "../components/ConfirmModal";
import {
  Grid3x3, Upload, Play, Trash2, MapPin, AlertTriangle,
  CheckCircle, Loader, FlaskConical, Info,
} from "lucide-react";
import { API } from "../api/config";

// ── Lawnmower grid ─────────────────────────────────────────────────────────────
function generateGrid(
  bounds: [[number, number], [number, number]],
  laneSpacingM: number,
  altM: number,
): Waypoint[] {
  const [[minLat, minLon], [maxLat, maxLon]] = bounds;
  const R = 6371000;
  const latSpan = (maxLat - minLat) * (Math.PI / 180) * R;
  const lonSpan = (maxLon - minLon) * (Math.PI / 180) * R * Math.cos(((minLat + maxLat) / 2) * (Math.PI / 180));
  const waypoints: Waypoint[] = [];
  const spacing = Math.max(0.5, laneSpacingM);

  if (lonSpan >= latSpan) {
    const numLanes = Math.max(1, Math.ceil(latSpan / spacing));
    const latStep = (maxLat - minLat) / numLanes;
    for (let i = 0; i <= numLanes; i++) {
      const lat = minLat + i * latStep;
      waypoints.push({ lat, lon: i % 2 === 0 ? minLon : maxLon, alt: altM });
      waypoints.push({ lat, lon: i % 2 === 0 ? maxLon : minLon, alt: altM });
    }
  } else {
    const numLanes = Math.max(1, Math.ceil(lonSpan / spacing));
    const lonStep = (maxLon - minLon) / numLanes;
    for (let i = 0; i <= numLanes; i++) {
      const lon = minLon + i * lonStep;
      waypoints.push({ lat: i % 2 === 0 ? minLat : maxLat, lon, alt: altM });
      waypoints.push({ lat: i % 2 === 0 ? maxLat : minLat, lon, alt: altM });
    }
  }
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

// ── Click-to-draw ──────────────────────────────────────────────────────────────
function AreaSelector({ onSelect }: { onSelect: (b: [[number,number],[number,number]]) => void }) {
  const [c1, setC1] = useState<[number,number] | null>(null);
  useMapEvents({
    click(e) {
      const pt: [number,number] = [e.latlng.lat, e.latlng.lng];
      if (!c1) { setC1(pt); }
      else {
        onSelect([[Math.min(c1[0],pt[0]),Math.min(c1[1],pt[1])],[Math.max(c1[0],pt[0]),Math.max(c1[1],pt[1])]]);
        setC1(null);
      }
    },
  });
  return c1 ? <Marker position={c1} icon={L.divIcon({ className:"", html:`<div style="width:10px;height:10px;border-radius:50%;background:#F59E0B;border:2px solid white"></div>`, iconAnchor:[5,5] })} /> : null;
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

  const [bounds, setBounds]         = useState<[[number,number],[number,number]] | null>(null);
  const [drawing, setDrawing]       = useState(false);
  const [altM, setAltM]             = useState(4);
  const [speedMs, setSpeedMs]       = useState(2);
  const [overlapPct, setOverlapPct] = useState(60);
  const [waypoints, setWaypoints]   = useState<Waypoint[]>([]);
  const [status, setStatus]         = useState<Status>({ type: "idle", msg: "" });
  const [testStatus, setTestStatus] = useState<Status>({ type: "idle", msg: "" });

  const [showMissionConfirm, setShowMissionConfirm] = useState(false);

  const footprintM   = 2 * altM * Math.tan((62 / 2) * (Math.PI / 180));
  const laneSpacingM = footprintM * (1 - overlapPct / 100);

  const handleSelect = useCallback((b: [[number,number],[number,number]]) => {
    setBounds(b); setWaypoints([]); setDrawing(false);
  }, []);

  function generate() {
    if (!bounds) return;
    setWaypoints(generateGrid(bounds, laneSpacingM, altM));
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
      ? { type: "ok",    msg: "Mission started ��� AUTO mode engaged" }
      : { type: "error", msg: "Failed: " + s.message });
  }

  async function handleTestFlight() {
    setTestStatus({ type: "uploading", msg: "Test flight running (~20s)…" });
    try {
      const res = await fetch(`${API.mavlink}/command/test-flight`, { method: "POST" });
      const data = await res.json();
      setTestStatus(data.ok
        ? { type: "ok",    msg: data.message }
        : { type: "error", msg: data.message });
    } catch {
      setTestStatus({ type: "error", msg: "Request failed" });
    }
    setTimeout(() => setTestStatus({ type: "idle", msg: "" }), 5000);
  }

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
          Define a survey area, generate a lawnmower grid, then upload and fly autonomously.
        </p>
      </div>

      <div className="flex-1 flex overflow-hidden">

        {/* ── Settings panel ────────────────────────────────────────────── */}
        <div className="w-72 shrink-0 flex flex-col gap-4 p-4 border-r border-parchment-darker overflow-y-auto">

          {/* GPS warning if no fix */}
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

          {/* Step 1 — Draw area */}
          <div>
            <h3 className="text-[10px] font-mono uppercase tracking-widest text-ink-muted mb-2">
              <span className="inline-flex w-4 h-4 rounded-full bg-terracotta/20 text-terracotta items-center justify-center mr-1.5 text-[9px] font-bold">1</span>
              Survey Area
            </h3>
            <button
              onClick={() => { setDrawing(true); setBounds(null); setWaypoints([]); }}
              className={`w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-md border text-xs font-mono font-medium transition-colors ${
                drawing
                  ? "border-terracotta bg-terracotta/15 text-terracotta"
                  : bounds
                  ? "border-moss/40 bg-moss/10 text-moss"
                  : "border-parchment-darker bg-surface/80 text-ink-soft hover:border-terracotta/30 hover:text-ink"
              }`}
            >
              <MapPin size={13} />
              {drawing ? "Click two corners on map…" : bounds ? "Area selected — click to redraw" : "Click to draw survey area"}
            </button>
            {bounds && (
              <div className="mt-2 text-[10px] font-mono text-ink-muted space-y-0.5 pl-1">
                <div>SW {bounds[0][0].toFixed(5)}, {bounds[0][1].toFixed(5)}</div>
                <div>NE {bounds[1][0].toFixed(5)}, {bounds[1][1].toFixed(5)}</div>
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
                { label: "Altitude AGL", min: 2, max: 20, step: 0.5, value: altM,        set: setAltM,        unit: "m"   },
                { label: "Speed",        min: 0.5, max: 5, step: 0.5, value: speedMs,     set: setSpeedMs,     unit: "m/s" },
                { label: "Side Overlap", min: 20,  max: 90, step: 5,  value: overlapPct,  set: setOverlapPct,  unit: "%"   },
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

          {/* Step 3 — Generate */}
          <div>
            <h3 className="text-[10px] font-mono uppercase tracking-widest text-ink-muted mb-2">
              <span className="inline-flex w-4 h-4 rounded-full bg-terracotta/20 text-terracotta items-center justify-center mr-1.5 text-[9px] font-bold">3</span>
              Generate & Upload
            </h3>
            <div className="flex flex-col gap-2">

              {/* Generate — neutral */}
              <button
                onClick={generate}
                disabled={!bounds}
                className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-md border text-xs font-mono font-medium transition-colors border-parchment-darker bg-surface/80 text-ink-soft hover:text-ink hover:bg-parchment-dark disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Grid3x3 size={13} /> Generate Lawnmower Grid
              </button>

              {/* Upload — accent/primary */}
              <button
                onClick={handleUpload}
                disabled={!waypoints.length || status.type === "uploading"}
                className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-md border text-xs font-mono font-bold transition-colors border-terracotta/50 bg-terracotta/15 text-terracotta hover:bg-terracotta/25 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Upload size={13} /> Upload Mission to Drone
              </button>

              {/* Start — green, requires upload success */}
              <button
                onClick={() => setShowMissionConfirm(true)}
                disabled={status.type !== "ok" || !waypoints.length}
                className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-md border text-xs font-mono font-bold transition-colors border-moss/50 bg-moss/15 text-moss hover:bg-moss/25 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Play size={13} /> Arm & Start Mission
              </button>

              {waypoints.length > 0 && (
                <button
                  onClick={() => { setWaypoints([]); setBounds(null); setStatus({ type: "idle", msg: "" }); }}
                  className="flex items-center justify-center gap-2 px-3 py-2 rounded-md border text-xs font-mono text-ink-faint border-parchment-darker hover:text-ink-muted transition-colors"
                >
                  <Trash2 size={12} /> Clear
                </button>
              )}
            </div>
          </div>

          <StatusBadge status={status} />

          {/* Step 4 — Test flight */}
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

            {drawing && <AreaSelector onSelect={handleSelect} />}

            {bounds && (
              <Rectangle
                bounds={[bounds[0], bounds[1]]}
                pathOptions={{ color: "rgb(6,182,212)", weight: 2, dashArray: "6 4", fillOpacity: 0.07 }}
              />
            )}

            {waypoints.length > 1 && (
              <Polyline
                positions={waypoints.map(w => [w.lat, w.lon] as [number,number])}
                pathOptions={{ color: "#F59E0B", weight: 2, opacity: 0.85 }}
              />
            )}

            {waypoints.length > 0 && (
              <>
                <Marker position={[waypoints[0].lat, waypoints[0].lon]} icon={startIcon} />
                <Marker position={[waypoints[waypoints.length-1].lat, waypoints[waypoints.length-1].lon]} icon={endIcon} />
              </>
            )}

            {telem.lat !== 0 && (
              <Marker position={[telem.lat, telem.lon]} icon={droneIcon} />
            )}
          </MapContainer>

          {drawing && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[1000] pointer-events-none">
              <div className="px-4 py-2 rounded-full text-white text-xs font-mono bg-black/70 backdrop-blur-sm">
                Click corner 1, then corner 2
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
