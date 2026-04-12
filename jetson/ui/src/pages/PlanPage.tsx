import { useState, useCallback } from "react";
import { MapContainer, TileLayer, useMapEvents, Polyline, Marker, Popup, Rectangle } from "react-leaflet";
import L from "leaflet";
import { uploadMission, startMission, arm, guided, type Waypoint } from "../api/flight";
import { useTelemetry } from "../hooks/useTelemetry";
import {
  Grid3x3, Upload, Play, Trash2, MapPin, AlertTriangle, CheckCircle, Loader
} from "lucide-react";

// ── Lawnmower grid generator ──────────────────────────────────────────────────
// Generates a bidi lawnmower (boustrophedon) pattern over a lat/lon bounding box.
// lane_spacing_m: metres between parallel passes
// heading: 0 = N-S passes (fly E-W strips), 1 = E-W passes (fly N-S strips)
function generateGrid(
  bounds: [[number, number], [number, number]],  // [[minLat,minLon],[maxLat,maxLon]]
  laneSpacingM: number,
  altM: number,
): Waypoint[] {
  const [[minLat, minLon], [maxLat, maxLon]] = bounds;
  const R = 6371000;
  const latSpan = (maxLat - minLat) * (Math.PI / 180) * R;
  const lonSpan = (maxLon - minLon) * (Math.PI / 180) * R * Math.cos(((minLat + maxLat) / 2) * (Math.PI / 180));

  const waypoints: Waypoint[] = [];

  // Use the shorter dimension as the direction we move across
  // Always fly strips parallel to the longer side
  if (lonSpan >= latSpan) {
    // Fly E→W or W→E strips, stepping northward
    const numLanes = Math.max(1, Math.ceil(latSpan / laneSpacingM));
    const latStep = (maxLat - minLat) / numLanes;
    for (let i = 0; i <= numLanes; i++) {
      const lat = minLat + i * latStep;
      const fromLon = i % 2 === 0 ? minLon : maxLon;
      const toLon   = i % 2 === 0 ? maxLon : minLon;
      waypoints.push({ lat, lon: fromLon, alt: altM });
      waypoints.push({ lat, lon: toLon,   alt: altM });
    }
  } else {
    // Fly N→S or S→N strips, stepping eastward
    const numLanes = Math.max(1, Math.ceil(lonSpan / laneSpacingM));
    const lonStep = (maxLon - minLon) / numLanes;
    for (let i = 0; i <= numLanes; i++) {
      const lon = minLon + i * lonStep;
      const fromLat = i % 2 === 0 ? minLat : maxLat;
      const toLat   = i % 2 === 0 ? maxLat : minLat;
      waypoints.push({ lat: fromLat, lon, alt: altM });
      waypoints.push({ lat: toLat,   lon, alt: altM });
    }
  }

  return waypoints;
}

// ── Drone icon ────────────────────────────────────────────────────────────────
const droneIcon = L.divIcon({
  className: "",
  html: `<div style="width:12px;height:12px;border-radius:50%;background:rgb(var(--accent));border:2px solid white;box-shadow:0 0 6px rgba(0,0,0,0.5)"></div>`,
  iconAnchor: [6, 6],
});

const wpIcon = (n: number) => L.divIcon({
  className: "",
  html: `<div style="width:18px;height:18px;border-radius:50%;background:#0891B2;border:2px solid white;display:flex;align-items:center;justify-content:center;font-size:8px;color:white;font-family:monospace;font-weight:bold">${n}</div>`,
  iconAnchor: [9, 9],
});

// ── Click-to-draw map handler ─────────────────────────────────────────────────
function AreaSelector({
  onSelect,
}: {
  onSelect: (bounds: [[number, number], [number, number]]) => void;
}) {
  const [corner1, setCorner1] = useState<[number, number] | null>(null);

  useMapEvents({
    click(e) {
      const pt: [number, number] = [e.latlng.lat, e.latlng.lng];
      if (!corner1) {
        setCorner1(pt);
      } else {
        const minLat = Math.min(corner1[0], pt[0]);
        const maxLat = Math.max(corner1[0], pt[0]);
        const minLon = Math.min(corner1[1], pt[1]);
        const maxLon = Math.max(corner1[1], pt[1]);
        onSelect([[minLat, minLon], [maxLat, maxLon]]);
        setCorner1(null);
      }
    },
  });

  return corner1 ? (
    <Marker position={corner1} icon={L.divIcon({
      className: "",
      html: `<div style="width:10px;height:10px;border-radius:50%;background:#F59E0B;border:2px solid white"></div>`,
      iconAnchor: [5, 5],
    })} />
  ) : null;
}

// ── Main page ─────────────────────────────────────────────────────────────────
type Status = { type: "idle" | "uploading" | "ok" | "error"; msg: string };

export default function PlanPage() {
  const telem = useTelemetry();

  const [bounds, setBounds] = useState<[[number, number], [number, number]] | null>(null);
  const [altM, setAltM]         = useState(4);
  const [speedMs, setSpeedMs]   = useState(2);
  const [overlapPct, setOverlapPct] = useState(60);
  const [waypoints, setWaypoints] = useState<Waypoint[]>([]);
  const [status, setStatus]     = useState<Status>({ type: "idle", msg: "" });
  const [drawing, setDrawing]   = useState(false);

  // Camera footprint at altitude, IMX477 ~62° HFOV
  const footprintM = 2 * altM * Math.tan((62 / 2) * (Math.PI / 180));
  const laneSpacingM = footprintM * (1 - overlapPct / 100);

  const handleSelect = useCallback((b: [[number, number], [number, number]]) => {
    setBounds(b);
    setWaypoints([]);
    setDrawing(false);
  }, []);

  function generate() {
    if (!bounds) return;
    const wps = generateGrid(bounds, Math.max(0.5, laneSpacingM), altM);
    setWaypoints(wps);
  }

  async function handleUpload() {
    if (waypoints.length === 0) return;
    setStatus({ type: "uploading", msg: "Uploading mission…" });
    const res = await uploadMission(waypoints, altM);
    setStatus(res.ok
      ? { type: "ok",    msg: res.message }
      : { type: "error", msg: res.message });
  }

  async function handleStart() {
    setStatus({ type: "uploading", msg: "Setting GUIDED → ARM → AUTO…" });
    const g = await guided();
    if (!g.ok) { setStatus({ type: "error", msg: "GUIDED failed: " + g.message }); return; }
    const a = await arm();
    if (!a.ok) { setStatus({ type: "error", msg: "Arm failed: " + a.message }); return; }
    const s = await startMission();
    setStatus(s.ok
      ? { type: "ok",    msg: "Mission started — AUTO mode" }
      : { type: "error", msg: "Start failed: " + s.message });
  }

  const center: [number, number] = telem.lat !== 0
    ? [telem.lat, telem.lon]
    : [24.423, 54.608]; // AUS campus fallback

  const routeLatLons: [number, number][] = waypoints.map(w => [w.lat, w.lon]);

  const statusIcon =
    status.type === "uploading" ? <Loader size={13} className="animate-spin" /> :
    status.type === "ok"        ? <CheckCircle size={13} /> :
    status.type === "error"     ? <AlertTriangle size={13} /> : null;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">

      {/* Header */}
      <div className="px-6 pt-5 pb-4 border-b border-parchment-darker shrink-0">
        <h1 className="font-display text-2xl font-bold text-ink tracking-tight">Mission Planner</h1>
        <p className="text-xs text-ink-muted mt-1 font-sans">
          Click two points on the map to define the survey area, then generate and upload the grid.
        </p>
      </div>

      <div className="flex-1 flex overflow-hidden">

        {/* Settings panel */}
        <div className="w-72 shrink-0 flex flex-col gap-4 p-4 border-r border-parchment-darker overflow-y-auto">

          {/* Draw area */}
          <div>
            <h3 className="text-[10px] font-mono uppercase tracking-widest text-ink-muted mb-2">Survey Area</h3>
            <button
              onClick={() => { setDrawing(true); setBounds(null); setWaypoints([]); }}
              className={`w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded border text-xs font-mono transition-colors ${
                drawing
                  ? "border-terracotta bg-terracotta/10 text-terracotta"
                  : "border-parchment-darker bg-surface/80 text-ink-soft hover:border-terracotta/40 hover:text-ink"
              }`}
            >
              <MapPin size={13} />
              {drawing ? "Click two corners on map…" : bounds ? "Redraw area" : "Click to draw area"}
            </button>
            {bounds && (
              <div className="mt-2 text-[10px] font-mono text-ink-muted space-y-0.5">
                <div>SW: {bounds[0][0].toFixed(5)}, {bounds[0][1].toFixed(5)}</div>
                <div>NE: {bounds[1][0].toFixed(5)}, {bounds[1][1].toFixed(5)}</div>
              </div>
            )}
          </div>

          {/* Settings */}
          <div>
            <h3 className="text-[10px] font-mono uppercase tracking-widest text-ink-muted mb-2">Flight Parameters</h3>
            <div className="space-y-3">

              <label className="block">
                <span className="text-[10px] font-mono text-ink-muted">Altitude AGL (m)</span>
                <div className="flex items-center gap-2 mt-1">
                  <input type="range" min={2} max={20} step={0.5} value={altM}
                    onChange={e => setAltM(Number(e.target.value))}
                    className="flex-1 accent-terracotta" />
                  <span className="text-xs font-mono text-ink w-10 text-right">{altM}m</span>
                </div>
              </label>

              <label className="block">
                <span className="text-[10px] font-mono text-ink-muted">Speed (m/s)</span>
                <div className="flex items-center gap-2 mt-1">
                  <input type="range" min={0.5} max={5} step={0.5} value={speedMs}
                    onChange={e => setSpeedMs(Number(e.target.value))}
                    className="flex-1 accent-terracotta" />
                  <span className="text-xs font-mono text-ink w-10 text-right">{speedMs} m/s</span>
                </div>
              </label>

              <label className="block">
                <span className="text-[10px] font-mono text-ink-muted">Side Overlap</span>
                <div className="flex items-center gap-2 mt-1">
                  <input type="range" min={20} max={90} step={5} value={overlapPct}
                    onChange={e => setOverlapPct(Number(e.target.value))}
                    className="flex-1 accent-terracotta" />
                  <span className="text-xs font-mono text-ink w-10 text-right">{overlapPct}%</span>
                </div>
              </label>
            </div>
          </div>

          {/* Computed info */}
          <div className="rounded border border-parchment-darker bg-surface/50 p-3 space-y-1.5">
            <div className="flex justify-between text-[10px] font-mono">
              <span className="text-ink-muted">Camera footprint</span>
              <span className="text-ink">{footprintM.toFixed(1)} m</span>
            </div>
            <div className="flex justify-between text-[10px] font-mono">
              <span className="text-ink-muted">Lane spacing</span>
              <span className="text-ink">{Math.max(0.5, laneSpacingM).toFixed(1)} m</span>
            </div>
            <div className="flex justify-between text-[10px] font-mono">
              <span className="text-ink-muted">Waypoints</span>
              <span className="text-ink">{waypoints.length}</span>
            </div>
            {waypoints.length > 0 && (
              <div className="flex justify-between text-[10px] font-mono">
                <span className="text-ink-muted">Est. distance</span>
                <span className="text-ink">
                  {(waypoints.reduce((acc, wp, i) => {
                    if (i === 0) return 0;
                    const prev = waypoints[i - 1];
                    const dlat = (wp.lat - prev.lat) * (Math.PI / 180) * 6371000;
                    const dlon = (wp.lon - prev.lon) * (Math.PI / 180) * 6371000 * Math.cos(wp.lat * Math.PI / 180);
                    return acc + Math.sqrt(dlat * dlat + dlon * dlon);
                  }, 0) / 1000).toFixed(0)} m
                </span>
              </div>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex flex-col gap-2">
            <button
              onClick={generate}
              disabled={!bounds}
              className="flex items-center justify-center gap-2 px-3 py-2 rounded border border-parchment-darker bg-surface/80 text-xs font-mono text-ink-soft hover:text-ink hover:border-terracotta/40 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Grid3x3 size={13} /> Generate Grid
            </button>

            <button
              onClick={handleUpload}
              disabled={waypoints.length === 0 || status.type === "uploading"}
              className="flex items-center justify-center gap-2 px-3 py-2 rounded border border-terracotta/40 bg-terracotta/10 text-xs font-mono text-terracotta hover:bg-terracotta/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Upload size={13} /> Upload to Drone
            </button>

            <button
              onClick={handleStart}
              disabled={status.type !== "ok" || waypoints.length === 0}
              className="flex items-center justify-center gap-2 px-3 py-2 rounded border border-moss/40 bg-moss/10 text-xs font-mono text-moss hover:bg-moss/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Play size={13} /> Arm & Start Mission
            </button>

            {waypoints.length > 0 && (
              <button
                onClick={() => { setWaypoints([]); setBounds(null); setStatus({ type: "idle", msg: "" }); }}
                className="flex items-center justify-center gap-2 px-3 py-2 rounded border border-parchment-darker bg-surface/80 text-xs font-mono text-ink-faint hover:text-ink-muted transition-colors"
              >
                <Trash2 size={12} /> Clear
              </button>
            )}
          </div>

          {/* Status */}
          {status.type !== "idle" && (
            <div className={`flex items-start gap-2 px-3 py-2 rounded border text-[10px] font-mono ${
              status.type === "ok"        ? "border-moss/30 bg-moss/10 text-moss" :
              status.type === "error"     ? "border-red-400/30 bg-red-400/10 text-red-400" :
              "border-parchment-darker bg-surface/50 text-ink-muted"
            }`}>
              <span className="mt-0.5 shrink-0">{statusIcon}</span>
              <span>{status.msg}</span>
            </div>
          )}
        </div>

        {/* Map */}
        <div className="flex-1 relative">
          <MapContainer
            center={center}
            zoom={17}
            className="w-full h-full"
            style={{ cursor: drawing ? "crosshair" : "grab" }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution="© OpenStreetMap"
            />

            {/* Draw mode handler */}
            {drawing && <AreaSelector onSelect={handleSelect} />}

            {/* Selected bounding box */}
            {bounds && (
              <Rectangle
                bounds={[bounds[0], bounds[1]]}
                pathOptions={{ color: "rgb(var(--accent))", weight: 2, dashArray: "6 4", fillOpacity: 0.08 }}
              />
            )}

            {/* Lawnmower route preview */}
            {waypoints.length > 1 && (
              <Polyline
                positions={routeLatLons}
                pathOptions={{ color: "#F59E0B", weight: 2, dashArray: "8 4" }}
              />
            )}

            {/* Waypoint markers (first + last only for legibility) */}
            {waypoints.length > 0 && (
              <>
                <Marker position={[waypoints[0].lat, waypoints[0].lon]} icon={wpIcon(1)}>
                  <Popup><span className="text-xs font-mono">Start</span></Popup>
                </Marker>
                <Marker position={[waypoints[waypoints.length - 1].lat, waypoints[waypoints.length - 1].lon]} icon={wpIcon(waypoints.length)}>
                  <Popup><span className="text-xs font-mono">End · WP {waypoints.length}</span></Popup>
                </Marker>
              </>
            )}

            {/* Drone position */}
            {telem.lat !== 0 && (
              <Marker position={[telem.lat, telem.lon]} icon={droneIcon}>
                <Popup>
                  <span className="text-xs font-mono">
                    Drone · {telem.lat.toFixed(5)}, {telem.lon.toFixed(5)}<br />
                    Alt: {telem.alt_m.toFixed(1)}m · {telem.armed ? "ARMED" : "Disarmed"}
                  </span>
                </Popup>
              </Marker>
            )}
          </MapContainer>

          {/* Map overlay instructions */}
          {drawing && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[1000] pointer-events-none">
              <div className="px-4 py-2 rounded-full bg-black/70 text-white text-xs font-mono backdrop-blur-sm">
                Click corner 1, then click corner 2 to define survey area
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
