import { MapContainer, TileLayer, CircleMarker, Popup, Polyline, useMap } from "react-leaflet";
import type { LatLngExpression } from "leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect } from "react";
import type { Telemetry } from "../hooks/useTelemetry";

export interface DetectionPoint {
  lat: number;
  lon: number;
  crack_ratio_pct: number;
  timestamp: string;
}

interface Props {
  telem?: Telemetry;
  points?: DetectionPoint[];
  /** Show drone position marker (live page only) */
  showDrone?: boolean;
  /** Center on this point if no drone / GPS */
  defaultCenter?: [number, number];
}

const DEFAULT_CENTER: [number, number] = [24.4539, 54.3773]; // Abu Dhabi

function DroneMarker({ telem }: { telem: Telemetry }) {
  const map = useMap();
  useEffect(() => {
    if (telem.lat !== 0 && telem.lon !== 0) {
      map.setView([telem.lat, telem.lon], map.getZoom(), { animate: true });
    }
  }, [telem.lat, telem.lon, map]);

  if (telem.lat === 0 && telem.lon === 0) return null;

  return (
    <CircleMarker
      center={[telem.lat, telem.lon] as LatLngExpression}
      radius={8}
      pathOptions={{ color: "#2B6CB0", fillColor: "#2B6CB0", fillOpacity: 0.9, weight: 2 }}
    >
      <Popup>
        <div className="font-mono text-xs">
          <div className="font-bold">Drone</div>
          <div>{telem.lat.toFixed(6)}, {telem.lon.toFixed(6)}</div>
          <div>Alt: {telem.alt_m.toFixed(1)}m · {telem.mode}</div>
        </div>
      </Popup>
    </CircleMarker>
  );
}

export function MissionMap({ telem, points = [], showDrone = false, defaultCenter }: Props) {
  const center: [number, number] =
    telem && telem.lat !== 0
      ? [telem.lat, telem.lon]
      : defaultCenter ?? DEFAULT_CENTER;

  // Build flight path from points that have GPS
  const pathCoords: LatLngExpression[] = points
    .filter((p) => p.lat !== 0)
    .map((p) => [p.lat, p.lon] as [number, number]);

  return (
    <div className="w-full h-full rounded-md overflow-hidden border border-parchment-darker">
      <MapContainer
        center={center as LatLngExpression}
        zoom={18}
        maxZoom={23}
        style={{ width: "100%", height: "100%" }}
        zoomControl={false}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          maxZoom={23}
          maxNativeZoom={19}
          attribution="&copy; OpenStreetMap contributors"
        />

        {/* Flight path polyline */}
        {pathCoords.length > 1 && (
          <Polyline
            positions={pathCoords}
            pathOptions={{ color: "#C4622D", weight: 2, opacity: 0.5, dashArray: "4 4" }}
          />
        )}

        {/* Detection points — radius ∝ crack_ratio */}
        {points.filter((p) => p.lat !== 0).map((p, i) => {
          const intensity = Math.min(p.crack_ratio_pct / 100, 1);
          const radius = 4 + intensity * 8;
          // terracotta gradient: low = faded, high = deep
          const opacity = 0.4 + intensity * 0.5;
          return (
            <CircleMarker
              key={i}
              center={[p.lat, p.lon] as LatLngExpression}
              radius={radius}
              pathOptions={{
                color: "#C4622D",
                fillColor: "#C4622D",
                fillOpacity: opacity,
                weight: 1,
                opacity: 0.8,
              }}
            >
              <Popup>
                <div className="font-mono text-xs">
                  <div className="font-bold text-terracotta">{p.crack_ratio_pct.toFixed(1)}% coverage</div>
                  <div className="text-ink-muted">{p.timestamp.slice(0, 19)}</div>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}

        {/* Live drone position */}
        {showDrone && telem && <DroneMarker telem={telem} />}
      </MapContainer>
    </div>
  );
}
