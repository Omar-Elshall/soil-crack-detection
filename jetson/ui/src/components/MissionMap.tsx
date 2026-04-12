import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect } from "react";
import type { Telemetry } from "../hooks/useTelemetry";

interface DetectionPoint {
  lat: number;
  lon: number;
  crack_ratio_pct: number;
  timestamp: string;
}

interface Props {
  telem: Telemetry;
  points?: DetectionPoint[];
}

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
      center={[telem.lat, telem.lon]}
      radius={7}
      pathOptions={{ color: "#2B6CB0", fillColor: "#2B6CB0", fillOpacity: 0.9, weight: 2 }}
    >
      <Popup>
        <div className="font-mono text-xs">
          <div>Drone</div>
          <div>{telem.lat.toFixed(6)}, {telem.lon.toFixed(6)}</div>
          <div>Alt: {telem.alt_m.toFixed(1)}m</div>
        </div>
      </Popup>
    </CircleMarker>
  );
}

export function MissionMap({ telem, points = [] }: Props) {
  const center: [number, number] =
    telem.lat !== 0 ? [telem.lat, telem.lon] : [24.4539, 54.3773]; // Abu Dhabi default

  return (
    <div className="w-full h-full rounded-md overflow-hidden border border-parchment-darker">
      <MapContainer
        center={center}
        zoom={18}
        style={{ width: "100%", height: "100%" }}
        zoomControl={false}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; OpenStreetMap contributors'
        />

        {/* Detection points */}
        {points.map((p, i) => {
          const intensity = Math.min(p.crack_ratio_pct / 100, 1);
          const r = Math.floor(196 + (intensity * 0));
          const g = Math.floor(98 - intensity * 98);
          const b = Math.floor(45 - intensity * 45);
          return (
            <CircleMarker
              key={i}
              center={[p.lat, p.lon]}
              radius={5}
              pathOptions={{
                color: `rgb(${r},${g},${b})`,
                fillColor: `rgb(${r},${g},${b})`,
                fillOpacity: 0.7,
                weight: 1,
              }}
            >
              <Popup>
                <div className="font-mono text-xs">
                  <div className="font-bold">{p.crack_ratio_pct.toFixed(1)}% coverage</div>
                  <div>{p.timestamp.slice(0, 19)}</div>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}

        <DroneMarker telem={telem} />
      </MapContainer>
    </div>
  );
}
