import { useEffect, useRef } from "react";
import { useTelemetry } from "../hooks/useTelemetry";
import { useDetections } from "../hooks/useDetections";
import { useMissions } from "../hooks/useMissions";
import { useMissionLogger } from "../hooks/useMissionLogger";
import { CameraFeed } from "../components/CameraFeed";
import { TelemetryPanel } from "../components/TelemetryPanel";
import { CrackRatioChart } from "../components/CrackRatioChart";
import { CrackLog } from "../components/CrackLog";
import { FlightControls } from "../components/FlightControls";
import { MissionControl } from "../components/MissionControl";
import { MissionMap, type DetectionPoint } from "../components/MissionMap";
import { StatusBar } from "../components/StatusBar";
import { MetricCard } from "../components/MetricCard";

export default function LivePage() {
  const telem = useTelemetry();
  const { latest, history } = useDetections();
  const { activeMissionId, loading, start, stop } = useMissions();

  useMissionLogger(activeMissionId, latest, telem);

  const mapPointsRef = useRef<DetectionPoint[]>([]);
  const prevMissionRef = useRef<string | null>(null);

  if (activeMissionId !== prevMissionRef.current) {
    if (activeMissionId) mapPointsRef.current = [];
    prevMissionRef.current = activeMissionId;
  }

  useEffect(() => {
    if (!activeMissionId || !latest || telem.lat === 0) return;
    mapPointsRef.current = [
      ...mapPointsRef.current.slice(-499),
      { lat: telem.lat, lon: telem.lon, crack_ratio_pct: latest.crack_ratio_pct, timestamp: new Date().toISOString() },
    ];
  }, [latest, activeMissionId, telem.lat, telem.lon]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <StatusBar statusMessages={telem.status_messages} />

      <div className="flex-1 flex overflow-hidden">

        {/* Left — camera + metrics + log — strictly no overflow */}
        <div className="w-72 shrink-0 flex flex-col gap-2 p-3 overflow-hidden bg-parchment-dark">

          <div className="shrink-0 pb-1">
            <h1 className="font-display text-xl font-bold text-ink tracking-tight">Live Survey</h1>
            <p className="text-[10px] font-mono text-ink-muted uppercase tracking-widest mt-0.5">
              EfficientCrackNet · Real-time
            </p>
          </div>

          {/* Camera fixed height — never aspect-ratio grows the column */}
          <div className="shrink-0">
            <CameraFeed />
          </div>

          <div className="shrink-0 grid grid-cols-2 gap-2">
            <MetricCard label="Coverage" value={latest?.crack_ratio_pct.toFixed(1) ?? "—"} unit="%" accent={(latest?.crack_ratio_pct ?? 0) > 10} />
            <MetricCard label="FPS"      value={latest?.fps.toFixed(1) ?? "—"} unit="fps" />
          </div>

          <div className="shrink-0 rounded-md border border-parchment-darker bg-surface/70 p-2">
            <div className="text-[9px] font-mono text-ink-faint uppercase tracking-widest mb-1">Coverage · last 300 frames</div>
            <CrackRatioChart history={history} />
          </div>

          <div className="shrink-0">
            <MissionControl activeMissionId={activeMissionId} loading={loading} onStart={start} onStop={stop} />
          </div>

          {/* CrackLog fills remaining space and scrolls internally */}
          <div className="flex-1 min-h-0">
            <CrackLog history={history} telem={telem} />
          </div>
        </div>

        {/* Center — map */}
        <div className="flex-1 relative p-3">
          <MissionMap telem={telem} points={mapPointsRef.current} showDrone />
        </div>

        {/* Right — telemetry + flight */}
        <div className="w-64 shrink-0 flex flex-col gap-3 p-3 overflow-y-auto bg-parchment-dark">
          <div className="shrink-0 pt-1">
            <h2 className="font-display text-base font-bold text-ink tracking-tight">Telemetry</h2>
          </div>

          <TelemetryPanel telem={telem} />

          <div className="border-t border-parchment-darker pt-3">
            <h2 className="font-display text-base font-bold text-ink tracking-tight mb-3">Flight</h2>
            <FlightControls armed={telem.armed} mode={telem.mode} />
          </div>
        </div>

      </div>
    </div>
  );
}
