import { useTelemetry } from "../hooks/useTelemetry";
import { useDetections } from "../hooks/useDetections";
import { useMissions } from "../hooks/useMissions";
import { CameraFeed } from "../components/CameraFeed";
import { TelemetryPanel } from "../components/TelemetryPanel";
import { CrackRatioChart } from "../components/CrackRatioChart";
import { FlightControls } from "../components/FlightControls";
import { MissionControl } from "../components/MissionControl";
import { MissionMap } from "../components/MissionMap";
import { MetricCard } from "../components/MetricCard";

export default function LivePage() {
  const telem = useTelemetry();
  const { latest, history } = useDetections();
  const { activeMissionId, loading, start, stop } = useMissions();

  return (
    <div className="flex-1 flex overflow-hidden">

      {/* Left column — camera + crack metrics */}
      <div className="w-72 shrink-0 flex flex-col gap-3 p-3 overflow-y-auto border-r border-parchment-darker">

        {/* Page title */}
        <div className="pt-1">
          <h1 className="font-display text-xl text-ink italic">Live Survey</h1>
          <p className="text-[10px] font-mono text-ink-muted uppercase tracking-widest mt-0.5">
            EfficientCrackNet · Real-time
          </p>
        </div>

        {/* Camera feed */}
        <CameraFeed />

        {/* Crack stats */}
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Coverage"
            value={latest?.crack_ratio_pct.toFixed(1) ?? "—"}
            unit="%"
            accent={(latest?.crack_ratio_pct ?? 0) > 10}
          />
          <MetricCard
            label="FPS"
            value={latest?.fps.toFixed(1) ?? "—"}
            unit="fps"
          />
        </div>

        {/* Trend chart */}
        <div className="rounded-md border border-parchment-darker bg-white/50 p-2">
          <div className="text-[9px] font-mono text-ink-faint uppercase tracking-widest mb-1">
            Crack coverage — last 300 frames
          </div>
          <CrackRatioChart history={history} />
        </div>

        {/* Mission control */}
        <MissionControl
          activeMissionId={activeMissionId}
          loading={loading}
          onStart={start}
          onStop={stop}
        />
      </div>

      {/* Center — map */}
      <div className="flex-1 relative p-3">
        <MissionMap telem={telem} />
      </div>

      {/* Right column — telemetry + flight */}
      <div className="w-64 shrink-0 flex flex-col gap-3 p-3 overflow-y-auto border-l border-parchment-darker">

        <div className="pt-1">
          <h2 className="font-display text-base text-ink italic">Telemetry</h2>
        </div>

        <TelemetryPanel telem={telem} />

        <div className="border-t border-parchment-darker pt-3">
          <h2 className="font-display text-base text-ink italic mb-3">Flight</h2>
          <FlightControls armed={telem.armed} mode={telem.mode} />
        </div>
      </div>

    </div>
  );
}
