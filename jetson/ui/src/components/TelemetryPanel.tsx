import type { Telemetry } from "../hooks/useTelemetry";
import { StatusBadge } from "./StatusBadge";
import { MetricCard } from "./MetricCard";

interface Props { telem: Telemetry }

export function TelemetryPanel({ telem }: Props) {
  const gpsOk = telem.gps_fix >= 3;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2 flex-wrap">
        <StatusBadge value={telem.connected} trueLabel="ONLINE" falseLabel="OFFLINE" variant="gps" />
        <StatusBadge value={telem.armed} trueLabel="ARMED" falseLabel="SAFE" variant="armed" />
        <StatusBadge value={telem.mode} variant="mode" />
        <StatusBadge value={gpsOk} trueLabel={`GPS ${telem.satellites}sat`} falseLabel="NO GPS" variant="gps" />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <MetricCard label="Altitude" value={telem.alt_m.toFixed(1)} unit="m" />
        <MetricCard label="Heading"  value={telem.heading_deg.toFixed(0)} unit="°" />
        <MetricCard label="Roll"     value={telem.roll_deg.toFixed(1)} unit="°" />
        <MetricCard label="Pitch"    value={telem.pitch_deg.toFixed(1)} unit="°" />
        <MetricCard label="Battery"  value={telem.battery_pct.toFixed(0)} unit="%" accent={telem.battery_pct < 20} />
        <MetricCard label="Voltage"  value={telem.battery_v.toFixed(2)} unit="V" />
      </div>

      {(telem.lat !== 0 || telem.lon !== 0) && (
        <div className="rounded-md border border-parchment-darker bg-surface/60 px-3 py-2 flex gap-4">
          <div>
            <div className="text-[9px] font-mono text-ink-faint uppercase tracking-widest">Lat</div>
            <div className="text-xs font-mono text-ink">{telem.lat.toFixed(7)}</div>
          </div>
          <div>
            <div className="text-[9px] font-mono text-ink-faint uppercase tracking-widest">Lon</div>
            <div className="text-xs font-mono text-ink">{telem.lon.toFixed(7)}</div>
          </div>
        </div>
      )}
    </div>
  );
}
