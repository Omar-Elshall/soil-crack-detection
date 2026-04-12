import type { MissionMeta } from "../api/missions";

function duration(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

interface Props { meta: MissionMeta }

export function SummaryCards({ meta }: Props) {
  const cards = [
    {
      label: "Total Detections",
      value: meta.total_detections.toLocaleString(),
      unit: "frames",
      accent: false,
    },
    {
      label: "Max Coverage",
      value: meta.max_coverage_pct.toFixed(1),
      unit: "%",
      accent: meta.max_coverage_pct > 10,
    },
    {
      label: "Mean Coverage",
      value: meta.mean_coverage_pct.toFixed(1),
      unit: "%",
      accent: false,
    },
    {
      label: "Flight Duration",
      value: duration(meta.flight_duration_s),
      unit: "",
      accent: false,
    },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {cards.map((c) => (
        <div
          key={c.label}
          className={`rounded-md border p-4 flex flex-col gap-1 ${
            c.accent
              ? "bg-terracotta/8 border-terracotta/25"
              : "bg-white/70 border-parchment-darker"
          }`}
          style={{ boxShadow: "0 1px 4px rgba(26,22,18,0.06)" }}
        >
          <span className="text-[10px] font-sans font-medium uppercase tracking-widest text-ink-muted">
            {c.label}
          </span>
          <div className="flex items-baseline gap-1 mt-0.5">
            <span className={`font-mono text-2xl font-medium leading-none ${c.accent ? "text-terracotta" : "text-ink"}`}>
              {c.value}
            </span>
            {c.unit && <span className="text-xs text-ink-muted font-sans">{c.unit}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
