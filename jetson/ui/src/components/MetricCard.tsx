interface Props {
  label: string;
  value: string | number;
  unit?: string;
  accent?: boolean;
  sublabel?: string;
}

export function MetricCard({ label, value, unit, accent, sublabel }: Props) {
  return (
    <div className={`rounded-md border p-3 flex flex-col gap-0.5 ${
      accent
        ? "bg-terracotta/10 border-terracotta/25"
        : "bg-surface/80 border-parchment-darker"
    }`}>
      <span className="text-[10px] font-sans font-medium uppercase tracking-widest text-ink-muted">{label}</span>
      <div className="flex items-baseline gap-1">
        <span className={`font-mono text-xl font-medium leading-none ${accent ? "text-terracotta" : "text-ink"}`}>
          {value}
        </span>
        {unit && <span className="text-xs text-ink-muted font-sans">{unit}</span>}
      </div>
      {sublabel && <span className="text-[10px] text-ink-faint font-sans">{sublabel}</span>}
    </div>
  );
}
