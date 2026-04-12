interface Props {
  value: boolean | string;
  trueLabel?: string;
  falseLabel?: string;
  variant?: "armed" | "gps" | "mode" | "mission";
}

export function StatusBadge({ value, trueLabel, falseLabel, variant = "mode" }: Props) {
  const isActive = typeof value === "boolean" ? value : !!value;
  const label = typeof value === "string" ? value : isActive ? (trueLabel ?? "ON") : (falseLabel ?? "OFF");

  const colors: Record<string, string> = {
    armed:   isActive ? "bg-red-100 text-red-700 border-red-300"   : "bg-ink/5 text-ink-muted border-ink-faint/30",
    gps:     isActive ? "bg-moss/10 text-moss border-moss/30"      : "bg-ink/5 text-ink-muted border-ink-faint/30",
    mode:    "bg-parchment-dark text-ink-soft border-parchment-darker",
    mission: isActive ? "bg-terracotta/10 text-terracotta border-terracotta/30" : "bg-ink/5 text-ink-muted border-ink-faint/30",
  };

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono font-medium border ${colors[variant]}`}>
      {(variant === "armed" || variant === "gps" || variant === "mission") && (
        <span className={`w-1.5 h-1.5 rounded-full ${isActive ? "bg-current" : "bg-ink-faint"}`} />
      )}
      {label}
    </span>
  );
}
