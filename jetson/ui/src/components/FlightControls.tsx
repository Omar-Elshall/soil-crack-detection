import { useState } from "react";
import * as flight from "../api/flight";
import { AlertTriangle } from "lucide-react";

interface ButtonProps {
  label: string;
  onClick: () => Promise<unknown>;
  variant?: "default" | "danger" | "primary" | "dark";
  disabled?: boolean;
}

function CmdButton({ label, onClick, variant = "default", disabled }: ButtonProps) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  async function handle() {
    setBusy(true); setResult(null);
    try {
      const r = await onClick() as { ok: boolean; message: string };
      setResult(r?.message ?? "OK");
    } catch { setResult("Error"); }
    finally {
      setBusy(false);
      setTimeout(() => setResult(null), 2500);
    }
  }

  const base = "px-3 py-1.5 rounded text-xs font-mono font-medium transition-all duration-150 border disabled:opacity-40 disabled:cursor-not-allowed";
  const variants: Record<string, string> = {
    default:  "bg-surface border-parchment-darker text-ink-soft hover:bg-parchment-dark",
    danger:   "bg-red-500/10 border-red-500/25 text-red-400 hover:bg-red-500/20",
    primary:  "bg-terracotta border-terracotta text-white hover:bg-terracotta-dark",
    dark:     "bg-chrome border-chrome-border text-white/80 hover:bg-chrome-soft",
  };

  return (
    <button className={`${base} ${variants[variant]}`} onClick={handle} disabled={disabled || busy}>
      {busy ? <span className="animate-pulse">…</span> : result ?? label}
    </button>
  );
}

interface Props { armed: boolean; mode: string }

export function FlightControls({ armed, mode }: Props) {
  const isGuided = mode === "GUIDED";
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-1.5 text-xs text-ink-muted font-mono">
        <AlertTriangle size={11} className="text-terracotta" />
        Ensure area is clear
      </div>
      <div className="grid grid-cols-3 gap-1.5">
        <CmdButton label="GUIDED"  onClick={flight.guided} />
        <CmdButton label="ARM"     onClick={flight.arm}   variant="danger" />
        <CmdButton label="DISARM"  onClick={flight.disarm} />
        <CmdButton label="TAKEOFF" onClick={() => flight.takeoff(0.3)} variant="primary" disabled={!armed || !isGuided} />
        <CmdButton label="LAND"    onClick={flight.land}  disabled={!armed} />
        <CmdButton label="RTL"     onClick={flight.rtl}   variant="dark" disabled={!armed} />
      </div>
    </div>
  );
}
