import { useState } from "react";
import * as flight from "../api/flight";
import { ConfirmModal } from "./ConfirmModal";
import { AlertTriangle, Shield, Navigation, ArrowDown, RotateCcw, Crosshair } from "lucide-react";

const ARM_CHECKLIST = [
  "All personnel are at least 10 m from the drone",
  "Propellers are securely attached and undamaged",
  "Battery is fully charged and properly secured",
  "Hardware safety switch has been pressed (LED solid)",
  "Area is clear of obstacles and overhead obstructions",
  "You have direct line-of-sight to the drone",
];

interface CmdButtonProps {
  label: string;
  icon?: React.ReactNode;
  onClick: () => Promise<unknown>;
  variant?: "default" | "danger" | "primary" | "emergency";
  disabled?: boolean;
  fullWidth?: boolean;
}

function CmdButton({ label, icon, onClick, variant = "default", disabled, fullWidth }: CmdButtonProps) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; msg: string } | null>(null);

  async function handle() {
    setBusy(true); setResult(null);
    try {
      const r = await onClick() as { ok: boolean; message: string };
      setResult({ ok: r?.ok ?? true, msg: r?.message ?? "OK" });
    } catch { setResult({ ok: false, msg: "Error" }); }
    finally {
      setBusy(false);
      setTimeout(() => setResult(null), 3000);
    }
  }

  const base = `flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-mono font-medium transition-all border disabled:opacity-40 disabled:cursor-not-allowed ${fullWidth ? "w-full" : ""}`;

  const styles: Record<string, string> = {
    default:   "bg-surface border-parchment-darker text-ink-soft hover:bg-parchment-dark hover:text-ink hover:border-parchment-darker",
    danger:    "bg-amber-500/10 border-amber-500/30 text-amber-500 hover:bg-amber-500/20",
    primary:   "bg-terracotta/90 border-terracotta text-white hover:bg-terracotta",
    emergency: "bg-red-500/15 border-red-500/40 text-red-400 hover:bg-red-500/25",
  };

  return (
    <button className={`${base} ${styles[variant]}`} onClick={handle} disabled={disabled || busy}>
      {busy ? <span className="animate-pulse">…</span>
        : result ? (
          <span className={result.ok ? "text-moss" : "text-red-400"}>
            {result.msg.slice(0, 18)}
          </span>
        ) : (
          <>
            {icon && <span className="opacity-75">{icon}</span>}
            {label}
          </>
        )
      }
    </button>
  );
}

interface Props { armed: boolean; mode: string }

export function FlightControls({ armed, mode }: Props) {
  const [showArmConfirm, setShowArmConfirm] = useState(false);

  const isGuided = mode === "GUIDED";

  return (
    <div className="flex flex-col gap-3">

      {/* Warning notice */}
      <div className="flex items-center gap-1.5 px-2.5 py-2 rounded-md border border-amber-500/20 bg-amber-500/8 text-[10px] font-mono text-amber-500/80">
        <AlertTriangle size={11} className="shrink-0" />
        Ensure area is clear before any command
      </div>

      {/* Mode setup */}
      <div>
        <p className="text-[9px] font-mono uppercase tracking-widest text-ink-faint mb-1.5">Mode</p>
        <CmdButton
          label="Set GUIDED"
          icon={<Crosshair size={12} />}
          onClick={flight.guided}
          variant="primary"
          fullWidth
        />
      </div>

      {/* Arm / Disarm — most prominent, requires confirmation */}
      <div>
        <p className="text-[9px] font-mono uppercase tracking-widest text-ink-faint mb-1.5">Arm State</p>
        <div className="grid grid-cols-2 gap-1.5">
          <button
            onClick={() => setShowArmConfirm(true)}
            className="flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-md text-xs font-mono font-bold border transition-all bg-amber-500/15 border-amber-500/40 text-amber-500 hover:bg-amber-500/25"
          >
            <Shield size={13} />
            ARM
          </button>
          <CmdButton
            label="DISARM"
            onClick={flight.disarm}
            variant="default"
          />
        </div>
      </div>

      {/* Flight actions */}
      <div>
        <p className="text-[9px] font-mono uppercase tracking-widest text-ink-faint mb-1.5">Flight</p>
        <div className="grid grid-cols-3 gap-1.5">
          <CmdButton
            label="TAKEOFF"
            icon={<Navigation size={11} />}
            onClick={() => flight.takeoff(2.0)}
            variant="primary"
            disabled={!armed || !isGuided}
          />
          <CmdButton
            label="LAND"
            icon={<ArrowDown size={11} />}
            onClick={flight.land}
            variant="default"
            disabled={!armed}
          />
          <CmdButton
            label="RTL"
            icon={<RotateCcw size={11} />}
            onClick={flight.rtl}
            variant="emergency"
            disabled={!armed}
          />
        </div>
      </div>

      {/* ARM confirmation modal */}
      {showArmConfirm && (
        <ConfirmModal
          title="Arm Motors"
          description="Arming will spin up the motors. The drone is ready to fly immediately after arming."
          checklist={ARM_CHECKLIST}
          confirmLabel="ARM MOTORS"
          confirmClass="bg-amber-500 hover:bg-amber-600 border border-amber-600"
          warning="Motors will become live. Keep all personnel clear."
          onConfirm={async () => {
            setShowArmConfirm(false);
            await flight.arm();
          }}
          onCancel={() => setShowArmConfirm(false)}
        />
      )}
    </div>
  );
}
