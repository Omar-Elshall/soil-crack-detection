import { useState } from "react";
import { AlertTriangle, X, CheckSquare, Square } from "lucide-react";

interface Props {
  title: string;
  description: string;
  checklist?: string[];
  confirmLabel: string;
  confirmClass?: string;
  warning?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmModal({
  title, description, checklist = [], confirmLabel,
  confirmClass, warning, onConfirm, onCancel,
}: Props) {
  const [checked, setChecked] = useState<boolean[]>(checklist.map(() => false));

  const allChecked = checklist.length === 0 || checked.every(Boolean);

  function toggle(i: number) {
    setChecked(prev => prev.map((v, idx) => idx === i ? !v : v));
  }

  return (
    <div className="fixed inset-0 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm" style={{ zIndex: 9999 }}>
      <div
        className="relative w-full max-w-md rounded-xl border bg-surface shadow-2xl overflow-hidden"
        style={{ borderColor: "rgb(var(--parchment-darker))" }}
      >
        {/* Header */}
        <div className="px-5 py-4 border-b flex items-start gap-3" style={{ borderColor: "rgb(var(--parchment-darker))" }}>
          <div className="mt-0.5 w-8 h-8 rounded-lg bg-amber-500/15 border border-amber-500/30 flex items-center justify-center shrink-0">
            <AlertTriangle size={16} className="text-amber-400" />
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="font-display font-bold text-base text-ink">{title}</h2>
            <p className="text-xs text-ink-muted mt-1 font-sans leading-relaxed">{description}</p>
          </div>
          <button onClick={onCancel} className="text-ink-faint hover:text-ink-muted transition-colors ml-2">
            <X size={16} />
          </button>
        </div>

        {/* Warning banner */}
        {warning && (
          <div className="mx-5 mt-4 px-3 py-2 rounded border border-red-400/25 bg-red-400/8 text-xs font-mono text-red-400">
            {warning}
          </div>
        )}

        {/* Checklist */}
        {checklist.length > 0 && (
          <div className="px-5 pt-4 pb-2">
            <p className="text-[10px] font-mono uppercase tracking-widest text-ink-faint mb-3">
              Confirm all before proceeding
            </p>
            <div className="space-y-2.5">
              {checklist.map((item, i) => (
                <button
                  key={i}
                  onClick={() => toggle(i)}
                  className="w-full flex items-start gap-2.5 text-left group"
                >
                  <span className={`mt-0.5 shrink-0 transition-colors ${checked[i] ? "text-moss" : "text-ink-faint"}`}>
                    {checked[i] ? <CheckSquare size={14} /> : <Square size={14} />}
                  </span>
                  <span className={`text-xs font-sans leading-snug transition-colors ${
                    checked[i] ? "text-ink-muted line-through" : "text-ink-soft group-hover:text-ink"
                  }`}>
                    {item}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="px-5 py-4 flex gap-2 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 rounded text-xs font-mono text-ink-muted border bg-surface/80 hover:text-ink transition-colors"
            style={{ borderColor: "rgb(var(--parchment-darker))" }}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={!allChecked}
            className={`px-4 py-2 rounded text-xs font-mono font-bold text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
              confirmClass ?? "bg-amber-500 hover:bg-amber-600 border border-amber-500"
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
