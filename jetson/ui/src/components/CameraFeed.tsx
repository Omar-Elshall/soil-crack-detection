import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { API } from "../api/config";
import { Maximize2, Minimize2, X } from "lucide-react";

interface Props { crackRatioPct?: number }

export function CameraFeed({ crackRatioPct = 0 }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [error, setError] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const onError = () => setError(true);
    const onLoad  = () => setError(false);
    img.addEventListener("error", onError);
    img.addEventListener("load",  onLoad);
    return () => { img.removeEventListener("error", onError); img.removeEventListener("load", onLoad); };
  }, []);

  // Close on Escape
  useEffect(() => {
    if (!expanded) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") setExpanded(false); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [expanded]);

  const ringColor =
    crackRatioPct > 20 ? "rgb(var(--accent))" :
    crackRatioPct > 5  ? "#F59E0B" :
    "rgb(var(--positive))";

  const feedContent = (fullscreen: boolean) => (
    <div
      className={`relative overflow-hidden border border-parchment-darker ${
        fullscreen
          ? "fixed inset-8 z-50 rounded-xl shadow-2xl"
          : "rounded-md aspect-square w-full"
      }`}
      style={{ background: "#000" }}
    >
      {error ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2"
          style={{ color: "rgb(var(--ink-faint))" }}>
          <div className="w-10 h-10 rounded-md border-2 border-current/30 flex items-center justify-center">
            <span className="text-xs font-mono">OFF</span>
          </div>
          <span className="text-xs font-mono">Camera offline</span>
        </div>
      ) : (
        <img
          ref={!fullscreen ? imgRef : undefined}
          src={`${API.inference}/stream`}
          alt="Live inference feed"
          className="w-full h-full object-contain"
        />
      )}

      {/* Crack ratio badge */}
      <div className="absolute top-2 left-2 pointer-events-none">
        <div
          className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-mono font-bold text-white transition-all duration-700"
          style={{ background: "rgba(0,0,0,0.55)", boxShadow: `0 0 0 1.5px ${ringColor}` }}
        >
          <span className="w-1.5 h-1.5 rounded-full transition-colors duration-700" style={{ background: ringColor }} />
          {crackRatioPct.toFixed(1)}%
        </div>
      </div>

      {/* Model label */}
      <div className="absolute top-2 right-10 pointer-events-none">
        <span className="text-[9px] font-mono text-white/35 bg-black/20 px-1.5 py-0.5 rounded tracking-widest uppercase">
          ECN
        </span>
      </div>

      {/* Expand / collapse button */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="absolute bottom-2 right-2 p-1.5 rounded text-white/50 hover:text-white transition-colors"
        style={{ background: "rgba(0,0,0,0.4)" }}
      >
        {fullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
      </button>

      {/* Close button when fullscreen */}
      {fullscreen && (
        <button
          onClick={() => setExpanded(false)}
          className="absolute top-2 right-2 p-1.5 rounded text-white/50 hover:text-white transition-colors"
          style={{ background: "rgba(0,0,0,0.4)" }}
        >
          <X size={14} />
        </button>
      )}
    </div>
  );

  return (
    <>
      {/* Normal inline feed */}
      {feedContent(false)}

      {/* Fullscreen portal */}
      {expanded && createPortal(
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40 bg-black/85 backdrop-blur-sm"
            onClick={() => setExpanded(false)}
          />
          {feedContent(true)}
        </>,
        document.body
      )}
    </>
  );
}
