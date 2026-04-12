import { useEffect, useRef, useState } from "react";
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

  const badges = (
    <>
      <div className="absolute top-2 left-2 pointer-events-none">
        <div
          className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-mono font-bold text-white transition-all duration-700"
          style={{ background: "rgba(0,0,0,0.55)", boxShadow: `0 0 0 1.5px ${ringColor}` }}
        >
          <span className="w-1.5 h-1.5 rounded-full transition-colors duration-700" style={{ background: ringColor }} />
          {crackRatioPct.toFixed(1)}%
        </div>
      </div>
      <div className="absolute top-2 right-10 pointer-events-none">
        <span className="text-[9px] font-mono text-white/35 bg-black/20 px-1.5 py-0.5 rounded tracking-widest uppercase">
          ECN
        </span>
      </div>
    </>
  );

  const feedImg = (
    error ? (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2"
        style={{ color: "rgb(var(--ink-faint))" }}>
        <div className="w-10 h-10 rounded-md border-2 border-current/30 flex items-center justify-center">
          <span className="text-xs font-mono">OFF</span>
        </div>
        <span className="text-xs font-mono">Camera offline</span>
      </div>
    ) : (
      <img
        ref={imgRef}
        src={`${API.inference}/stream`}
        alt="Live inference feed"
        className="w-full h-full object-contain"
      />
    )
  );

  return (
    <>
      {/* Inline feed */}
      <div
        className="relative rounded-md overflow-hidden border border-parchment-darker h-44 w-full shrink-0"
        style={{ background: "#000" }}
      >
        {feedImg}
        {badges}
        <button
          onClick={() => setExpanded(true)}
          className="absolute bottom-2 right-2 p-1.5 rounded text-white/50 hover:text-white transition-colors"
          style={{ background: "rgba(0,0,0,0.4)" }}
        >
          <Maximize2 size={14} />
        </button>
      </div>

      {/* Fullscreen overlay — fixed, no portal needed, always on top */}
      {expanded && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-8"
          onClick={() => setExpanded(false)}
        >
          <div
            className="relative rounded-xl overflow-hidden border border-white/10"
            style={{
              background: "#000",
              width: "min(85vw, calc(85vh * 4 / 3))",
              aspectRatio: "4/3",
            }}
            onClick={e => e.stopPropagation()}
          >
            {feedImg}
            {badges}
            <button
              onClick={() => setExpanded(false)}
              className="absolute top-3 right-3 p-2 rounded-lg text-white/60 hover:text-white transition-colors"
              style={{ background: "rgba(0,0,0,0.5)" }}
            >
              <X size={16} />
            </button>
            <button
              onClick={() => setExpanded(false)}
              className="absolute bottom-3 right-3 p-1.5 rounded text-white/50 hover:text-white transition-colors"
              style={{ background: "rgba(0,0,0,0.4)" }}
            >
              <Minimize2 size={14} />
            </button>
          </div>
        </div>
      )}
    </>
  );
}
