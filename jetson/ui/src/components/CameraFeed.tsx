import { useEffect, useRef, useState } from "react";
import { API } from "../api/config";
import { Maximize2, Minimize2 } from "lucide-react";

export function CameraFeed() {
  const [expanded, setExpanded] = useState(false);
  const [error, setError] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const streamUrl = `${API.inference}/stream`;

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const onError = () => setError(true);
    const onLoad  = () => setError(false);
    img.addEventListener("error", onError);
    img.addEventListener("load",  onLoad);
    return () => {
      img.removeEventListener("error", onError);
      img.removeEventListener("load",  onLoad);
    };
  }, []);

  return (
    <div className={`relative bg-ink rounded-md overflow-hidden border border-parchment-darker ${
      expanded ? "fixed inset-4 z-50 rounded-lg" : "aspect-square w-full"
    }`}>
      {error ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-ink-faint">
          <div className="w-8 h-8 rounded border-2 border-ink-faint/30 flex items-center justify-center">
            <span className="text-xs font-mono">NO</span>
          </div>
          <span className="text-xs font-mono">Camera offline</span>
        </div>
      ) : (
        <img
          ref={imgRef}
          src={streamUrl}
          alt="Live inference feed"
          className="w-full h-full object-contain"
        />
      )}

      {/* HUD overlay — top */}
      <div className="absolute top-2 left-2 right-2 flex justify-between items-start pointer-events-none">
        <span className="text-[9px] font-mono text-white/60 bg-black/30 px-1.5 py-0.5 rounded">
          LIVE · EfficientCrackNet
        </span>
      </div>

      {/* Expand toggle */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="absolute bottom-2 right-2 p-1.5 rounded bg-black/30 text-white/70 hover:text-white hover:bg-black/50 transition-colors"
      >
        {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
      </button>

      {expanded && (
        <button
          onClick={() => setExpanded(false)}
          className="absolute inset-0 z-[-1]"
          aria-label="Close"
        />
      )}
    </div>
  );
}
