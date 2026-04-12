import { useEffect, useRef, useState } from "react";
import { API } from "../api/config";
import { Maximize2, Minimize2 } from "lucide-react";

interface Props {
  crackRatioPct?: number;
}

export function CameraFeed({ crackRatioPct = 0 }: Props) {
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

  // Badge ring color: green → amber → terracotta based on coverage
  const badgeColor =
    crackRatioPct > 20 ? "#C4622D" :
    crackRatioPct > 5  ? "#D4932A" :
    "#3D7A5A";

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

      {/* Animated crack ratio badge — top-left with color ring */}
      <div className="absolute top-2 left-2 pointer-events-none">
        <div
          className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-mono font-medium bg-black/50 text-white transition-all duration-500"
          style={{
            boxShadow: `0 0 0 1.5px ${badgeColor}`,
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full transition-colors duration-500"
            style={{ backgroundColor: badgeColor }}
          />
          {crackRatioPct.toFixed(1)}%
        </div>
      </div>

      {/* Top-right label */}
      <div className="absolute top-2 right-8 pointer-events-none">
        <span className="text-[9px] font-mono text-white/40 bg-black/20 px-1.5 py-0.5 rounded">
          EfficientCrackNet
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
        <div
          className="absolute inset-0 z-[-1]"
          onClick={() => setExpanded(false)}
        />
      )}
    </div>
  );
}
