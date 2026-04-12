import { useEffect, useRef, useState } from "react";
import { API } from "../api/config";
import { Maximize2, Minimize2, X } from "lucide-react";

export function CameraFeed() {
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
          className="fixed inset-0 flex items-center justify-center bg-black/80 p-8"
          style={{ zIndex: 9999 }}
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
