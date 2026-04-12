"""
Inference Service — port 8001
Starts GStreamer camera, loads EfficientCrackNet, runs inference loop.
Exposes MJPEG stream, status API, and WebSocket detections feed.

Usage:
    cd ~/soil-crack-detection
    python3 -m uvicorn jetson.services.inference.main:app --host 0.0.0.0 --port 8001
"""

import asyncio
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .camera import FrameGrabber
from .model import InferenceEngine
from .routes import router
from .streamer import build_overlay, state

app = FastAPI(title="Inference Service", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

_grabber: FrameGrabber = None
_engine: InferenceEngine = None


def _inference_loop():
    """Background thread: grab frame → infer → update shared state."""
    state.running = True
    while state.running:
        frame = _grabber.get_frame()
        if frame is None:
            import time; time.sleep(0.01)
            continue
        mask, crack_ratio = _engine.run(frame)
        overlay = build_overlay(frame, mask, crack_ratio, _engine.fps)
        state.update(overlay, frame, mask, crack_ratio, _engine.fps)


@app.on_event("startup")
async def startup():
    global _grabber, _engine

    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    if dry_run:
        print("DRY_RUN mode — skipping model load and camera, using blank frames")
        import time
        import numpy as np
        def _dummy_loop():
            state.running = True
            while state.running:
                blank = np.ones((512, 512, 3), dtype="uint8") * 40
                state.update(blank, blank, (blank[:, :, 0] * 0), 0.0, 0.0)
                time.sleep(0.2)
        threading.Thread(target=_dummy_loop, daemon=True).start()
    else:
        _engine = InferenceEngine(
            model_path=os.environ.get("MODEL_PATH"),
            fp16=os.environ.get("FP16", "1") == "1",
        )
        _engine.load()

        sensor_mode = int(os.environ.get("SENSOR_MODE", "0"))
        _grabber = FrameGrabber(sensor_mode=sensor_mode)
        _grabber.start()

        import time
        print("Waiting for first frame...")
        while _grabber.get_frame() is None:
            time.sleep(0.05)
        print("Camera ready.")

        threading.Thread(target=_inference_loop, daemon=True).start()

    print("Inference service ready.")


@app.on_event("shutdown")
async def shutdown():
    state.running = False
    if _grabber:
        _grabber.stop()
