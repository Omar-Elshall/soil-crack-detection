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
    """Background thread: grab full-res cropped frame → sharpen → downsize for
    model → upscale mask → overlay on full-res → update shared state.

    Sharpening (mild unsharp mask, amount=0.30) is applied to the full-res
    crop before downsizing to 512×512 — this preserves thin crack edges
    through the resize. The on-screen overlay uses the unsharpened original
    so colours stay natural.
    """
    import time
    import cv2
    import numpy as np
    SHARP_AMOUNT = float(os.environ.get("SHARP_AMOUNT", "0.30"))
    state.running = True
    while state.running:
        frame_full = _grabber.get_frame()  # crop×crop BGR (e.g. 1080×1080)
        if frame_full is None:
            time.sleep(0.01)
            continue
        h_full = frame_full.shape[0]
        # Sharpen on full-res for the model input — preserves edge detail
        # through the downsize.
        if SHARP_AMOUNT > 0:
            blur = cv2.GaussianBlur(frame_full, (0, 0), 1.0)
            sharpened = cv2.addWeighted(frame_full, 1.0 + SHARP_AMOUNT, blur, -SHARP_AMOUNT, 0)
        else:
            sharpened = frame_full
        # Downsize to model input
        frame_small = cv2.resize(sharpened, (512, 512), interpolation=cv2.INTER_AREA)
        mask_small, crack_ratio = _engine.run(frame_small)
        # Upscale mask to full-res for overlay (nearest = sharp pixel edges)
        mask_full = cv2.resize(mask_small, (h_full, h_full), interpolation=cv2.INTER_NEAREST)
        overlay = build_overlay(frame_full, mask_full, crack_ratio, _engine.fps)
        state.update(overlay, frame_full, mask_full, crack_ratio, _engine.fps)


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

        sensor_mode = int(os.environ.get("SENSOR_MODE", "1"))      # 4K@30 default for crisper detail
        max_preview = int(os.environ.get("MAX_PREVIEW", "1440"))
        wbmode = int(os.environ.get("WBMODE", "1"))  # 1=auto, 5=daylight (warmer)
        tnr_strength = float(os.environ.get("TNR_STRENGTH", "0"))  # 0 = off, 0.2-0.4 = minimal
        ee_strength = float(os.environ.get("EE_STRENGTH", "0"))    # 0 = off, ~0.3 = moderate edge enhance
        exposure_max_ms = float(os.environ.get("EXPOSURE_MAX_MS", "8"))  # cap shutter at 8 ms to freeze hand motion
        _grabber = FrameGrabber(
            sensor_mode=sensor_mode,
            wbmode=wbmode,
            max_preview=max_preview,
            tnr_strength=tnr_strength,
            ee_strength=ee_strength,
            exposure_max_ms=exposure_max_ms,
        )
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
