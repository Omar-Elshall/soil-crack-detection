"""
routes.py — FastAPI routes for inference service.
  GET  /stream          — MJPEG video stream
  GET  /status          — JSON status snapshot
  WS   /ws/detections   — real-time crack ratio broadcast
"""

import asyncio
import time
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .streamer import mjpeg_generator, state

router = APIRouter()

# Active WebSocket clients
_ws_clients: Set[WebSocket] = set()


@router.get("/stream")
async def stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/status")
async def status():
    _, crack_ratio, fps = state.snapshot()
    return {
        "running": state.running,
        "crack_ratio_pct": round(crack_ratio * 100, 2),
        "fps": round(fps, 1),
        "model": "EfficientCrackNet",
    }


@router.websocket("/ws/detections")
async def ws_detections(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(0.2)   # 5 Hz push
            _, crack_ratio, fps = state.snapshot()
            payload = {
                "crack_ratio_pct": round(crack_ratio * 100, 2),
                "fps": round(fps, 1),
                "timestamp_ms": int(time.time() * 1000),
            }
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        _ws_clients.discard(websocket)
    except Exception:
        _ws_clients.discard(websocket)
