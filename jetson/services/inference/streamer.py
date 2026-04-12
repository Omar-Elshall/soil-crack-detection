"""
streamer.py — Shared inference state + MJPEG frame generator.
The inference loop writes here; HTTP /stream reads from here.
"""

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class InferenceState:
    overlay_frame: np.ndarray = None   # BGR frame with mask overlay
    raw_frame: np.ndarray = None       # BGR frame without overlay
    mask: np.ndarray = None            # uint8 binary mask
    crack_ratio: float = 0.0
    fps: float = 0.0
    running: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, overlay_frame, raw_frame, mask, crack_ratio, fps):
        with self.lock:
            self.overlay_frame = overlay_frame
            self.raw_frame = raw_frame
            self.mask = mask
            self.crack_ratio = crack_ratio
            self.fps = fps

    def snapshot(self):
        with self.lock:
            return (
                self.overlay_frame.copy() if self.overlay_frame is not None else None,
                self.crack_ratio,
                self.fps,
            )


# Global singleton shared between inference loop and routes
state = InferenceState()


def build_overlay(frame_bgr: np.ndarray, mask: np.ndarray, crack_ratio: float, fps: float) -> np.ndarray:
    """Overlay crack mask on frame with ratio + fps HUD."""
    overlay = frame_bgr.copy()
    colored = np.zeros_like(frame_bgr)
    colored[mask > 127] = (46, 98, 196)   # terracotta in BGR: #C4622D → BGR (46, 98, 196)
    overlay = cv2.addWeighted(overlay, 0.55, colored, 0.45, 0)

    # HUD text
    pct = crack_ratio * 100
    color = (46, 98, 196) if crack_ratio > 0.05 else (90, 122, 61)  # terracotta or green
    cv2.putText(overlay, f"{pct:.1f}% cracks", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(overlay, f"{fps:.1f} fps", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    return overlay


def mjpeg_generator(jpeg_quality: int = 80):
    """Yields MJPEG multipart frames. One generator per client connection."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        frame, _, _ = state.snapshot()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            continue
        yield boundary + buf.tobytes() + b"\r\n"
        time.sleep(0.033)   # ~30 fps cap for stream clients
