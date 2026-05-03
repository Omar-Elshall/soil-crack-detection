"""
camera.py — GStreamer frame grabber for Arducam IMX477.
Identical pipeline to live_inference.py (sensor_mode=0, TNR, EE, center crop → 512x512).
"""

import threading
import time

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

SENSOR_MODE_DIMS = {
    0: (4032, 3040),  # 4:3   21 FPS  (full sensor; long exposure → motion blur)
    1: (3840, 2160),  # 16:9  30 FPS
    2: (1920, 1080),  # 16:9  60 FPS  (low motion blur — preferred for live demo)
}
SENSOR_MODE_FPS = {0: 21, 1: 30, 2: 60}


def build_pipeline(
    sensor_mode: int = 2,
    wbmode: int = 1,
    max_preview: int = 1440,
    tnr_strength: float = 0.0,
    ee_strength: float = 0.0,
    exposure_max_ms: float = 0.0,
) -> str:
    """
    wbmode: 0=off 1=auto 2=incandescent 3=fluorescent 4=warm-fluorescent
            5=daylight 6=cloudy-daylight 7=twilight 8=shade 9=manual
    Default is auto (1).

    sensor_mode default is 2 (1920×1080 @ 60 FPS). Higher FPS lets the auto-
    exposure algorithm pick a shorter shutter time, dramatically reducing
    motion blur for hand-held demo work. We center-crop to a square (so the
    network sees a 1:1 region matching training) regardless of mode.
    """
    w, h = SENSOR_MODE_DIMS[sensor_mode]
    fps = SENSOR_MODE_FPS[sensor_mode]
    crop = min(w, h)                # square = the smaller dim
    left = (w - crop) // 2
    top  = (h - crop) // 2
    right  = left + crop
    bottom = top + crop
    # Cap streamed/preview dimension to keep WiFi payload sane while still
    # benefiting from the higher-res sensor mode for the model input.
    out = min(crop, max_preview)
    # Output the cropped square at native sensor resolution. Python downsizes
    # to the model's 512×512 input itself (after sharpening) and uses the
    # full-res frame for the operator-facing overlay/stream.
    # TNR (temporal noise reduction): 0=off, otherwise mode 1 with given strength.
    # Strength range -1.0 to 1.0; minimal denoise around 0.2-0.4.
    if tnr_strength > 0:
        tnr_prop = f"tnr-mode=1 tnr-strength={tnr_strength} "
    else:
        tnr_prop = "tnr-mode=0 "

    # EE (edge enhancement): 0=off, otherwise mode 1 with given strength.
    if ee_strength > 0:
        ee_prop = f"ee-mode=1 ee-strength={ee_strength} "
    else:
        ee_prop = "ee-mode=0 "

    # Optional shutter cap: clamp auto-exposure MAX for less motion blur.
    # Floor is the sensor minimum (13 µs) so AE can fully dim under bright
    # light — previously the floor was 1 ms which made bright scenes clip.
    if exposure_max_ms > 0:
        exp_max_ns = int(exposure_max_ms * 1_000_000)
        exp_min_ns = 13_000  # IMX477 hardware minimum
        exposure_prop = f'exposuretimerange="{exp_min_ns} {exp_max_ns}" '
    else:
        exposure_prop = ""

    return (
        f"nvarguscamerasrc sensor-id=0 sensor-mode={sensor_mode} wbmode={wbmode} "
        f"{tnr_prop}{ee_prop}{exposure_prop}"
        f"! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 "
        f"! nvvidconv left={left} right={right} top={top} bottom={bottom} interpolation-method=5 "
        f"! video/x-raw(memory:NVMM),width={out},height={out} "
        f"! nvvidconv "
        f"! video/x-raw,format=BGRx "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        f"! appsink max-buffers=1 drop=true sync=false"
    )


class FrameGrabber:
    """Background thread that continuously reads frames from the GStreamer pipeline."""

    def __init__(
        self,
        sensor_mode: int = 2,
        wbmode: int = 1,
        max_preview: int = 1440,
        tnr_strength: float = 0.0,
        ee_strength: float = 0.0,
        exposure_max_ms: float = 0.0,
    ):
        self.sensor_mode = sensor_mode
        self.wbmode = wbmode
        self.max_preview = max_preview
        self.tnr_strength = tnr_strength
        self.ee_strength = ee_strength
        self.exposure_max_ms = exposure_max_ms
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._cap = None
        self._thread = None

    def start(self):
        if not CV2_AVAILABLE:
            raise RuntimeError("cv2 not available — camera only works on Jetson")
        pipeline = build_pipeline(
            self.sensor_mode, self.wbmode, self.max_preview,
            self.tnr_strength, self.ee_strength, self.exposure_max_ms,
        )
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open GStreamer camera pipeline")
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)

    def get_frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
