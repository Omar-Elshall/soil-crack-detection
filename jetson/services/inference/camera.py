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
    0: (4032, 3040),
    1: (3840, 2160),
    2: (1920, 1080),
}


def build_pipeline(sensor_mode: int = 0, wbmode: int = 1) -> str:
    w, h = SENSOR_MODE_DIMS[sensor_mode]
    fps = {0: 21, 1: 30, 2: 60}[sensor_mode]
    return (
        f"nvarguscamerasrc sensor-id=0 sensor-mode={sensor_mode} wbmode={wbmode} "
        f"tnr-mode=1 tnr-strength=1.0 ee-mode=1 ee-strength=0.1 "
        f"! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 "
        f"! nvvidconv left=496 right=3544 top=0 bottom=3040 "
        f"! video/x-raw(memory:NVMM),width=512,height=512 "
        f"! nvvidconv "
        f"! video/x-raw,format=BGRx "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        f"! appsink max-buffers=1 drop=true sync=false"
    )


class FrameGrabber:
    """Background thread that continuously reads frames from the GStreamer pipeline."""

    def __init__(self, sensor_mode: int = 0, wbmode: int = 1):
        self.sensor_mode = sensor_mode
        self.wbmode = wbmode
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._cap = None
        self._thread = None

    def start(self):
        if not CV2_AVAILABLE:
            raise RuntimeError("cv2 not available — camera only works on Jetson")
        pipeline = build_pipeline(self.sensor_mode, self.wbmode)
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
