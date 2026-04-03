"""
Live crack detection inference from Arducam IMX477 via GStreamer on Jetson.

The Arducam IMX477 does not appear as /dev/video* on this Jetson setup — it is
accessed exclusively via the nvarguscamerasrc GStreamer element.

Usage:
    # PyTorch inference (slow, ~1 FPS)
    python3 jetson/live_inference.py

    # TensorRT inference (fast, run build_trt.sh first)
    python3 jetson/live_inference.py --engine results/efficientcracknet_fp16.trt

    # 4K display (same inference speed, sharper overlay)
    python3 jetson/live_inference.py --engine results/efficientcracknet_fp16.trt --sensor_mode 0

Controls:
    q — quit
    s — save current frame + mask to results/live_captures/

Sensor modes (IMX477 on this install):
    0 — 3840x2160 @ 30fps  (4K, sharper display, same inference speed)
    1 — 1920x1080 @ 60fps  (default)
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

from crack_detection.models.efficientcracknet import EfficientCrackNet


# ---------------------------------------------------------------------------
# GStreamer pipeline string for OpenCV capture
# ---------------------------------------------------------------------------

SENSOR_MODE_DIMS = {
    0: (3840, 2160),
    1: (1920, 1080),
}

def build_gst_pipeline(sensor_mode: int, wbmode: int = 1) -> str:
    w, h = SENSOR_MODE_DIMS[sensor_mode]
    fps = {0: 30, 1: 60}[sensor_mode]
    return (
        f"nvarguscamerasrc sensor-id=0 sensor-mode={sensor_mode} wbmode={wbmode} "
        f"tnr-mode=0 ee-mode=0 "
        f"! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 "
        f"! nvvidconv interpolation-method=5 "
        f"! video/x-raw,format=BGRx "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        f"! appsink drop=1"
    )


# ---------------------------------------------------------------------------
# PyTorch inference (fallback — no TRT engine)
# ---------------------------------------------------------------------------

def load_pytorch_model(model_path: str, device: torch.device) -> EfficientCrackNet:
    model = EfficientCrackNet().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"]
    )
    model.eval()
    return model


TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


def predict_pytorch(model, frame_bgr: np.ndarray,
                    device: torch.device, threshold: float,
                    fp16: bool = False) -> np.ndarray:
    tensor = TRANSFORM(frame_bgr).unsqueeze(0).to(device)
    if fp16:
        tensor = tensor.half()
    with torch.no_grad():
        out = model(tensor)   # sigmoid applied inside model — do NOT apply again
    return (out[0, 0].float().cpu().numpy() > threshold).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# TensorRT inference
# ---------------------------------------------------------------------------

class TRTInferencer:
    """Wraps a TensorRT engine for single-image inference."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initializes CUDA context

        self._cuda = cuda
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate pinned host buffers and device buffers
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

    def infer(self, frame_bgr: np.ndarray, threshold: float) -> np.ndarray:
        # Preprocess
        img = cv2.resize(frame_bgr, (512, 512)).astype(np.float32) / 255.0
        img = img[:, :, ::-1]  # BGR → RGB
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])  # NCHW

        np.copyto(self.inputs[0]["host"], img.ravel())

        # H2D
        self._cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        # Run
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        # D2H
        self._cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()

        out = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        return (out[0, 0] > threshold).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = frame_bgr.copy()
    overlay[mask_resized > 127] = (0, 0, 220)   # red in BGR
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)


# ---------------------------------------------------------------------------
# Threaded frame grabber — keeps latest frame ready so inference never waits
# ---------------------------------------------------------------------------

class FrameGrabber:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._grab, daemon=True)
        self.thread.start()

    def _grab(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.thread.join()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Choose inference backend
    if args.engine:
        print(f"Loading TensorRT engine: {args.engine}")
        inferencer = TRTInferencer(args.engine)
        predict = lambda frame: inferencer.infer(frame, args.threshold)
        print("TensorRT engine ready.")
    else:
        print(f"Loading PyTorch model: {args.model_path}")
        model = load_pytorch_model(args.model_path, device)
        if args.fp16:
            model = model.half()
            print("Running in FP16 mode.")
        if args.compile:
            print("Compiling model with torch.compile() — first inference will be slow...")
            model = torch.compile(model)
        predict = lambda frame: predict_pytorch(model, frame, device, args.threshold, args.fp16)
        print("Model ready.")

    pipeline = build_gst_pipeline(args.sensor_mode, args.wbmode)
    print(f"Opening GStreamer pipeline:\n  {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("ERROR: Could not open camera via GStreamer.")
        return

    grabber = FrameGrabber(cap)
    os.makedirs("results/live_captures", exist_ok=True)
    print("Running. Press 'q' to quit, 's' to save a frame.")

    # Wait for first frame
    while grabber.get() is None:
        time.sleep(0.05)

    prev_time = time.time()
    while True:
        frame = grabber.get()
        if frame is None:
            continue

        mask = predict(frame)
        display = overlay_mask(frame, mask, alpha=args.overlay_alpha)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        backend = "TRT" if args.engine else ("FP16" if args.fp16 else "FP32")
        cv2.putText(display, f"{backend} | FPS: {fps:.1f}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Crack Detection — Live", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"results/live_captures/{ts}_frame.png", frame)
            cv2.imwrite(f"results/live_captures/{ts}_mask.png", mask)
            cv2.imwrite(f"results/live_captures/{ts}_overlay.png", display)
            print(f"Saved: {ts}")

    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Live crack detection on Jetson")
    p.add_argument("--engine",      default=None,
                   help="Path to TensorRT .trt engine (omit to use PyTorch)")
    p.add_argument("--model_path",  default="results/saved_models/EfficientCrackNet/best_model_num_real_4.pt")
    p.add_argument("--sensor_mode", type=int, default=1, choices=[0, 1],
                   help="0=3840x2160@30fps  1=1920x1080@60fps")
    p.add_argument("--wbmode",      type=int, default=1,
                   help="White balance: 1=auto, 5=daylight")
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--overlay_alpha", type=float, default=0.45)
    p.add_argument("--fp16",        action="store_true",
                   help="Run PyTorch model in FP16 (faster on Orin Nano tensor cores)")
    p.add_argument("--compile",     action="store_true",
                   help="Apply torch.compile() for extra speedup (slow first frame)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
