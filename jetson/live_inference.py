"""
Live crack detection inference from Arducam IMX477 via GStreamer on Jetson.

The Arducam IMX477 does not appear as /dev/video* on this Jetson setup — it is
accessed exclusively via the nvarguscamerasrc GStreamer element.

Usage:
    python3 jetson/live_inference.py
    python3 jetson/live_inference.py --model_path results/saved_models/EfficientCrackNet/best_model_num_real_4.pt
    python3 jetson/live_inference.py --sensor_mode 0   # full 4032x3040 @ 21fps (sharpest)
    python3 jetson/live_inference.py --sensor_mode 2   # 1920x1080 @ 60fps (faster)

Controls:
    q — quit
    s — save current frame + mask to results/live_captures/

Sensor modes (IMX477):
    0 — 4032x3040 @ 21fps  (full res, best quality, slowest)
    1 — 3840x2160 @ 30fps
    2 — 1920x1080 @ 60fps  (default — good balance of fps and quality)
"""

import argparse
import os
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
    0: (4032, 3040),
    1: (3840, 2160),
    2: (1920, 1080),
}

def build_gst_pipeline(sensor_mode: int, wbmode: int = 1) -> str:
    w, h = SENSOR_MODE_DIMS[sensor_mode]
    fps = {0: 21, 1: 30, 2: 60}[sensor_mode]
    return (
        f"nvarguscamerasrc sensor-id=0 sensor-mode={sensor_mode} wbmode={wbmode} "
        f"! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 "
        f"! nvvidconv "
        f"! video/x-raw,format=BGRx "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        f"! appsink drop=1"
    )


# ---------------------------------------------------------------------------
# Model helpers (mirrors predict.py — do NOT apply sigmoid again)
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device) -> EfficientCrackNet:
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


def predict_frame(model: EfficientCrackNet, frame_bgr: np.ndarray,
                  device: torch.device, threshold: float) -> np.ndarray:
    """Return a uint8 binary mask (0 or 255) at 512x512."""
    tensor = TRANSFORM(frame_bgr).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)   # sigmoid applied inside model — do NOT apply again
    mask = (out[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
    return mask


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.45) -> np.ndarray:
    """Blend a red crack overlay onto the frame. Both inputs resized to match."""
    h, w = frame_bgr.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = frame_bgr.copy()
    crack_pixels = mask_resized > 127
    overlay[crack_pixels] = (0, 0, 220)   # red in BGR

    return cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model = load_model(args.model_path, device)
    print("Model loaded.")

    pipeline = build_gst_pipeline(args.sensor_mode, args.wbmode)
    print(f"Opening GStreamer pipeline:\n  {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("ERROR: Could not open camera via GStreamer.")
        print("Make sure nvarguscamerasrc is available and the camera is connected.")
        return

    os.makedirs("results/live_captures", exist_ok=True)
    save_idx = 0

    print("Running. Press 'q' to quit, 's' to save a frame.")

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to grab frame.")
            continue

        # Run inference
        mask = predict_frame(model, frame, device, args.threshold)

        # Overlay
        display = overlay_mask(frame, mask, alpha=args.overlay_alpha)

        # FPS counter
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
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
            print(f"Saved frame + mask + overlay: {ts}")
            save_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Live crack detection on Jetson")
    p.add_argument("--model_path", default="results/saved_models/EfficientCrackNet/best_model_num_real_4.pt")
    p.add_argument("--sensor_mode", type=int, default=2, choices=[0, 1, 2],
                   help="Camera sensor mode: 0=4032x3040@21fps, 1=3840x2160@30fps, 2=1920x1080@60fps")
    p.add_argument("--wbmode", type=int, default=1,
                   help="White balance mode (1=auto, 5=daylight). Auto takes ~15s to settle.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Prediction threshold (0.0–1.0)")
    p.add_argument("--overlay_alpha", type=float, default=0.45,
                   help="Crack overlay opacity (0=invisible, 1=solid)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
