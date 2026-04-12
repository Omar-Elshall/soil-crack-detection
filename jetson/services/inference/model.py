"""
model.py — EfficientCrackNet loader and inference engine.
Handles FP16, sigmoid-internally (do NOT apply sigmoid again).
"""

import os
import time

import numpy as np
import torch
from torchvision import transforms

from crack_detection.models.efficientcracknet import EfficientCrackNet

DEFAULT_MODEL_PATH = "jetson/models/best_model_num_real_5.pt"
CRACK_THRESHOLD = 0.5

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class InferenceEngine:
    def __init__(self, model_path: str = None, fp16: bool = True):
        self.model_path = model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
        self.fp16 = fp16
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._last_fps_time = time.time()
        self._frame_count = 0
        self.fps = 0.0

    def load(self):
        model = EfficientCrackNet().to(self.device)
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        if self.fp16:
            model = model.half()
        self.model = model
        print(f"Model loaded: {self.model_path} | device={self.device} | fp16={self.fp16}")

    def run(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Run inference on a 512x512 BGR frame.
        Returns (mask_uint8, crack_ratio) where mask is 0/255 binary image.
        """
        tensor = TRANSFORM(frame_bgr).unsqueeze(0).to(self.device)
        if self.fp16:
            tensor = tensor.half()
        with torch.no_grad():
            out = self.model(tensor)
        mask = (out[0, 0].float().cpu().numpy() > CRACK_THRESHOLD).astype(np.uint8) * 255
        crack_ratio = float((mask > 127).sum()) / mask.size

        # FPS tracking
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

        return mask, crack_ratio

    @property
    def model_name(self) -> str:
        return os.path.basename(self.model_path)
