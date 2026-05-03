"""
model.py — EfficientCrackNet inference engine.
Loads TensorRT FP16 engine if present; falls back to PyTorch FP16.
Sigmoid is applied internally by the model — do NOT apply it again.
"""

import os
import time

import numpy as np
import torch
from torchvision import transforms

from crack_detection.models.efficientcracknet import EfficientCrackNet

import re

DEFAULT_TRT_PATH = "results/efficientcracknet_fp16.trt"
DEFAULT_PT_PATH  = "results/saved_models/EfficientCrackNet/best_model_num_real_4.pt"


def _derive_trt_path(model_path: str) -> str:
    """Map a checkpoint path to its matching TRT engine path.
    `best_model_num_real_4.pt` -> `results/efficientcracknet_real4_fp16.trt`
    Falls back to DEFAULT_TRT_PATH if the name doesn't match the convention.
    """
    base = os.path.basename(model_path or "")
    m = re.search(r"real_?(\d+)", base)
    if m:
        return f"results/efficientcracknet_real{m.group(1)}_fp16.trt"
    return DEFAULT_TRT_PATH


# real_6 has higher held-out F1 (0.83) but is much more conservative on the
# live camera feed — it under-predicts on real-world frames at demo distance
# and the on-screen detection is sparse. real_4 is slightly less precise on
# the test set but visually matches what a human marks as crack on the live
# feed, so it's the better deployment checkpoint.
CRACK_THRESHOLD  = float(os.environ.get("CRACK_THRESHOLD", "0.5"))
# Lower this (e.g. 0.30) to make a conservative checkpoint detect more pixels.
# real_6's sigmoid output is well-calibrated on the test distribution but
# undershoots 0.5 on the live camera feed — set CRACK_THRESHOLD=0.30 when
# running real_6 to recover the visual density of detections that real_4
# produces by default.

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class InferenceEngine:
    def __init__(self, model_path: str = None, trt_path: str = None, fp16: bool = True):
        self.trt_path   = trt_path   or os.environ.get("TRT_ENGINE_PATH", DEFAULT_TRT_PATH)
        self.model_path = model_path or os.environ.get("MODEL_PATH",      DEFAULT_PT_PATH)
        self.fp16       = fp16
        self.device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model      = None
        self._trt       = None
        self._last_fps_time = time.time()
        self._frame_count   = 0
        self.fps            = 0.0

    def load(self):
        if os.path.exists(self.trt_path):
            try:
                self._trt = _TRTEngine(self.trt_path)
                print(f"TensorRT engine loaded: {self.trt_path}")
                return
            except Exception as e:
                print(f"TRT load failed ({e}), falling back to PyTorch")
                self._trt = None

        model = EfficientCrackNet().to(self.device)
        ckpt  = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        if self.fp16:
            model = model.half()
        self.model = model
        print(f"PyTorch model loaded: {self.model_path} | device={self.device} | fp16={self.fp16}")

    def run(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Run inference on a 512x512 BGR frame.
        Returns (mask_uint8, crack_ratio) where mask is 0/255 binary.
        """
        if self._trt is not None:
            mask = self._trt.infer(frame_bgr, CRACK_THRESHOLD)
        else:
            tensor = TRANSFORM(frame_bgr).unsqueeze(0).to(self.device)
            if self.fp16:
                tensor = tensor.half()
            with torch.no_grad():
                out = self.model(tensor)
            mask = (out[0, 0].float().cpu().numpy() > CRACK_THRESHOLD).astype(np.uint8) * 255

        crack_ratio = float((mask > 127).sum()) / mask.size

        self._frame_count += 1
        now     = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self.fps          = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

        return mask, crack_ratio

    @property
    def model_name(self) -> str:
        return os.path.basename(self.trt_path if self._trt else self.model_path)

    @property
    def backend(self) -> str:
        return "tensorrt" if self._trt is not None else "pytorch"


class _TRTEngine:
    """TensorRT FP16 engine wrapper for the inference microservice."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initialises CUDA context

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime      = trt.Runtime(logger)
            self.engine  = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name      = self.engine.get_tensor_name(i)
            dtype     = trt.nptype(self.engine.get_tensor_dtype(name))
            shape     = tuple(self.engine.get_tensor_shape(name))
            host_mem  = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev_mem   = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))
            bucket = self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs
            bucket.append({"host": host_mem, "device": dev_mem, "shape": shape})

    def infer(self, frame_bgr: np.ndarray, threshold: float) -> np.ndarray:
        import cv2
        img = cv2.resize(frame_bgr, (512, 512)).astype(np.float32) / 255.0
        img = img[:, :, ::-1]                                         # BGR → RGB
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])  # NCHW
        np.copyto(self.inputs[0]["host"], img.ravel())
        self._cuda.memcpy_htod_async(self.inputs[0]["device"],  self.inputs[0]["host"],  self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        self._cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        out = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        return (out[0, 0] > threshold).astype(np.uint8) * 255
