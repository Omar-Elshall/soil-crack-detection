# Architecture

## EfficientCrackNet (Primary Model)

**File:** `crack_detection/models/efficientcracknet.py`
**Paper:** https://arxiv.org/abs/2409.18099

### Overview
Lightweight encoder-decoder for binary crack segmentation. Input is resized to 512×512 (RGB). Output is a single-channel binary mask (crack=1, background=0).

### Critical implementation detail — sigmoid
The model applies `torch.sigmoid()` **internally** at the end of `forward()` (line 338):
```python
last_out = torch.sigmoid(self.last_block(decoder_block6_out))
```
**Do NOT apply sigmoid again in evaluation or inference scripts.** Double-applying sigmoid compresses all outputs to [0.5, 0.62], making everything threshold as crack (all-white masks). This was a confirmed bug — see `context/known_issues.md`.

### Key modules

**SeparableConv2d** — Depthwise separable convolution (depthwise + pointwise). Reduces parameters vs standard conv while preserving spatial information.

**SEM (Spatial Enhancement Module)** — Attention mechanism applied to encoder feature maps. Uses global average pooling + FC layers to compute channel-wise attention weights, then multiplies back onto the feature map. Helps the model focus on crack-relevant channels.

**EEM (Edge Enhancement Module)** — Applies a Gaussian-based Laplacian kernel to highlight edges/boundaries in the feature maps. This is especially useful for thin cracks that might otherwise be lost during downsampling.

**SubSpace** — Projects feature maps into a lower-dimensional subspace for efficient computation before attention.

**ULSAM (Ultra-Lightweight Subspace Attention Module)** — Combines SubSpace + attention. Used in the decoder to refine upsampled feature maps.

**MobileViTBlock** (`crack_detection/models/mobile_vit.py`) — Hybrid conv+transformer block. Uses `einops` for tensor rearrangement. Provides global context that pure CNN blocks miss, important for detecting long crack patterns.

### Architecture flow
```
Input (512×512×3)
    → EfficientNet encoder (B0 backbone, 5 stages)
    → SEM applied at each encoder stage
    → EEM applied to edge features
    → Decoder (6 upsampling blocks with skip connections)
    → ULSAM at decoder stages
    → 1×1 conv → sigmoid
Output (512×512×1)
```

### Input size note
The original paper uses 192×256. This codebase resizes to 512×512 (set in `data/dataset.py` lines 88–91) to better handle the square real-world images (720×720 real images, 512×512 simulated). If you change the input size, update both the dataset loader and verify the encoder stages still align.

---

## Baselines

**File:** `crack_detection/models/baselines.py`

### UNet_FCN
Standard U-Net with optional batch normalization. Constructor args:
- `scaler` — controls channel multiplier (use 2 for standard size)
- `batchnorm` — enable/disable batch norm

### LMM_Net
LMM-Net from https://ieeexplore.ieee.org/document/10539282. More complex than U-Net, uses multi-scale feature fusion. Slower than EfficientCrackNet but was the previous SOTA on DeepCrack.

---

## Loss Functions

**File:** `crack_detection/models/losses.py`

Three losses combined during training:
```
total_loss = BCE + alpha * (Dice + IoU)
```

- **BCELoss** — Binary cross-entropy. Pixel-level classification loss.
- **DiceLoss** — 1 - Dice coefficient. Good for imbalanced datasets (few crack pixels vs many background pixels).
- **IoULoss** — 1 - IoU. Directly optimizes the evaluation metric.

`alpha` starts at 0.8 and decreases by 0.2 every 60 epochs, gradually shifting weight from Dice/IoU toward BCE as training progresses.

---

## Metrics

**File:** `crack_detection/metrics.py`

- `f1_score(y_true, y_pred)` — Custom F1 implementation (not sklearn's). Used for training-time monitoring.
- `iou_score(y_true, y_pred)` — IoU/Jaccard score.

Note: `scripts/evaluate.py` also uses sklearn's `precision_score` and `recall_score` for reporting. The custom `f1_score` and sklearn's may give slightly different results on edge cases.
