# Dataset

## Overview

Two datasets are used in this project: a large synthetic (simulated) dataset and a small real-world dataset. Both live outside the git repo and must be linked/copied locally.

---

## Simulated Dataset

**Location (Windows):** `<project-root>/Dataset/simulated/0/`
**Size:** 8,000 train + 2,000 test images
**Format:** PNG, 512×512

Generated in Blender with domain randomization:
- Soil color variation (brown, red, grey, sandy)
- Lighting variation (sun angle, intensity, shadow direction)
- Crack width variation (hairline to 5mm+)
- Camera altitude variation (2m–8m simulated)
- Background texture variation

Structure:
```
simulated/0/
  train/
    images/    # img_0001.png … img_8000.png
    masks/     # img_0001.png … img_8000.png  (binary: white=crack, black=background)
  test/
    images/    # img_8001.png … img_10000.png
    masks/     # img_8001.png … img_10000.png
```

Mask convention: pixel value 255 = crack, 0 = background. After `transforms.ToTensor()` this becomes 1.0 / 0.0.

**Benchmark results on simulated test set:**
| Model | F1 | Precision | Recall | mIoU |
|---|---|---|---|---|
| EfficientCrackNet | 0.77 | 0.78 | 0.78 | 0.83 |
| U-Net | — | — | — | 0.34 |

---

## Real Dataset

**Location (Windows):** `<project-root>/Dataset/real/Real Dataset & Masks/`
**Size:** 36 train + 9 test images
**Format:** PNG, ~720×720 (varies)

Actual soil crack photographs taken in the field.

Structure on disk:
```
Real Dataset & Masks/
  train/
    Images/    # IMG001.png … IMG036.png   ← capital I
    Masks/     # MASK001.png … MASK036.png ← capital M
  test/
    Images/    # IMG004.png, IMG011.png … (9 images)
    Masks/     # MASK004.png, MASK011.png …
```

### Linux/WSL case sensitivity issue
The dataset loader (`data/dataset.py`) globs for `test/images/*.png` (lowercase). On Linux this won't match `test/Images/`. **Do not rename the actual files.** Instead create a symlink adapter:

```bash
mkdir -p data/train data/test
REAL="/mnt/c/Users/Omar/OneDrive - aus.edu/Senior Design Project/Dataset/real/Real Dataset & Masks"
ln -s "$REAL/train/Images" data/train/images
ln -s "$REAL/train/Masks"  data/train/masks
ln -s "$REAL/test/Images"  data/test/images
ln -s "$REAL/test/Masks"   data/test/masks
```

### Image/mask filename mismatch
Image files are named `IMG*.png` and masks `MASK*.png`. The loader uses `sorted(glob(...))` independently for each. This works correctly because the numeric suffixes sort in the same order (IMG004↔MASK004, IMG011↔MASK011, etc.). Do not add files that would break this sorted correspondence.

### Domain shift
The model trained on simulated data does **not** generalize to real images. See `context/known_issues.md` for details. Always use `best_model_num_real_1.pt` when evaluating on real images.

---

## Roboflow Dataset

**Location (Windows):** `<project-root>/Dataset/roboflow/`

Two exports: `final.v1i.folder/` (classification) and `final.v1i.multiclass/` (multiclass segmentation). Not currently used in training. Contains real crack images exported from Roboflow with augmentation. Could be used to augment the real training set.

---

## Dataset Loader

**File:** `crack_detection/data/dataset.py`

`DeepCrackDataset` — primary loader used by all scripts.

Constructor:
```python
DeepCrackDataset(args, data_part='train'|'test'|'valid', subset_size=None)
```

- `data_part='valid'` uses the training set (no separate validation split exists)
- `subset_size=N` randomly samples N images (seed=42) — useful for quick debugging
- Augmentation (random flip, color jitter, random invert) only applied during `data_part='train'`

Glob patterns are at lines 55–64. If your dataset has a different folder structure, change them there.

Preprocessing applied to all images:
1. `transforms.Resize((512, 512))`
2. `transforms.ToTensor()` — normalizes to [0, 1]
3. If mask has >1 channel: `transforms.Grayscale(num_output_channels=1)`

---

## Blender Project Files

**Location (Windows):** `<project-root>/Dataset/blender/`

- `Cracked Soil.blend` — main Blender scene for rendering
- `CrackedSoil_GPU_Render.bat` — Windows batch script to trigger GPU render
- `force_gpu.py` — Python script run inside Blender to force CUDA rendering
- `assets/materials/` — Blender material files (soil textures, crack procedural material)
- `brick_factory_01_4k.hdr` — HDRI lighting map
- `*.psd` — Photoshop files used for manual mask creation and verification
