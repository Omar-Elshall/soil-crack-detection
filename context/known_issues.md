# Known Issues & Gotchas

## 1. Double Sigmoid (FIXED)

**Symptom:** Prediction masks are all-white. Eval prints `sigmoid out min=0.50, max=0.62` — extremely narrow range, all above 0.5 threshold.

**Root cause:** `EfficientCrackNet.forward()` applies `torch.sigmoid()` internally on line 338 of `efficientcracknet.py`. If you apply `torch.sigmoid(output_mask)` again in eval/inference scripts, you get `sigmoid(sigmoid(x))` which compresses all values into [0.5, 0.62] → everything thresholds as crack.

**Fix:** Do NOT apply sigmoid in evaluation or inference scripts. The model output is already in [0, 1]. Only threshold:
```python
output_mask[output_mask >= 0.5] = 1.
output_mask[output_mask < 0.5] = 0.
```

**Status:** Fixed in `scripts/evaluate.py` and `scripts/predict.py`.

---

## 2. Domain Shift (Sim → Real)

**Symptom:** Model trained on simulated data outputs all-white masks on real images (`min=0.62, max=0.73`, all above threshold).

**Root cause:** The model learned features specific to Blender-rendered images (lighting, texture, crack appearance). Real soil photos have different color distributions, noise, lighting conditions, and crack appearances.

**Mitigation options:**
1. Use `best_model_num_real_1.pt` — trained on 36 real images, shows basic crack detection
2. Fine-tune the simulated checkpoint on real images (transfer learning)
3. Expand the real dataset (current 36 images is too small for robust generalization)
4. Add domain randomization during simulated rendering to better match real conditions

---

## 3. Real Dataset Too Small

**Symptom:** `best_model_num_real_1.pt` outputs near-constant values (~0.5–0.55) across all pixels, suggesting it converged to a trivial solution.

**Root cause:** 36 training images is insufficient for a model with millions of parameters. The model likely overfit immediately.

**Mitigation:**
- Use data augmentation (already implemented — flip, color jitter, random invert)
- Collect more real images
- Use the Roboflow dataset (`Dataset/roboflow/`) to supplement
- Apply stronger regularization (higher weight decay)

---

## 4. Linux Case Sensitivity for Real Dataset

**Symptom:** `Dataset size: 0 images` when using the real dataset on Linux/WSL.

**Root cause:** The real dataset folders are named `Images/` and `Masks/` (capital first letter). The dataset loader globs for `images/` and `masks/` (lowercase). On Windows (case-insensitive) this works fine. On Linux it fails silently — glob returns empty list.

**Fix:** Create a symlink adapter with lowercase names:
```bash
mkdir -p data/train data/test
REAL="/mnt/c/.../Dataset/real/Real Dataset & Masks"
ln -s "$REAL/train/Images" data/train/images
ln -s "$REAL/train/Masks"  data/train/masks
ln -s "$REAL/test/Images"  data/test/images
ln -s "$REAL/test/Masks"   data/test/masks
```

---

## 5. Hardcoded Paths (FIXED)

**Symptom:** `FileNotFoundError: ./saved_models/EfficientCrackNet/best_model_num_1.pt`

**Root cause:** After project reorganization, all output paths moved from `./saved_models/`, `./plots/`, `./tensorboard/` to `./results/saved_models/`, `./results/plots/`, `./results/tensorboard/`. Several scripts still had the old paths hardcoded.

**Status:** Fixed in `scripts/train.py`, `scripts/evaluate.py`, and `crack_detection/data/dataset.py`.

---

## 6. `pyproject.toml` Build Backend

**Symptom:** `pip install -e .` fails with `BackendUnavailable: Cannot import 'setuptools.backends.legacy'`

**Root cause:** Wrong build backend string in `pyproject.toml`.

**Fix:** Use `"setuptools.build_meta"` not `"setuptools.backends.legacy:build"`.

**Status:** Fixed in current `pyproject.toml`.

---

## 7. Missing `einops` Dependency

**Symptom:** `ModuleNotFoundError: No module named 'einops'`

**Root cause:** `einops` is used by `crack_detection/models/mobile_vit.py` but was missing from `requirements.txt`.

**Status:** Fixed — `einops>=0.6.0` added to `requirements.txt`.

---

## 8. `data/` gitignore Pattern Too Broad (FIXED)

**Symptom:** `crack_detection/data/` package files not staged when running `git add`.

**Root cause:** `.gitignore` had `data/` which matched any directory named `data` including `crack_detection/data/`.

**Fix:** Changed to `/data/` (root-anchored) so only the top-level `data/` directory is ignored.

**Status:** Fixed in `.gitignore`.
