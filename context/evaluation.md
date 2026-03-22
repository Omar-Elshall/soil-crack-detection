# Evaluation & Inference

## Evaluation Script

```bash
python scripts/evaluate.py \
  --data_dir data/ \
  --model_name EfficientCrackNet \
  --data_name deepcrack \
  --run_num 1
```

Use `--run_num real_1` to evaluate the real-trained model. Use `--run_num 1` for the simulated-trained model.

### What it outputs
- Prints F1, Recall, Precision, mIoU averaged across all test images
- Saves to `results/real_eval_predictions/`:
  - `{image_name}_pred_mask.png` — binary prediction mask (white=crack, black=background)
  - `{image_name}_comparison.png` — side-by-side: input | ground truth | prediction

### Threshold
Default threshold is 0.5 — set at lines 59–60 of `scripts/evaluate.py`:
```python
output_mask[output_mask >= 0.5] = 1.
output_mask[output_mask < 0.5] = 0.
```
Lower threshold (e.g. 0.3) = more sensitive, detects more cracks but more false positives.
Higher threshold (e.g. 0.7) = more conservative, fewer false positives but misses hairline cracks.

---

## Prediction Script (Inference)

Three modes via `--mode`:

### Single image
```bash
python scripts/predict.py \
  --mode single \
  --image img_0292 \
  --data_dir data/ \
  --output_dir results/predictions/
```
Saves `{image_name}_input.png`, `{image_name}_ground_truth.png`, `{image_name}_prediction.png`.

### Batch (folder of images)
```bash
python scripts/predict.py \
  --mode batch \
  --input_dir results/new_images/ \
  --output_dir results/predictions/
```
Runs inference on every `.png`/`.jpg` in `input_dir`. No ground truth needed.

### Random sample comparison
```bash
python scripts/predict.py \
  --mode sample \
  --data_dir data/ \
  --num_images 4 \
  --output_dir results/predictions/
```
Picks `num_images` random test images and saves side-by-side comparisons.

All modes accept `--model_path` (default: `results/saved_models/EfficientCrackNet/best_model_num_1.pt`) and `--threshold` (default: 0.6).

---

## Metrics Reference

All metrics computed per batch then averaged.

**F1 Score** — harmonic mean of precision and recall. Primary metric. Target: >0.75 on simulated data.

**mIoU (mean Intersection over Union)** — mean of per-class IoU (crack class + background class). Computed from confusion matrix. More robust to class imbalance than F1.

**Precision** — of all predicted crack pixels, how many are actually crack. Low precision = too many false positives (over-detection).

**Recall** — of all actual crack pixels, how many did the model find. Low recall = missing cracks (under-detection).

For the real dataset (9 test images), some images have 0 crack pixels in the ground truth (`positive px=0`). sklearn's `precision_score` returns 0 for these with `zero_division=0`, which pulls the average down. This is expected behavior.

---

## Diagnostic Output

The eval script prints per-batch diagnostics:
```
output min=0.XX  max=0.XX  mean=0.XX
mask   min=0.00  max=1.00  positive px=NNN
```

Healthy output for a well-trained model:
- `min` near 0.0, `max` near 1.0, `mean` between 0.05–0.3 (most pixels are background)
- `positive px` > 0 for images with actual cracks

Warning signs:
- `min` and `max` both close to 0.5 → double sigmoid applied (see `context/known_issues.md`)
- `min` near 0, `max` near 0.1, all below threshold → model predicts nothing (under-detection)
- `min` near 0.6, `max` near 0.7, all above threshold → model predicts everything (over-detection)

---

## Visualize Script

```bash
python scripts/visualize.py
```
Configured via environment variables (or edit defaults at top of file):
- `DATA_DIR` — dataset root
- `MODEL_PATH` — checkpoint path
- `OUTPUT_DIR` — where to save visualizations
- `NUM_SAMPLES` — how many test images to visualize (default 10)

---

## Compare Models Script

```bash
python scripts/compare_models.py \
  --data_dir data/ \
  --run_num 1
```
Loads UNet, LMM-Net, and EfficientCrackNet side-by-side on the same test images. Useful for benchmark comparisons.
