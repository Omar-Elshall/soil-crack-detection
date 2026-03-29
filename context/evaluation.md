# Evaluation & Inference

## Evaluation Script

```bash
python scripts/evaluate.py \
  --data_dir data/ \
  --model_name EfficientCrackNet \
  --data_name deepcrack \
  --run_num real_4
```

Use `--run_num real_4` for best real model. Use `--run_num 1` for simulated model.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Path to dataset root |
| `--model_name` | required | `EfficientCrackNet`, `UNet`, or `LMM_Net` |
| `--data_name` | required | `deepcrack` |
| `--run_num` | required | Maps to `best_model_num_{run_num}.pt` |
| `--threshold` | 0.5 | Binarization threshold |
| `--output_dir` | `./results/real_eval_predictions` | Where to save masks and comparisons |
| `--subset_size` | None | Evaluate on N random test images only |

### What it outputs

- Prints per-image F1, Precision, Recall, mIoU during eval
- Prints final averaged metrics across all test images
- Saves to `--output_dir`:
  - `{image_name}_pred_mask.png` — binary prediction mask
  - `{image_name}_comparison.png` — 4-panel: input | ground truth | prediction | overlay (red=crack)
  - `metrics_summary.csv` — per-image metrics table

### Threshold

Default 0.5. Lower = more sensitive (more cracks detected, more false positives). Higher = more conservative.

```bash
# Test different thresholds without retraining
python scripts/evaluate.py --data_dir data/ --model_name EfficientCrackNet --data_name deepcrack --run_num real_4 --threshold 0.3
```

---

## Prediction Script (Inference)

Three modes via `--mode`:

### Single image
```bash
python scripts/predict.py --mode single --image img_0292 --data_dir data/
```

### Batch (folder of images)
```bash
python scripts/predict.py --mode batch --input_dir results/new_images/
```

### Random sample comparison
```bash
python scripts/predict.py --mode sample --data_dir data/ --num_images 4
```

All modes accept `--model_path` (default: `results/saved_models/EfficientCrackNet/best_model_num_1.pt`) and `--threshold` (default: 0.6).

---

## Metrics Reference

All metrics computed per image (batch_size=1 in eval), then averaged.

**F1 Score** — harmonic mean of precision and recall. Primary metric.

**mIoU** — mean IoU across both classes (crack + background). More robust to class imbalance than F1.

**Precision** — of predicted crack pixels, how many are actually crack. Low = too many false positives.

**Recall** — of actual crack pixels, how many were found. Low = missing cracks.

### Model results history

| Run | F1 | Recall | Precision | mIoU | Notes |
|-----|-----|--------|-----------|------|-------|
| `1` (sim) | 0.77 | 0.78 | 0.78 | 0.83 | Simulated data |
| `real_2` | 0.50 | 0.43 | 0.70 | 0.70 | 36 real images |
| `real_3` | 0.17 | 0.27 | 0.16 | 0.55 | 101 images, visually better than real_2 |
| `real_4` | **0.58** | **0.60** | **0.63** | **0.71** | 80 images, all optimizations, 1276 epochs |

---

## Diagnostic Output

The eval script prints per-image diagnostics:
```
output min=0.XX  max=0.XX  mean=0.XX
mask   min=0.00  max=1.00  positive px=NNN
[IMG0000] F1=0.6123  Precision=0.6500  Recall=0.5800  mIoU=0.7100
```

Healthy output:
- `min` near 0.0, `max` near 1.0, `mean` between 0.05–0.3

Warning signs:
- `min` and `max` both near 0.5 → double sigmoid (see `known_issues.md`)
- All values below threshold → model predicts nothing (under-detection)
- All values above threshold → model predicts everything (over-detection)
