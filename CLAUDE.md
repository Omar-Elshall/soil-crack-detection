# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For deep context on any topic, read the relevant file in `context/`:

| Need context on… | Read |
|---|---|
| Model internals, sigmoid behavior, SEM/EEM | `context/architecture.md` |
| Dataset structure, real vs simulated, case sensitivity | `context/dataset.md` |
| Training loop, CLI args, checkpoints, loss function | `context/training.md` |
| Evaluation metrics, threshold, diagnostic output | `context/evaluation.md` |
| Past bugs and fixes | `context/known_issues.md` |
| Project goals, hardware, history, file layout | `context/project.md` |

---

## Critical Facts

1. **`EfficientCrackNet` applies sigmoid internally** — do NOT apply `torch.sigmoid()` again in eval/inference scripts. Double-sigmoid compresses all outputs to [0.5, 0.62] → all-white masks. See `context/known_issues.md`.

2. **Two checkpoints exist:**
   - `results/saved_models/EfficientCrackNet/best_model_num_1.pt` — trained on simulated data (F1=0.77)
   - `results/saved_models/EfficientCrackNet/best_model_num_real_1.pt` — trained on real data (36 images)

3. **`data/` and `results/` are gitignored** — must be set up locally via symlink or copy. See README.md.

4. **Real dataset has capitalized folder names** (`Images/`, `Masks/`) — fails silently on Linux. Use symlink adapter. See `context/dataset.md`.

5. **`pip install -e .` is required** before running any script so `crack_detection` resolves as a package.

---

## Commands

```bash
# Setup (run once)
pip install -e .

# Evaluate — simulated model
python scripts/evaluate.py --data_dir data/ --model_name EfficientCrackNet --data_name deepcrack --run_num 1

# Evaluate — real model
python scripts/evaluate.py --data_dir data/ --model_name EfficientCrackNet --data_name deepcrack --run_num real_1

# Train
python scripts/train.py --data_dir data/ --model_name EfficientCrackNet --epochs 50 --alpha 0.8 --data_name deepcrack --run_num 1

# Predict (single image)
python scripts/predict.py --mode single --image img_0292 --data_dir data/

# Predict (batch)
python scripts/predict.py --mode batch --input_dir results/new_images/

# Predict (N random samples)
python scripts/predict.py --mode sample --data_dir data/ --num_images 4

# Monitor training
tensorboard --logdir=results/tensorboard/EfficientCrackNet/run_1/

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Package Layout

```
crack_detection/
  models/
    efficientcracknet.py   # Primary model — sigmoid applied internally in forward()
    baselines.py           # UNet_FCN, LMM_Net
    mobile_vit.py          # MobileViTBlock (requires einops)
    losses.py              # BCELoss, DiceLoss, IoULoss
  data/
    dataset.py             # DeepCrackDataset — glob patterns at lines 55-64
                           # save_checkpoint, init_weights, save_plots
    transforms.py          # 4-way rotation augmentation
  metrics.py               # f1_score, iou_score

scripts/
  train.py                 # Output paths: results/saved_models/, results/plots/, results/tensorboard/
  evaluate.py              # Threshold at lines 59-60; output to results/real_eval_predictions/
  predict.py               # --mode single|batch|sample
  visualize.py             # Configured via DATA_DIR, MODEL_PATH, OUTPUT_DIR env vars
  compare_models.py        # Loads UNet + LMM_Net + EfficientCrackNet
```

## Stack

PyTorch 2.6.0, CUDA 12.4, torchvision, numpy, Pillow, scikit-learn, tensorboard, tqdm, einops
