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
| Session history, what changed, next steps | `context/session_history.md` |

---

## Critical Facts

1. **`EfficientCrackNet` applies sigmoid internally** — do NOT apply `torch.sigmoid()` again in eval/inference scripts. Double-sigmoid compresses all outputs to [0.5, 0.62] → all-white masks. See `context/known_issues.md`.

2. **Best real checkpoint:** `results/saved_models/EfficientCrackNet/best_model_num_real_4.pt` — F1=0.58 on 20 test images, trained on 80 real images with all optimizations.

3. **`data/` and `results/` are gitignored** — must be set up locally via symlink or copy. See README.md.

4. **Real dataset has capitalized folder names** (`Images/`, `Masks/`) — fails silently on Linux. Use symlink adapter. See `context/dataset.md`.

5. **`pip install -e .` is required** before running any script so `crack_detection` resolves as a package.

6. **Training stops automatically** — no `--epochs` needed. The plateau-based alpha schedule stops training when alpha=0.2 plateaus. See `context/training.md`.

---

## Commands

```bash
# Setup (run once)
pip install -e .

# Evaluate — best real model
python scripts/evaluate.py --data_dir data/ --model_name EfficientCrackNet --data_name deepcrack --run_num real_4

# Evaluate — simulated model
python scripts/evaluate.py --data_dir data/ --model_name EfficientCrackNet --data_name deepcrack --run_num 1

# Train (best command — stops automatically on plateau)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py --data_dir data/ --model_name EfficientCrackNet --data_name deepcrack --alpha 0.8 --batch_size 12 --learning_rate 5e-4 --num_epochs_decay 10 --num_workers 12 --pin_memory True --persistent_workers True --grad_accum_steps 8 --prefetch_factor 4 --run_num real_4 --alpha_patience 15

# Predict (single image)
python scripts/predict.py --mode single --image img_0292 --data_dir data/

# Predict (batch)
python scripts/predict.py --mode batch --input_dir results/new_images/

# Predict (N random samples)
python scripts/predict.py --mode sample --data_dir data/ --num_images 4

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
    mobile_vit.py          # MobileViTBlock — uses Flash Attention (F.scaled_dot_product_attention)
    losses.py              # BCELoss, DiceLoss, IoULoss
  data/
    dataset.py             # DeepCrackDataset — pairing logic at lines 55-68
                           # save_checkpoint, init_weights, save_plots
    transforms.py          # 4-way rotation augmentation
  metrics.py               # f1_score, iou_score

scripts/
  train.py                 # Plateau-based alpha schedule; stops automatically
  evaluate.py              # --threshold, --output_dir; saves CSV + 4-panel comparisons
  predict.py               # --mode single|batch|sample
  visualize.py             # Configured via DATA_DIR, MODEL_PATH, OUTPUT_DIR env vars
  compare_models.py        # Loads UNet + LMM_Net + EfficientCrackNet
```

## Stack

PyTorch 2.6.0, CUDA 12.4, torchvision, numpy, Pillow, scikit-learn, tensorboard, tqdm, einops
