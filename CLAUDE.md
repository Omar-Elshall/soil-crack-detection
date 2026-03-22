# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install package (required before running scripts)
pip install -e .

# Train
python scripts/train.py --data_dir data/ --model_name EfficientCrackNet --epochs 50 --alpha 0.8 --data_name deepcrack --run_num 1

# Evaluate
python scripts/evaluate.py --data_dir data/ --model_name EfficientCrackNet --data_name deepcrack --run_num 1

# Predict (single image)
python scripts/predict.py --mode single --image img_0292 --data_dir data/

# Predict (batch)
python scripts/predict.py --mode batch --input_dir results/new_images/

# Predict (N random samples with side-by-side comparison)
python scripts/predict.py --mode sample --data_dir data/ --num_images 4

# Monitor training
tensorboard --logdir=results/tensorboard/EfficientCrackNet/run_1/

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Architecture

**Package:** `crack_detection/`

| File | Purpose |
|------|---------|
| `models/efficientcracknet.py` | Primary model — EfficientNet backbone with SEM (spatial attention) and EEM (edge enhancement) modules. Input: 192×256 px, output: binary mask. |
| `models/baselines.py` | UNet_FCN and LMM_Net baselines |
| `models/mobile_vit.py` | MobileViT blocks (experimental variants) |
| `models/losses.py` | BCE, Dice, IoU loss functions |
| `data/dataset.py` | CustomImageDataset, DeepCrackDataset, save_plots, init_weights, save_checkpoint |
| `data/transforms.py` | 4-way rotation augmentation pipeline |
| `metrics.py` | f1_score, iou_score |

**Scripts:** `scripts/`

| File | Purpose |
|------|---------|
| `train.py` | Training loop (Adam, ReduceLROnPlateau) |
| `evaluate.py` | Evaluation — F1, IoU, Precision, Recall |
| `predict.py` | Single image, batch, or random sample inference |
| `visualize.py` | Prediction overlays |
| `compare_models.py` | Side-by-side model comparison |

## Dataset

Place data at `data/` (gitignored). On WSL, symlink from Windows:
```bash
ln -s "/mnt/c/Users/Omar/OneDrive - aus.edu/Senior Design Project/Dataset/simulated/0" data
```

Expected structure:
```
data/
  train/images/   # 8000 PNG
  train/masks/    # 8000 PNG
  test/images/    # 2000 PNG
  test/masks/     # 2000 PNG
```
Dataset loader in `data/dataset.py` expects this exact structure (glob patterns around line 55–64).

## Outputs (gitignored, in `results/`)

On WSL, symlink from Windows:
```bash
ln -s "/mnt/c/Users/Omar/OneDrive - aus.edu/Senior Design Project/crack-detection/results" results
```

- `results/saved_models/EfficientCrackNet/best_model_num_1.pt`
- `results/plots/EfficientCrackNet/run_1/loss_plots.jpg`
- `results/tensorboard/EfficientCrackNet/run_1/`
- `results/predictions/`

## Stack

PyTorch 2.6.0, CUDA 12.4, torchvision, numpy, Pillow, scikit-learn, tensorboard, tqdm
