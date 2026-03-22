# soil-crack-detection

UAV-based soil crack detection using EfficientCrackNet on a Jetson Orin Nano.

A Holybro X500 v2 drone with Arducam IMX477 captures RGB images at fixed altitude. Onboard inference runs on a Jetson Orin Nano and outputs geotagged crack masks for a farmer dashboard. No cloud dependency.

## Performance (Simulated Dataset)

| Model | F1 | Precision | Recall | mIoU |
|---|---|---|---|---|
| EfficientCrackNet | 0.77 | 0.78 | 0.78 | 0.83 |
| U-Net | — | — | — | 0.34 |

## Setup

```bash
pip install -r requirements.txt
```

Place your dataset at `data/` following the structure:
```
data/
  train/images/   # PNG
  train/masks/    # PNG
  test/images/
  test/masks/
```

## Usage

```bash
# Train
python scripts/train.py --data_dir data/ --model_name EfficientCrackNet --epochs 50 --alpha 0.8 --data_name deepcrack --run_num 1

# Evaluate
python scripts/evaluate.py --data_dir data/ --model_name EfficientCrackNet --data_name deepcrack --run_num 1

# Predict
python scripts/predict.py

# Visualize
python scripts/visualize.py
```

## Architecture

```
crack_detection/
  models/
    efficientcracknet.py   # EfficientNet encoder-decoder with SEM + EEM modules
    baselines.py           # U-Net and LMM-Net
    losses.py              # BCE, Dice, IoU losses
  data/
    dataset.py             # Dataset loaders
    transforms.py          # Augmentation pipeline
  metrics.py               # F1, IoU, Precision, Recall
scripts/                   # train, evaluate, predict, visualize
dashboard/                 # Farmer-facing web UI
```

## References

- [EfficientCrackNet paper](https://arxiv.org/abs/2409.18099)
- [LMM-Net paper](https://ieeexplore.ieee.org/document/10539282)
- [U-Net paper](https://arxiv.org/abs/1505.04597)
