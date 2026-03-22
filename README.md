# soil-crack-detection

UAV-based soil crack detection using EfficientCrackNet. A drone captures RGB images at fixed altitude; onboard edge inference on a Jetson Orin Nano outputs geotagged crack masks for a farmer dashboard. No cloud dependency.

## Performance

| Model | F1 | Precision | Recall | mIoU |
|---|---|---|---|---|
| EfficientCrackNet | 0.77 | 0.78 | 0.78 | 0.83 |
| U-Net | — | — | — | 0.34 |

*Evaluated on 2,000-image simulated test set.*

---

## Quick Start

### Prerequisites

- Python 3.9+
- Git
- CUDA-capable GPU (recommended) or CPU
- NVIDIA drivers + CUDA toolkit (for GPU) — [install guide](https://developer.nvidia.com/cuda-downloads)

---

## Setup — Linux / WSL (Ubuntu)

```bash
# 1. Clone
git clone git@github.com:Omar-Elshall/soil-crack-detection.git
cd soil-crack-detection

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Setup — Windows

```bash
# 1. Clone (run in Git Bash)
git clone git@github.com:Omar-Elshall/soil-crack-detection.git
cd soil-crack-detection

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Dataset Setup

The dataset is not included in the repo. You need to provide it at `data/` with this structure:

```
data/
  train/
    images/    # PNG files
    masks/     # PNG files (binary: white=crack, black=background)
  test/
    images/
    masks/
```

### Option A — Simulated dataset (Linux/WSL symlink)

```bash
# Adjust this path to where your dataset lives
SIM="/mnt/c/Users/Omar/OneDrive - aus.edu/Senior Design Project/Dataset/simulated/0"

mkdir -p data/train data/test
ln -s "$SIM/train/images" data/train/images
ln -s "$SIM/train/masks"  data/train/masks
ln -s "$SIM/test/images"  data/test/images
ln -s "$SIM/test/masks"   data/test/masks
```

### Option B — Real dataset (Linux/WSL symlink)

The real dataset uses capitalized folder names (`Images/`, `Masks/`). The symlink adapter handles the case difference:

```bash
REAL="/mnt/c/Users/Omar/OneDrive - aus.edu/Senior Design Project/Dataset/real/Real Dataset & Masks"

mkdir -p data/train data/test
ln -s "$REAL/train/Images" data/train/images
ln -s "$REAL/train/Masks"  data/train/masks
ln -s "$REAL/test/Images"  data/test/images
ln -s "$REAL/test/Masks"   data/test/masks
```

### Option C — Windows (copy or junction)

```bash
# Copy (safe, works anywhere)
xcopy /E /I "path\to\your\dataset\train\images" "data\train\images"
xcopy /E /I "path\to\your\dataset\train\masks"  "data\train\masks"
xcopy /E /I "path\to\your\dataset\test\images"  "data\test\images"
xcopy /E /I "path\to\your\dataset\test\masks"   "data\test\masks"

# Or use a directory junction (no data duplication)
mklink /J data\train\images "path\to\your\dataset\train\images"
mklink /J data\train\masks  "path\to\your\dataset\train\masks"
mklink /J data\test\images  "path\to\your\dataset\test\images"
mklink /J data\test\masks   "path\to\your\dataset\test\masks"
```

### Verify dataset loads

```bash
python -c "
import argparse, sys
sys.argv = ['','--data_dir','data/','--model_name','EfficientCrackNet','--data_name','deepcrack']
from types import SimpleNamespace
args = SimpleNamespace(data_dir='data/', model_name='EfficientCrackNet', data_name='deepcrack')
from crack_detection.data.dataset import DeepCrackDataset
d = DeepCrackDataset(args, data_part='test')
print(f'Test images found: {len(d)}')
"
```

---

## Results Directory

Training outputs (checkpoints, plots, predictions) go to `results/` which is gitignored. On a fresh clone this directory does not exist — it is created automatically when you train.

If you already have a trained checkpoint, place it at:
```
results/
  saved_models/
    EfficientCrackNet/
      best_model_num_1.pt
```

### Linux/WSL — symlink to existing results on Windows

```bash
ln -s "/mnt/c/Users/Omar/OneDrive - aus.edu/Senior Design Project/crack-detection/results" results
```

---

## Evaluate

Run evaluation on the test set:

```bash
# Simulated-trained model (run_num=1)
python scripts/evaluate.py \
  --data_dir data/ \
  --model_name EfficientCrackNet \
  --data_name deepcrack \
  --run_num 1

# Real-trained model (run_num=real_1)
python scripts/evaluate.py \
  --data_dir data/ \
  --model_name EfficientCrackNet \
  --data_name deepcrack \
  --run_num real_1
```

Output: F1, Precision, Recall, mIoU printed to console. Prediction masks and comparison images saved to `results/real_eval_predictions/`.

**Adjust threshold** (default 0.5) — edit lines 59–60 of `scripts/evaluate.py`:
```python
output_mask[output_mask >= 0.5] = 1.   # ← change this value
output_mask[output_mask < 0.5]  = 0.   # ← and this
```
Lower = more sensitive (more detections). Higher = more conservative.

---

## Predict

Three inference modes:

```bash
# Single image — saves input, ground truth, and prediction separately
python scripts/predict.py \
  --mode single \
  --image img_0292 \
  --data_dir data/

# Batch — run on a folder of images (no ground truth needed)
python scripts/predict.py \
  --mode batch \
  --input_dir results/new_images/ \
  --output_dir results/predictions/

# Random sample — N test images with side-by-side comparison
python scripts/predict.py \
  --mode sample \
  --data_dir data/ \
  --num_images 4
```

To use a specific checkpoint: `--model_path results/saved_models/EfficientCrackNet/best_model_num_real_1.pt`

---

## Train

```bash
python scripts/train.py \
  --data_dir data/ \
  --model_name EfficientCrackNet \
  --epochs 50 \
  --alpha 0.8 \
  --data_name deepcrack \
  --run_num 1
```

Key options:
- `--batch_size 4` — reduce if out of GPU memory (default 8)
- `--subset_size 100` — train on 100 random images (quick pipeline test)
- `--validate True` — enable train/val split

Monitor training:
```bash
tensorboard --logdir=results/tensorboard/EfficientCrackNet/run_1/
# open http://localhost:6006
```

---

## Visualize

```bash
# Side-by-side overlay visualizations for 10 test images
python scripts/visualize.py

# Compare all three models on the same images
python scripts/compare_models.py --data_dir data/ --run_num 1
```

---

## Project Structure

```
soil-crack-detection/
├── crack_detection/               # Python package (pip install -e .)
│   ├── models/
│   │   ├── efficientcracknet.py   # Primary model — EfficientNet + SEM + EEM
│   │   ├── baselines.py           # U-Net, LMM-Net
│   │   ├── mobile_vit.py          # MobileViT blocks (einops required)
│   │   └── losses.py              # BCE, Dice, IoU losses
│   ├── data/
│   │   ├── dataset.py             # DeepCrackDataset, data loading, checkpointing
│   │   └── transforms.py          # 4-way rotation augmentation
│   └── metrics.py                 # f1_score, iou_score
├── scripts/
│   ├── train.py                   # Training loop (Adam + ReduceLROnPlateau)
│   ├── evaluate.py                # Evaluation + mask visualization
│   ├── predict.py                 # Inference: single / batch / sample modes
│   ├── visualize.py               # Overlay predictions on input images
│   └── compare_models.py          # Multi-model side-by-side comparison
├── docs/
│   └── SETUP_README.md            # Legacy setup notes
├── configs/
│   └── default.yaml               # Default hyperparameter reference
├── context/                       # Detailed knowledge base (see below)
│   ├── architecture.md
│   ├── dataset.md
│   ├── training.md
│   ├── evaluation.md
│   ├── known_issues.md
│   └── project.md
├── data/                          # GITIGNORED — place/symlink dataset here
├── results/                       # GITIGNORED — checkpoints, plots, predictions
├── pyproject.toml
├── requirements.txt
├── CLAUDE.md                      # Claude Code guidance
└── README.md
```

---

## Context Directory

The `context/` directory contains detailed knowledge for anyone (or any AI assistant) working on this codebase:

| File | Contents |
|---|---|
| `context/architecture.md` | Model internals, SEM/EEM/ULSAM modules, sigmoid behavior, input size notes |
| `context/dataset.md` | Dataset structure, simulated vs real, case sensitivity on Linux, Blender files |
| `context/training.md` | All CLI args, loss function, alpha schedule, checkpoint format, fine-tuning tips |
| `context/evaluation.md` | Metrics explanation, threshold tuning, diagnostic output interpretation |
| `context/known_issues.md` | All confirmed bugs and fixes: double sigmoid, domain shift, hardcoded paths, etc. |
| `context/project.md` | Hardware stack, project timeline, repository structure, key decisions |

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.5.0
tqdm>=4.65.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
tifffile>=2023.4.0
tensorboard>=2.12.0
einops>=0.6.0
```

---

## References

- [EfficientCrackNet paper](https://arxiv.org/abs/2409.18099)
- [LMM-Net paper](https://ieeexplore.ieee.org/document/10539282)
- [U-Net paper](https://arxiv.org/abs/1505.04597)
