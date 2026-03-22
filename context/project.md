# Project Context

## What This Is

Senior Design I & II project at the American University of Sharjah (AUS), CSE490/CSE491. The goal is an end-to-end UAV-based system for detecting soil cracks in agricultural fields, producing geotagged crack maps for a farmer dashboard. All inference runs onboard the drone — no cloud dependency.

## Hardware Stack

| Component | Part |
|---|---|
| Frame | Holybro X500 v2 |
| Flight controller | Pixhawk |
| Camera | Arducam IMX477 (12MP, fixed focal length) |
| Edge compute | NVIDIA Jetson Orin Nano |
| Ground station | Laptop running Mission Planner |

The Jetson Orin Nano runs inference using this model in real time as the drone flies a survey pattern at fixed altitude (~4m AGL). Results are geotagged and pushed to a farmer dashboard over WiFi when the drone returns to the base station. The dashboard UI mockup (`soil_mapping_ui.html`) lives in `docs/presentations/` — it is a static report mockup, not production code.

## Software Stack

| Layer | Technology |
|---|---|
| Model | PyTorch 2.6.0, CUDA 12.4 |
| Inference runtime | PyTorch (TensorRT export planned) |
| Dashboard | HTML/CSS/JS (standalone, no server required) |
| Flight planning | Mission Planner (MAVLink) |
| Dataset generation | Blender 4.x with Python scripting |

## Project Timeline

- **Fall 2025 (Senior Design I):** Literature review, dataset generation (Blender), baseline model training (U-Net), initial EfficientCrackNet training on simulated data. F1=0.77 achieved.
- **Spring 2026 (Senior Design II, ongoing):** Real dataset collection (36 images), real-image fine-tuning, hardware integration, dashboard development, Jetson deployment.

## Repository Structure

```
soil-crack-detection/
├── crack_detection/          # Importable Python package
│   ├── models/
│   │   ├── efficientcracknet.py   # Primary model
│   │   ├── baselines.py           # U-Net, LMM-Net
│   │   ├── mobile_vit.py          # MobileViT blocks
│   │   └── losses.py              # BCE, Dice, IoU losses
│   ├── data/
│   │   ├── dataset.py             # DeepCrackDataset loader
│   │   └── transforms.py          # Augmentation pipeline
│   └── metrics.py                 # F1, IoU
├── scripts/
│   ├── train.py                   # Training loop
│   ├── evaluate.py                # Evaluation with metrics + mask output
│   ├── predict.py                 # Inference (single/batch/sample modes)
│   ├── visualize.py               # Overlay visualizations
│   └── compare_models.py          # Multi-model comparison
├── docs/
│   └── SETUP_README.md            # Legacy setup guide
├── configs/
│   └── default.yaml               # Default hyperparameters
├── context/                       # Detailed knowledge base for Claude Code
│   ├── architecture.md
│   ├── dataset.md
│   ├── training.md
│   ├── evaluation.md
│   ├── known_issues.md
│   └── project.md                 # This file
├── data/                          # GITIGNORED — symlink or copy dataset here
├── results/                       # GITIGNORED — checkpoints, plots, predictions
├── pyproject.toml
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Files Outside the Repo (Windows only)

The full project directory at `C:\Users\Omar\OneDrive - aus.edu\Senior Design Project\` contains:
- `Dataset/` — all datasets (simulated, real, roboflow, blender source files)
- `docs/` — all project reports, presentations, admin documents
- `archive/` — old model implementations (EfficientCrackNet-OG, U-Net)
- `MISC/` — artifacts, temp files, duplicates (nothing important)

## Key Decisions Made

**Why synthetic data?** Real crack images at the right altitude and lighting are hard to collect. Blender allows generating thousands of labeled images with controlled variation.

**Why EfficientCrackNet over U-Net?** 4.44% higher precision, 0.77% higher mIoU, significantly smaller model size — critical for Jetson deployment.

**Why no cloud?** Agricultural fields often have no connectivity. All processing must run onboard.

**Why gitignore `data/` and `results/`?** Dataset is 10GB+. Model checkpoints are binary. Neither belongs in git. Users set up their own local data symlink/copy.
