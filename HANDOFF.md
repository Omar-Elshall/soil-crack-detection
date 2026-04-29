# Session Handoff — Soil Crack Detection Project

## Project Summary
Senior design project (CSE490/491) at AUS. Autonomous UAV system for detecting soil cracks using a custom deep learning model (EfficientCrackNet), deployed on a Jetson Orin Nano 8GB mounted on a Holybro X500 V2 drone with a Pixhawk 6C flight controller and Arducam IMX477 camera.

---

## Current Model Performance (real_6 — FINAL)
| Metric | Score |
|---|---|
| F1 | 0.83 |
| Precision | 0.85 |
| Recall | 0.83 |
| mIoU | 0.86 |

- Checkpoint: `results/saved_models/EfficientCrackNet/best_model_num_real_6.pt`
- Test set: 71 images (80/20 split of full dataset)
- Note: IMG011 and IMG066 in test set have empty masks (no cracks) — mIoU is degenerate (0.5) for these, not a real model failure

---

## Dataset State
- Total images: 351 (280 train / 71 test), 80/20 random split with seed=42
- Images are named IMG001–IMGxxx, masks MASK001–MASKxxx
- Dataset lives in DATASET2 on OneDrive: `C:\Users\Omar\OneDrive - aus.edu\Senior Design Project\DATASET2\`
- On WSL: `/mnt/c/Users/Omar/OneDrive - aus.edu/Senior Design Project/DATASET2/`
- `counter.txt` in DATASET2 tracks next available IMG index for train — currently 281
- `data/` in the repo is a symlink to DATASET2

---

## What Was Built This Session

### 1. Polygon Mission Planner (`jetson/ui/src/pages/PlanPage.tsx`)
Replaced the old rectangle-based area selector with a free-form polygon planner:
- Click vertices on the map to draw any polygon shape
- Self-intersection validation (invalid polygon = error message, can't generate)
- Scanline clipping algorithm generates lawnmower grid inside the polygon
- Works for any convex or concave field shape, any orientation
- Same `Waypoint[]` format sent to MAVLink — nothing else changed
- All flight parameters (altitude, speed, overlap) and stats unchanged

### 2. Semi-Supervised Pipeline (`scripts/pseudo_label.py`)
Three subcommands:
```bash
# Generate pseudo-masks for unlabeled images
python scripts/pseudo_label.py generate \
    --input_dir data/unlabeled/ \
    --output_dir data/pseudo_labels/pending/ \
    --run_num real_6

# Accept approved masks into training set (reads/updates counter.txt automatically)
python scripts/pseudo_label.py accept \
    --pending_dir data/pseudo_labels/pending/approved/ \
    --data_dir data/ --all

# Fine-tune from existing checkpoint
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/pseudo_label.py retrain \
    --data_dir data/ --pretrained_run real_6 --run_num real_7
```
- `generate` also writes `confidence_scores.csv` (sorted by model confidence)
- `accept` reads starting index from `counter.txt` and writes back after adding images
- `counter.txt` path is hardcoded in script: `/mnt/c/Users/Omar/OneDrive - aus.edu/Senior Design Project/DATASET2/counter.txt`

### 3. Mask Review UI (`scripts/review_masks.py`)
Single-file Python web server for reviewing pseudo-labels:
```bash
python scripts/review_masks.py --pending_dir data/pseudo_labels/pending/
# Opens http://localhost:7000
```
- Shows 3 panels: Original | Predicted Mask | Red overlay (cracks highlighted in red on original)
- Arrow keys or buttons: → approve, ← reject
- Approve → moves pair to `pending/approved/`, Reject → `pending/rejected/`
- Done screen shows the accept command ready to copy

---

## Web App Architecture
- React + TypeScript + Vite (`jetson/ui/`)
- Three FastAPI microservices on Jetson (ports 8001/8002/8003): inference, mavlink, data
- UI served as pre-built static files via Python http.server on port 5173
- Start everything: `bash jetson/start.sh` (run from `~/soil-crack-detection` on Jetson)
- To rebuild UI after source changes: `cd jetson/ui && npm install && npm run build`
- Jetson IP: 192.168.1.222, SSH: `ssh jetson`

---

## Report Changes Needed
These sections need updating in the final report (lives in OneDrive, not repo):

| Section | What to update |
|---|---|
| Model Performance | Update to real_6: F1=0.83, P=0.85, R=0.83, mIoU=0.86 |
| Mission Planning | Replace rectangle grid with polygon description + screenshots |
| Training Data Strategy | Add semi-supervised loop (generate → review → accept → retrain) |
| System Architecture | Update data pipeline diagram to include semi-supervised flow |
| Future Work | Remove semi-supervised (it's done) |

Also worth noting: images with all-zero masks should be excluded from mIoU reporting or noted as edge cases.

---

## Key File Paths
| File | Purpose |
|---|---|
| `jetson/ui/src/pages/PlanPage.tsx` | Polygon mission planner UI |
| `scripts/pseudo_label.py` | Semi-supervised pipeline (generate/accept/retrain) |
| `scripts/review_masks.py` | Tinder-style mask review web UI |
| `context/` | Architecture, dataset, training, evaluation, jetson docs |
| `CLAUDE.md` | Critical facts + commands for working in this repo |
