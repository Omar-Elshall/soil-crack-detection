# Session History

This file is updated at the end of each working session so any Claude instance — on any machine — can pick up where we left off without re-explanation.

---

## Session: 2026-03-23 | Branch: `improve-real-model`

### What was done

**Image augmentation pipeline added to `DeepCrackDataset`** (`crack_detection/data/dataset.py`)

The old dataset had no augmentation. A full augmentation pipeline was added directly in `__getitem__` for `data_part == 'train'`, controlled by `self.augmentation_prob = 0.7`. Spatial transforms are applied identically to image and mask; photometric transforms are image-only.

Spatial:
- Random rotation ±30° (p=0.5)
- Random resized crop, scale 0.7–1.0, ratio 0.9–1.1, output 512×512 (p=0.4)
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)

Photometric:
- ColorJitter: brightness=0.35, contrast=0.22, saturation=0.3, hue=0.02 (always applied when augmentation triggers)
- Random gamma shift 0.6–1.5 (p=0.4) — simulates exposure variation from drone altitude
- Gaussian blur kernel 3 or 5 (p=0.3) — simulates drone camera focus variation
- Rare invert (p=0.05)

Note: the `--augment` CLI flag in `train.py` is parsed but NOT wired to the dataset — augmentation is always active during training regardless of that flag.

**`train.py` cleanup** (`scripts/train.py`)

Removed the `del inputs / del masks / torch.cuda.empty_cache() / gc.collect()` block that was previously commented out — it was causing ~90% training slowdown when active. Already commented out but left as dead code; now removed cleanly.

### Bugs hit during this session

1. **OneDrive sync (WSL)** — all 36 real training images returned `OSError: [Errno 5] Input/output error`. Root cause: symlinks in `data/train/images/` and `data/train/masks/` point to `/mnt/c/Users/OmarE/OneDrive - aus.edu/...` which was not synced (files were online-only). Fix: open OneDrive on Windows → right-click dataset folder → "Always keep on this device". Not a code issue.

2. **CUDA OOM on WSL machine (8 GB GPU)** — `MobileViTBlock1` receives a 128×128 feature map (after two maxpools from 512×512 input). With `patch_size=(2,2)` the rearrange produces n=64×64=4096 tokens. Full self-attention matrix at batch_size=4 is `[4, 4, 4heads, 4096, 4096]` ≈ 4 GB. Crashed with `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB`. Fix for 8 GB GPU: `--batch_size 1`. On a 4090 (24 GB): use `--batch_size 8` or higher.

### Current model checkpoints
- `results/saved_models/EfficientCrackNet/best_model_num_1.pt` — simulated data, F1=0.77
- `results/saved_models/EfficientCrackNet/best_model_num_real_1.pt` — real data, 36 images, underperforms (domain shift + data scarcity)
- `results/saved_models/EfficientCrackNet/best_model_num_real_2.pt` — **IN PROGRESS** — real data + augmentation pipeline (this run)

### Training command for desktop (RTX 4090, 24 GB VRAM)

Run this from the repo root after pulling `improve-real-model`:

```bash
PYTHONPATH=. python3 scripts/train.py \
    --data_dir data/ \
    --model_name EfficientCrackNet \
    --data_name deepcrack \
    --epochs 150 \
    --alpha 0.8 \
    --batch_size 8 \
    --learning_rate 3e-4 \
    --run_num real_2 \
    --num_workers 4 \
    --persistent_workers True \
    --pin_memory True
```

Checkpoints save to `results/saved_models/EfficientCrackNet/best_model_num_real_2.pt`.
Monitor with:
```bash
tensorboard --logdir=results/tensorboard/EfficientCrackNet/run_real_2/
```

Evaluate after training:
```bash
python scripts/evaluate.py --data_dir data/ --model_name EfficientCrackNet --data_name deepcrack --run_num real_2
```

### Next steps (carry into next session)
- Evaluate `real_2` vs `real_1` — expect F1 improvement from augmentation
- If still underfitting: collect more real images or use fine-tuning from the simulated checkpoint
- Longer term: TensorRT export for Jetson Orin Nano deployment
