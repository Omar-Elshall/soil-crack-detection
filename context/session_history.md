# Session History

This file is updated at the end of each working session so any Claude instance — on any machine — can pick up where we left off without re-explanation.

---

## Session: 2026-03-23 | Branch: `improve-real-model`

### What was done

**Image augmentation pipeline added to `DeepCrackDataset`** (`crack_detection/data/dataset.py`)

Spatial transforms (applied to image and mask identically):
- Random rotation ±30° (p=0.5)
- Random resized crop, scale 0.7–1.0, ratio 0.9–1.1, output 512×512 (p=0.4)
- Horizontal flip (p=0.5), Vertical flip (p=0.5)

Photometric (image only):
- ColorJitter: brightness=0.35, contrast=0.22, saturation=0.3, hue=0.02
- Random gamma shift 0.6–1.5 (p=0.4)
- Gaussian blur kernel 3 or 5 (p=0.3)
- Rare invert (p=0.05)

augmentation_prob raised to 0.85.

---

## Session: 2026-03-27 to 2026-03-29 | Branch: `improve-real-model`

### What was done

- Dataset expanded to 101 train images
- num_workers fix for WSL2 + /mnt/c bottleneck
- Alpha going negative fix (floor at 0.2)
- `--pretrained_path` added to train.py
- `--threshold` added to evaluate.py

### Model results at end of session

| Run | F1 | Notes |
|-----|-----|-------|
| `real_3` | 0.17 | 101 images, batch_size=1, 150 epochs — visually better than real_2 |
| `real_4` | 0.16 | Fine-tuned from real_3, collapsed — alpha going to 0.0 caused model to predict nothing |

---

## Session: 2026-03-29 | Branch: `improve-real-model`

### What was done

#### Major training speed overhaul

All of these optimizations were added to `scripts/train.py` and tested end-to-end:

**Flash Attention** (`crack_detection/models/mobile_vit.py`)
Replaced manual QK matmul + softmax with `F.scaled_dot_product_attention`. This eliminated CUDA OOM at batch_size > 1 (attention matrix was [B, 4, 4, 4096, 4096] = 4GB at B=1). Flash Attention is O(N) memory instead of O(N²).

**BF16 autocast** (`scripts/train.py`, `scripts/evaluate.py`)
Switched from float16 to bfloat16. BF16 has the same exponent range as float32 so it never overflows — eliminates NaN that float16 was causing on QK dot products. No GradScaler needed. Added `.float()` cast on model output before loss computation.

**Gradient accumulation**
Added `--grad_accum_steps` (default 4). `batch_size=12, accum=8` → effective batch of 96.

**DataLoader optimizations**
Added `--prefetch_factor`, `--pin_memory`, `--persistent_workers` args. `num_workers=12` works correctly on Linux filesystem (NOT on /mnt/c).

**`cudnn.benchmark = True`**
Added to training script. Auto-tunes conv kernels for fixed 512×512 inputs.

**`zero_grad(set_to_none=True)`**
Frees gradient memory immediately rather than zeroing in place.

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**
Added to `~/.zshrc`. Prevents false OOM from CUDA memory fragmentation.

#### Batch size limit on RTX 4090
- batch_size=16: OOM on ULSAM decoder layer
- batch_size=13: unstable
- batch_size=12: stable limit ✓

#### Plateau-based alpha schedule (replaces epoch-based)
Replaced `if epoch % 125 == 0` with a patience counter. Alpha steps down when loss doesn't improve by more than `1e-4` for `--alpha_patience` consecutive epochs. Training stops automatically when alpha=0.2 plateaus. LR is reset at each alpha step.

Added `--alpha_patience` arg (default 15). `--epochs` changed to default=9999 (safety cap only).

**Floating point fix:** `0.4 - 0.2 = 0.20000000000000007` in Python, so `alpha <= 0.2` was never True — model would reset LR forever instead of stopping. Fixed with `round(alpha, 1) <= 0.2`.

#### tqdm progress bar
Replaced fixed-range tqdm with `tqdm()` counter (no total). Shows elapsed time and speed per epoch. Postfix shows loss, alpha, patience.

#### Dataset fix — image/mask pairing
Previous code used sorted() on file paths, which worked by coincidence when all files were present. After orphan masks were deleted from test set, sorted order mismatched. Rewrote to pair by filename number:
```python
mask_lookup = {basename.replace('MASK', ''): path for ...}
paired = [(img, mask_lookup[basename.replace('IMG', '')]) for img in img_paths if ...]
```

#### Dataset — 22 orphan masks deleted
Test set had 42 masks but only 20 images — 22 masks had no corresponding image (likely from a previous dataset version). Deleted the orphans.

#### evaluate.py — merged with Aafan's PR improvements
Aafan (group partner) submitted PR #1 with evaluate.py improvements. PR was closed without merging; changes manually incorporated into our branch:
- 4-panel comparison image (added overlay: predicted cracks in red)
- Per-image metrics printed during eval
- `metrics_summary.csv` saved after each run
- `--output_dir` CLI arg
- Unused imports and CLI args removed

BF16 autocast preserved (was not in Aafan's version).

### Final model results this session

| Run | F1 | Recall | Precision | mIoU | Notes |
|-----|-----|--------|-----------|------|-------|
| `real_4` | **0.58** | 0.60 | 0.63 | 0.71 | 80 images, 1276 epochs, all optimizations |

Massive improvement over previous best (real_3: F1=0.17). Loss trajectory:
- alpha=0.8: 1176 epochs, loss 2.4 → 0.94
- alpha=0.6: 22 epochs, loss 0.94 → 0.72
- alpha=0.4: 36 epochs, loss 0.72 → 0.46
- alpha=0.2: 40 epochs, loss 0.46 → 0.23 → stop

### Git history cleanup
Removed `Co-Authored-By: Claude` lines from all commits using `git filter-branch --msg-filter`. Both branches force-pushed to GitHub.

### Next steps (at end of session)

1. **Label more images** — currently 80 train / 20 test. Target 150+ train for next run.
2. **Retrain real_5** with same optimizations on larger dataset.
3. **Enable branch protection** on GitHub main branch.
4. ~~**Merge `improve-real-model` into `main`**~~ — **DONE** (merged via PR #2 on GitHub).
5. **TensorRT export** for Jetson Orin Nano deployment.

---

## Session: 2026-04-01 | Branch: `improve-real-model`

### What was done

#### Git / repo sync
- Confirmed PR #2 merged `improve-real-model` → `main` on GitHub. Local `main` was behind — pulled to sync.

#### Jetson Orin Nano — Phase 1 complete

**SSH key auth set up** from WSL → Jetson (no password). SSH alias `jetson` configured in `~/.ssh/config`.

**Repo cloned** on Jetson at `~/soil-crack-detection` via HTTPS.

**PYTHONPATH** set in `~/.profile` and `~/.bashrc`:
```
PYTHONPATH=$PYTHONPATH:/home/sdp-w-nano/soil-crack-detection
```

**torchvision install journey:**
- Standard `pip install torchvision` → manylinux wheel incompatible with Jetson PyTorch C++ extensions (`operator torchvision::nms does not exist`)
- Jetson AI Lab index (`pypi.jetson-ai-lab.io/jp6/cu126`) has torch 2.10.0 + torchvision 0.25.0
- Upgraded both with `--no-deps --upgrade` from Jetson AI Lab index
- torch 2.10 required `libcudss.so.0` → installed `libcudss0-cuda-12` via apt
- Added `/usr/lib/aarch64-linux-gnu/libcudss/12` to `LD_LIBRARY_PATH` in `~/.profile`

**Model checkpoint** `best_model_num_real_4.pt` copied to Jetson via scp.

**predict.py verified working** on Jetson with CUDA:
```bash
ssh jetson "bash -l -c 'cd ~/soil-crack-detection && python3 scripts/predict.py --mode sample --data_dir data/ --model_path results/saved_models/EfficientCrackNet/best_model_num_real_4.pt --num_images 3'"
```

**Bug found:** Initial inference used ImageNet normalization (`mean=[0.485,0.456,0.406]`) that the model was never trained with — results were noisy. Dataset uses only `ToTensor()` (divide by 255, no normalization).

**New context file** `context/jetson.md` created with full hardware, software, install notes, and working commands.

### Next steps

1. **Phase 2** — Live camera inference from Arducam IMX477 via CSI (GStreamer pipeline)
2. **Phase 3** — Jetson ↔ Pixhawk 6C MAVLink over `/dev/ttyTHS1` or `/dev/ttyTHS2`
3. **Phase 4** — PID tuning for stable flight (QGroundControl)
4. **Phase 5** — Autonomous offboard control via MAVSDK-Python
