# Training

## Existing Checkpoints

| File | Trained on | Images | F1 | Notes |
|---|---|---|---|---|
| `best_model_num_1.pt` | Simulated | 8k | 0.77 | mIoU=0.83 on sim test set |
| `best_model_num_real_1.pt` | Real | 36 | — | Underperforms, too little data |
| `best_model_num_real_2.pt` | Real | 36 | 0.50 | Augmentation helped |
| `best_model_num_real_3.pt` | Real | 101 | 0.17 | batch_size=1, 150 epochs — visually better than real_2 despite lower F1 |
| `best_model_num_real_4.pt` | Real | 80 | **0.58** | Best real model. Trained with all optimizations — Flash Attention, BF16, gradient accumulation, plateau-based alpha schedule. 1276 epochs, final loss 0.23 |

The `run_num` argument maps directly to the filename: `--run_num real_4` → `best_model_num_real_4.pt`.

---

## Best Training Command (Current Standard)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py \
  --data_dir data/ \
  --model_name EfficientCrackNet \
  --data_name deepcrack \
  --alpha 0.8 \
  --batch_size 12 \
  --learning_rate 5e-4 \
  --num_epochs_decay 10 \
  --num_workers 12 \
  --pin_memory True \
  --persistent_workers True \
  --grad_accum_steps 8 \
  --prefetch_factor 4 \
  --run_num real_4 \
  --alpha_patience 15
```

One-liner (no line breaks):
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py --data_dir data/ --model_name EfficientCrackNet --data_name deepcrack --alpha 0.8 --batch_size 12 --learning_rate 5e-4 --num_epochs_decay 10 --num_workers 12 --pin_memory True --persistent_workers True --grad_accum_steps 8 --prefetch_factor 4 --run_num real_4 --alpha_patience 15
```

No `--epochs` needed — training stops automatically when the model plateaus at alpha=0.2.

---

## All CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Path to dataset root (must have train/images, train/masks) |
| `--model_name` | required | `EfficientCrackNet`, `UNet`, or `LMM_Net` |
| `--data_name` | required | `deepcrack` (only supported option) |
| `--run_num` | required | String ID for this run, used in checkpoint filenames |
| `--alpha` | required | Initial Dice/IoU loss weight (0.8 recommended) |
| `--epochs` | 9999 | Safety cap — training stops on plateau before this |
| `--alpha_patience` | 15 | Epochs without improvement before alpha steps down |
| `--batch_size` | 2 | 12 is the RTX 4090 limit. 16 OOMs on ULSAM decoder |
| `--learning_rate` | 0.001 | 5e-4 recommended for real data |
| `--grad_accum_steps` | 4 | Gradient accumulation. batch=12, accum=8 → effective batch=96 |
| `--num_workers` | 0 | 12 recommended for i9-13900K on Linux filesystem |
| `--prefetch_factor` | 2 | Batches prefetched per worker. 4 recommended |
| `--pin_memory` | False | Faster CPU→GPU transfer. Use True when num_workers > 0 |
| `--persistent_workers` | False | Keep workers alive between epochs. Use True when num_workers > 0 |
| `--optim_w_decay` | 2e-4 | Adam weight decay |
| `--lr_decay` | 0.8 | ReduceLROnPlateau factor |
| `--num_epochs_decay` | 5 | ReduceLROnPlateau patience. 10 recommended |
| `--validate` | False | Enable train/val split |
| `--subset_size` | None | Train on N random images (for quick testing) |
| `--pretrained_path` | None | Load checkpoint before training (fine-tuning) |

---

## Loss Function

```
total_loss = BCE(output, mask) + alpha * (Dice(output, mask) + IoU(output, mask))
```

Alpha starts high (0.8) to emphasize spatial accuracy (Dice+IoU), then decays toward 0.2 where BCE dominates (pixel-wise classification). This curriculum helps the model first learn where cracks are, then sharpen the boundaries.

**Plateau-based alpha schedule (current):** Alpha steps down by 0.2 when training loss doesn't improve by more than `1e-4` for `--alpha_patience` consecutive epochs. On step-down, LR is reset to `--learning_rate` and ReduceLROnPlateau is re-initialized. Training stops when alpha=0.2 plateau is reached.

Each alpha step causes a large sudden drop in loss (~40-50%) because the LR reset unlocks new learning capacity. This is expected and desired.

**What each alpha phase looks like:**
- alpha=0.8 → 1176 epochs on real_4, loss 2.4 → 0.94
- alpha=0.6 → 22 epochs, loss 0.94 → 0.72
- alpha=0.4 → 36 epochs, loss 0.72 → 0.46
- alpha=0.2 → 40 epochs, loss 0.46 → 0.23 → stop

**Floating point gotcha:** `0.4 - 0.2` in Python = `0.20000000000000007`, not exactly `0.2`. The stopping condition uses `round(alpha, 1) <= 0.2` to handle this.

---

## Speed Optimizations (all active in best command)

| Optimization | What it does | Impact |
|---|---|---|
| Flash Attention (`F.scaled_dot_product_attention`) | O(N) memory vs O(N²) for self-attention | Enables batch_size=12 vs batch_size=1 |
| BF16 autocast | Half-precision with float32 exponent range — no overflow, no GradScaler | ~2× faster, no NaN |
| Gradient accumulation | Simulates larger batch without extra VRAM | More stable gradients |
| `num_workers=12` | Parallel data loading on i9-13900K | Keeps GPU fed |
| `pin_memory=True` | Direct DMA to GPU | Minor speedup |
| `persistent_workers=True` | No process respawn per epoch | Saves ~1-2s per epoch |
| `prefetch_factor=4` | Workers queue 4 batches ahead | Smoother GPU feeding |
| `cudnn.benchmark=True` | Auto-tunes conv kernels for fixed input size | ~5-10% speedup |
| `zero_grad(set_to_none=True)` | Frees gradient memory immediately | Reduces peak VRAM |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Prevents false OOM from memory fragmentation | Required for stability |

Training speed on RTX 4090: ~3.1-3.5s/epoch with 80 images, batch_size=12, grad_accum_steps=8.

---

## Loss Function Notes

The model applies sigmoid internally in `forward()`. Loss functions receive [0,1] outputs — they do NOT receive raw logits. BF16 autocast wraps only the forward pass; loss is computed in float32.

---

## Outputs

Training saves to:
- `results/saved_models/{model_name}/best_model_num_{run_num}.pt` — best checkpoint (lowest train loss)
- `results/plots/{model_name}/run_{run_num}/loss_plots.jpg` — loss curve

Checkpoint format:
```python
{'model_state_dict': model.state_dict(), 'train_loss': best_loss}
```
Load with: `torch.load(path, weights_only=False)['model_state_dict']`

---

## Two Training Modes

`scripts/train.py` has two code paths depending on `--validate`:

- `run_without_validation()` — trains on full training set, plateau-based alpha schedule, stops automatically. **This is the standard mode.**
- `run_with_validation()` — splits training data, saves checkpoint on val loss improvement, fixed epoch count. Better for hyperparameter tuning.

---

## Tips

**Quick sanity check** — verify the pipeline works before a full run:
```bash
python scripts/train.py --data_dir data/ --model_name EfficientCrackNet --alpha 0.8 --data_name deepcrack --run_num test --subset_size 10 --epochs 3
```

**Out of GPU memory** — reduce `--batch_size`. On RTX 4090: 12 is the limit, 8 is safe. OOM happens on the ULSAM decoder layer, not the attention blocks.

**Data on WSL2 Linux filesystem** — `num_workers=12` works well. Do NOT put data on `/mnt/c` (OneDrive/Windows drive) — the WSL2 filesystem bridge kills multiprocessing performance. Copy data to `~/data/` on the Linux side.
