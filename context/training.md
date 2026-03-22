# Training

## Existing Checkpoints

Two trained checkpoints are in `results/saved_models/EfficientCrackNet/`:

| File | Trained on | Date | Notes |
|---|---|---|---|
| `best_model_num_1.pt` | Simulated (8k images) | Nov 2025 | F1=0.77, mIoU=0.83 on sim test set |
| `best_model_num_real_1.pt` | Real (36 images) | Mar 2026 | Shows crack detection on real images, needs more data |

The `run_num` argument maps directly to the filename: `--run_num 1` → `best_model_num_1.pt`, `--run_num real_1` → `best_model_num_real_1.pt`.

---

## Training Command

```bash
python scripts/train.py \
  --data_dir data/ \
  --model_name EfficientCrackNet \
  --epochs 50 \
  --alpha 0.8 \
  --data_name deepcrack \
  --run_num 1
```

All CLI arguments and their defaults:

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Path to dataset root (must have train/images, train/masks) |
| `--model_name` | required | `EfficientCrackNet`, `UNet`, or `LMM_Net` |
| `--data_name` | required | `deepcrack` (only supported option currently) |
| `--run_num` | required | String ID for this run, used in checkpoint filenames |
| `--epochs` | required | Number of training epochs |
| `--alpha` | required | Initial Dice/IoU loss weight (0.8 recommended) |
| `--batch_size` | 8 | Reduce to 4 or 2 if out of GPU memory |
| `--learning_rate` | 0.001 | Adam learning rate |
| `--optim_w_decay` | 2e-4 | Adam weight decay |
| `--lr_decay` | 0.8 | ReduceLROnPlateau factor |
| `--num_epochs_decay` | 5 | ReduceLROnPlateau patience |
| `--validate` | False | Enable train/val split |
| `--num_workers` | 8 | DataLoader workers (reduce if CPU bottleneck) |
| `--subset_size` | None | Train on N random images (for quick testing) |

---

## Loss Function

```
total_loss = BCE(output, mask) + alpha * (Dice(output, mask) + IoU(output, mask))
```

Alpha schedule: decreases by 0.2 every 60 epochs starting from the initial value. At epoch 0: alpha=0.8. At epoch 60: alpha=0.6. At epoch 120: alpha=0.4. Etc.

The model does NOT apply sigmoid before computing loss — the loss functions handle raw logits. However the model DOES apply sigmoid in `forward()` before returning the output. This means:
- During training: `outputs = model(input)` → outputs already have sigmoid applied → loss functions must accept [0,1] inputs (they do)
- During eval: same — sigmoid already applied, do NOT apply again

---

## Outputs

Training saves to:
- `results/saved_models/{model_name}/best_model_num_{run_num}.pt` — best checkpoint (lowest loss)
- `results/plots/{model_name}/run_{run_num}/loss_plots.jpg` — loss curve
- `results/tensorboard/{model_name}/run_{run_num}/` — TensorBoard logs

Checkpoint format (saved by `save_checkpoint` in `data/dataset.py`):
```python
{
    'model_state_dict': model.state_dict(),
    'loss': best_loss,
    'validate': args.validate
}
```
Load with: `torch.load(path, weights_only=False)['model_state_dict']`

---

## TensorBoard

```bash
tensorboard --logdir=results/tensorboard/EfficientCrackNet/run_1/
# then open http://localhost:6006
```

---

## Training Tips

**Quick sanity check** — train on a tiny subset to verify the pipeline works before a full run:
```bash
python scripts/train.py --data_dir data/ --model_name EfficientCrackNet --epochs 3 --alpha 0.8 --data_name deepcrack --run_num test --subset_size 50
```

**Out of GPU memory** — reduce `--batch_size` to 4 or 2.

**Slow training** — reduce `--num_workers` if CPU is the bottleneck, or set `--persistent_workers True --pin_memory True` for faster data loading.

**Fine-tuning on real data from simulated checkpoint** — load the simulated checkpoint and continue training on real images with a lower learning rate (e.g. 0.0001). Currently not implemented as a flag — requires modifying `train.py` to load weights before training.

---

## Two Training Modes

`scripts/train.py` has two code paths depending on `--validate`:

- `run_without_validation()` — trains on full training set, saves checkpoint when train loss improves. Used for final training runs.
- `run_with_validation()` — splits training data, saves checkpoint when validation loss improves. Better for hyperparameter tuning.
