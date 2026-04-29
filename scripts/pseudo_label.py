"""
Semi-supervised pseudo-labeling pipeline.

Subcommands:
  generate  — Run the model on unlabeled images and save predicted masks +
               a confidence_scores.csv so you can sort/prioritize review.
  accept    — Copy approved image+mask pairs into data/train/ with correct
               IMG{N}/MASK{N} naming so the existing dataset loader picks them up.
  retrain   — Fine-tune from an existing checkpoint using the expanded dataset.

Usage examples:
  # 1. Generate pseudo-masks for raw field images
  python scripts/pseudo_label.py generate \\
      --input_dir  data/unlabeled/ \\
      --output_dir data/pseudo_labels/pending/ \\
      --model_name EfficientCrackNet \\
      --run_num    real_4

  # 2. Visually review data/pseudo_labels/pending/ in any image viewer.
  #    Then accept the ones you approve (comma-separated basenames, no extension):
  python scripts/pseudo_label.py accept \\
      --pending_dir data/pseudo_labels/pending/ \\
      --data_dir    data/ \\
      --approved    field_001,field_003,field_007

  #    Or accept everything that's in pending/:
  python scripts/pseudo_label.py accept \\
      --pending_dir data/pseudo_labels/pending/ \\
      --data_dir    data/ \\
      --all

  # 3. Retrain from the expanded dataset
  python scripts/pseudo_label.py retrain \\
      --data_dir       data/ \\
      --pretrained_run real_4 \\
      --run_num        real_5
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from crack_detection.models.efficientcracknet import EfficientCrackNet


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_model(model_name: str, run_num: str, device: torch.device) -> EfficientCrackNet:
    ckpt_path = os.path.join(
        "results", "saved_models", model_name, f"best_model_num_{run_num}.pt"
    )
    if not os.path.exists(ckpt_path):
        sys.exit(f"[error] Checkpoint not found: {ckpt_path}")
    model = EfficientCrackNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[generate] Loaded checkpoint: {ckpt_path}")
    return model


def _get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])


def _predict_with_confidence(
    model: EfficientCrackNet,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float,
):
    """
    Returns (binary_mask_uint8, mean_confidence).

    binary_mask_uint8 — np.ndarray shape (512,512), values 0 or 255.
    mean_confidence   — float, mean sigmoid output over the whole image (before
                        thresholding). Higher = model is more certain overall.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        # Model applies sigmoid internally; output is in [0, 1]
        out = model(image_tensor)
    probs = out[0, 0].cpu().numpy()           # (512, 512) float
    mask  = (probs > threshold).astype(np.uint8) * 255
    return mask, float(probs.mean())


# ── generate ───────────────────────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] Device: {device}")

    model = _load_model(args.model_name, args.run_num, device)
    tf    = _get_transform()
    os.makedirs(args.output_dir, exist_ok=True)

    exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG")
    image_paths: list[str] = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        sys.exit(f"[error] No images found in {args.input_dir}")

    print(f"[generate] Found {len(image_paths)} images → {args.output_dir}")

    scores: list[dict] = []
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert("RGB")
        tensor = tf(image).unsqueeze(0)
        mask, conf = _predict_with_confidence(model, tensor, device, args.threshold)

        # Save the predicted mask
        mask_out = os.path.join(args.output_dir, f"{basename}_mask.png")
        Image.fromarray(mask, mode="L").save(mask_out)

        # Copy original image alongside mask for easy side-by-side review
        img_out = os.path.join(args.output_dir, os.path.basename(img_path))
        if not os.path.exists(img_out):
            shutil.copy2(img_path, img_out)

        scores.append({"basename": basename, "mean_confidence": f"{conf:.4f}"})
        print(f"  {basename}: conf={conf:.4f}")

    # Write confidence CSV (sorted descending — most confident first)
    scores.sort(key=lambda r: float(r["mean_confidence"]), reverse=True)
    csv_path = os.path.join(args.output_dir, "confidence_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["basename", "mean_confidence"])
        writer.writeheader()
        writer.writerows(scores)

    print(f"\n[generate] Done. Masks and confidence_scores.csv saved to {args.output_dir}")
    print(f"[generate] Open {args.output_dir} in an image viewer, review the masks,")
    print(f"           then run the 'accept' subcommand for images you approve.")


# ── accept ─────────────────────────────────────────────────────────────────────

COUNTER_FILE = "/mnt/c/Users/Omar/OneDrive - aus.edu/Senior Design Project/DATASET2/counter.txt"


def _read_counter() -> int | None:
    """Read next index from counter.txt, or None if unavailable."""
    try:
        return int(Path(COUNTER_FILE).read_text().strip())
    except Exception:
        return None


def _write_counter(next_idx: int) -> None:
    try:
        Path(COUNTER_FILE).write_text(str(next_idx))
        print(f"[accept] counter.txt updated → {next_idx}")
    except Exception as e:
        print(f"[accept] WARNING: could not update counter.txt: {e}")


def _next_index(train_images_dir: str) -> int:
    """Return next index from counter.txt, falling back to scanning IMG files."""
    idx = _read_counter()
    if idx is not None:
        return idx
    existing = glob.glob(os.path.join(train_images_dir, "IMG*.png"))
    if not existing:
        return 1
    indices = []
    for p in existing:
        name = os.path.splitext(os.path.basename(p))[0]
        num  = "".join(filter(str.isdigit, name))
        if num:
            indices.append(int(num))
    return max(indices) + 1 if indices else 1


def cmd_accept(args: argparse.Namespace) -> None:
    train_img_dir  = os.path.join(args.data_dir, "train", "images")
    train_mask_dir = os.path.join(args.data_dir, "train", "masks")
    os.makedirs(train_img_dir,  exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)

    pending_dir = args.pending_dir

    # Determine which basenames to accept
    if args.all:
        mask_paths = sorted(glob.glob(os.path.join(pending_dir, "*_mask.png")))
        basenames  = [os.path.splitext(os.path.basename(p))[0].removesuffix("_mask")
                      for p in mask_paths]
    else:
        raw = getattr(args, "approved", "") or ""
        basenames = [b.strip() for b in raw.split(",") if b.strip()]

    if not basenames:
        sys.exit("[error] No basenames to accept. Use --all or --approved name1,name2,…")

    idx = _next_index(train_img_dir)
    accepted = 0

    for basename in basenames:
        # Find original image (try common extensions)
        img_src = None
        for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
            candidate = os.path.join(pending_dir, basename + ext)
            if os.path.exists(candidate):
                img_src = candidate
                break
        mask_src = os.path.join(pending_dir, f"{basename}_mask.png")

        if img_src is None:
            print(f"[accept] WARNING: image not found for '{basename}', skipping.")
            continue
        if not os.path.exists(mask_src):
            print(f"[accept] WARNING: mask not found for '{basename}', skipping.")
            continue

        img_dst  = os.path.join(train_img_dir,  f"IMG{idx:03d}.png")
        mask_dst = os.path.join(train_mask_dir, f"MASK{idx:03d}.png")

        # Convert image to PNG if needed
        img = Image.open(img_src).convert("RGB")
        img.save(img_dst)
        shutil.copy2(mask_src, mask_dst)

        print(f"[accept] {basename} → IMG{idx:03d} / MASK{idx:03d}")
        idx += 1
        accepted += 1

    if accepted > 0:
        _write_counter(idx)

    print(f"\n[accept] {accepted} image(s) added to {train_img_dir}")
    if accepted > 0:
        print(f"[accept] Ready to retrain — run the 'retrain' subcommand next.")


# ── retrain ────────────────────────────────────────────────────────────────────

def cmd_retrain(args: argparse.Namespace) -> None:
    ckpt_path = os.path.join(
        "results", "saved_models", "EfficientCrackNet",
        f"best_model_num_{args.pretrained_run}.pt"
    )
    if not os.path.exists(ckpt_path):
        sys.exit(f"[error] Pretrained checkpoint not found: {ckpt_path}")

    # Count training images so the user can sanity-check
    train_imgs = glob.glob(os.path.join(args.data_dir, "train", "images", "IMG*.png"))
    print(f"[retrain] Training images found: {len(train_imgs)}")
    print(f"[retrain] Pretrained checkpoint:  {ckpt_path}")
    print(f"[retrain] Output run number:      {args.run_num}")
    print()

    cmd = [
        sys.executable, "scripts/train.py",
        "--data_dir",        args.data_dir,
        "--model_name",      "EfficientCrackNet",
        "--data_name",       "deepcrack",
        "--alpha",           "0.8",
        "--batch_size",      "12",
        "--learning_rate",   "5e-4",
        "--num_epochs_decay","10",
        "--num_workers",     "12",
        "--pin_memory",      "True",
        "--persistent_workers", "True",
        "--grad_accum_steps","8",
        "--prefetch_factor", "4",
        "--alpha_patience",  "15",
        "--pretrained_path", ckpt_path,
        "--run_num",         args.run_num,
    ]

    print("[retrain] Running:", " ".join(cmd))
    print("-" * 60)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.run(cmd, env=env, check=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semi-supervised pseudo-labeling pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Run model on unlabeled images and save masks.")
    p_gen.add_argument("--input_dir",  required=True,  help="Folder of raw unlabeled images.")
    p_gen.add_argument("--output_dir", required=True,  help="Where to save masks + confidence CSV.")
    p_gen.add_argument("--model_name", default="EfficientCrackNet")
    p_gen.add_argument("--run_num",    required=True,  help="Checkpoint run number (e.g. real_4).")
    p_gen.add_argument("--threshold",  type=float, default=0.5,
                       help="Binarization threshold (default 0.5).")

    # accept
    p_acc = sub.add_parser("accept", help="Move approved masks into the training set.")
    p_acc.add_argument("--pending_dir", required=True,  help="Folder with generated masks.")
    p_acc.add_argument("--data_dir",    required=True,  help="Dataset root (contains train/).")
    group = p_acc.add_mutually_exclusive_group(required=True)
    group.add_argument("--approved", metavar="NAMES",
                       help="Comma-separated basenames to accept (without extension).")
    group.add_argument("--all", action="store_true",
                       help="Accept every mask in pending_dir.")

    # retrain
    p_ret = sub.add_parser("retrain", help="Fine-tune from a checkpoint with the expanded dataset.")
    p_ret.add_argument("--data_dir",        required=True, help="Dataset root.")
    p_ret.add_argument("--pretrained_run",  required=True, help="Source checkpoint run_num.")
    p_ret.add_argument("--run_num",         required=True, help="Output checkpoint run_num.")

    args = parser.parse_args()

    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "accept":
        cmd_accept(args)
    elif args.cmd == "retrain":
        cmd_retrain(args)


if __name__ == "__main__":
    main()
