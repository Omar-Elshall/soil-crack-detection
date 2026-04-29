"""
Find which images in the train set have manual ground truth masks (exist in Dataset2).
Uses MD5 hash comparison — no assumptions about file ordering or numbering.

Usage:
    python scripts/find_manual_images.py \
        --train_dir data/Images/ \
        --dataset2_dir "/mnt/c/Users/Omar/OneDrive - aus.edu/Senior Design Project/DATASET2/Images/"
"""

import argparse
import hashlib
import shutil
from pathlib import Path


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Path to train Images/ folder")
    parser.add_argument("--dataset2_dir", required=True, help="Path to Dataset2 Images/ folder (manual only)")
    parser.add_argument("--mask_train_dir", default=None, help="Path to train Masks/ folder (optional, for copying)")
    parser.add_argument("--copy_to", default=None, help="If set, copy matched image+mask pairs here")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    dataset2_dir = Path(args.dataset2_dir)

    print(f"Hashing Dataset2 images ({dataset2_dir})...")
    dataset2_hashes = {}
    for f in sorted(dataset2_dir.glob("*")):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            dataset2_hashes[md5(f)] = f

    print(f"  {len(dataset2_hashes)} unique images in Dataset2")

    print(f"\nScanning train set ({train_dir})...")
    matched = []
    unmatched = []
    for f in sorted(train_dir.glob("*")):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            h = md5(f)
            if h in dataset2_hashes:
                matched.append(f)
            else:
                unmatched.append(f)

    print(f"  {len(matched)} manual images found in train set")
    print(f"  {len(unmatched)} pseudo-labeled or other images in train set")

    print("\n--- Manual images in train set ---")
    for f in matched:
        print(f"  {f.name}")

    if args.copy_to and args.mask_train_dir:
        mask_dir = Path(args.mask_train_dir)
        out_dir = Path(args.copy_to)
        out_img = out_dir / "Images"
        out_mask = out_dir / "Masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)

        print(f"\nCopying matched pairs to {out_dir}...")
        copied = 0
        for img_path in matched:
            # Find corresponding mask (same stem, any extension)
            stem = img_path.stem.replace("IMG", "MASK")
            mask_candidates = list(mask_dir.glob(f"{stem}*"))
            if not mask_candidates:
                # Try lowercase
                stem_lower = img_path.stem.lower().replace("img", "mask")
                mask_candidates = list(mask_dir.glob(f"{stem_lower}*"))
            if mask_candidates:
                shutil.copy2(img_path, out_img / img_path.name)
                shutil.copy2(mask_candidates[0], out_mask / mask_candidates[0].name)
                copied += 1
            else:
                print(f"  WARNING: No mask found for {img_path.name}")

        print(f"  Copied {copied} image+mask pairs to {out_dir}")


if __name__ == "__main__":
    main()
