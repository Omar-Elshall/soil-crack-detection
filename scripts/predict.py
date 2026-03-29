"""
Prediction script — three modes:
  single  : predict on one named image, export input / ground truth / prediction
  batch   : predict on every image in a folder
  sample  : pick N random test images and save side-by-side comparisons

Usage examples:
  python scripts/predict.py --mode single --image img_0292 --data_dir data/
  python scripts/predict.py --mode batch  --input_dir data/new_images/
  python scripts/predict.py --mode sample --data_dir data/ --num_images 4
"""
import argparse
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from crack_detection.models.efficientcracknet import EfficientCrackNet

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device) -> EfficientCrackNet:
    model = EfficientCrackNet().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False)['model_state_dict'])
    model.eval()
    return model


def get_transform(size: int = 512) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


def predict_tensor(model, image_tensor, device, threshold=0.6):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        out = model(image_tensor)
        # model applies sigmoid internally — do not apply again
    return (out[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Mode: single
# ---------------------------------------------------------------------------

def run_single(args, model, device):
    os.makedirs(args.output_dir, exist_ok=True)
    tf = get_transform(512)

    test_imgs = sorted(glob.glob(os.path.join(args.data_dir, "test/images/*.png")))
    test_masks = sorted(glob.glob(os.path.join(args.data_dir, "test/masks/*.png")))

    image_path = mask_path = None
    for ip, mp in zip(test_imgs, test_masks):
        if args.image in os.path.basename(ip):
            image_path, mask_path = ip, mp
            break

    if image_path is None:
        print(f"ERROR: '{args.image}' not found in test set.")
        print("First 10 available:")
        for p in test_imgs[:10]:
            print(f"  {os.path.basename(p)}")
        return

    image = Image.open(image_path)
    mask  = Image.open(mask_path)
    name  = args.image

    # Input
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(image)); plt.axis('off'); plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{name}_input.png'), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Ground truth
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(mask), cmap='gray'); plt.axis('off'); plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{name}_ground_truth.png'), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Prediction
    pred = predict_tensor(model, tf(image).unsqueeze(0), device, args.threshold)
    plt.figure(figsize=(8, 8))
    plt.imshow(pred, cmap='gray'); plt.axis('off'); plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{name}_prediction.png'), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved to {args.output_dir}: {name}_input.png, _ground_truth.png, _prediction.png")


# ---------------------------------------------------------------------------
# Mode: batch
# ---------------------------------------------------------------------------

def run_batch(args, model, device):
    os.makedirs(args.output_dir, exist_ok=True)
    tf = get_transform(512)

    image_paths = (glob.glob(os.path.join(args.input_dir, "*.png")) +
                   glob.glob(os.path.join(args.input_dir, "*.jpg")))

    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    for img_path in image_paths:
        image = Image.open(img_path)
        pred  = predict_tensor(model, tf(image).unsqueeze(0), device, args.threshold)
        out_name = os.path.basename(img_path).rsplit('.', 1)[0] + '_mask.png'
        plt.imsave(os.path.join(args.output_dir, out_name), pred, cmap='gray')
        print(f"Saved: {out_name}")

    print(f"\nAll masks saved to {args.output_dir}")


# ---------------------------------------------------------------------------
# Mode: sample
# ---------------------------------------------------------------------------

def run_sample(args, model, device):
    os.makedirs(args.output_dir, exist_ok=True)
    tf = get_transform(512)

    test_imgs  = sorted(glob.glob(os.path.join(args.data_dir, "test/images/*.png")))
    test_masks = sorted(glob.glob(os.path.join(args.data_dir, "test/masks/*.png")))

    random.seed(args.seed)
    indices = random.sample(range(len(test_imgs)), min(args.num_images, len(test_imgs)))

    for idx in indices:
        image = Image.open(test_imgs[idx])
        mask  = Image.open(test_masks[idx])
        name  = os.path.splitext(os.path.basename(test_imgs[idx]))[0]

        input_arr = np.array(image.resize((512, 512)))
        gt_arr    = np.array(mask.resize((512, 512)))
        if gt_arr.ndim == 3:
            gt_arr = gt_arr[:, :, 0]
        gt_arr = (gt_arr > 127).astype(np.uint8) * 255

        pred = predict_tensor(model, tf(image).unsqueeze(0), device, args.threshold)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0)
        axes[0].imshow(input_arr);                       axes[0].axis('off')
        axes[1].imshow(gt_arr,  cmap='gray', vmin=0, vmax=255); axes[1].axis('off')
        axes[2].imshow(pred,    cmap='gray', vmin=0, vmax=255); axes[2].axis('off')

        save_path = os.path.join(args.output_dir, f'{name}_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {name}_comparison.png")

    print(f"\nAll comparisons saved to {args.output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EfficientCrackNet prediction utility")
    p.add_argument('--mode',       choices=['single', 'batch', 'sample'], default='single')
    p.add_argument('--data_dir',   default='data/',                       help='Root dataset dir (single/sample modes)')
    p.add_argument('--input_dir',  default='results/new_images/',         help='Input image folder (batch mode)')
    p.add_argument('--output_dir', default='results/predictions/',        help='Where to save outputs')
    p.add_argument('--model_path', default='results/saved_models/EfficientCrackNet/best_model_num_1.pt')
    p.add_argument('--image',      default='img_0292',                    help='Image name (single mode)')
    p.add_argument('--num_images', type=int, default=4,                   help='How many random samples (sample mode)')
    p.add_argument('--threshold',  type=float, default=0.5)
    p.add_argument('--seed',       type=int, default=43)
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.model_path, device)
    print("Model loaded.")

    if args.mode == 'single':
        run_single(args, model, device)
    elif args.mode == 'batch':
        run_batch(args, model, device)
    elif args.mode == 'sample':
        run_sample(args, model, device)
