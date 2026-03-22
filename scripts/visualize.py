"""
Visualize model predictions on test images
Shows input image, ground truth mask, and predicted mask side by side
"""
import torch
import os
import matplotlib.pyplot as plt
from crack_detection.models.efficientcracknet import EfficientCrackNet
from crack_detection.data.dataset import DeepCrackDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

# Configuration
DATA_DIR = os.environ.get("DATA_DIR", "data/")
MODEL_PATH = os.environ.get("MODEL_PATH", "results/saved_models/EfficientCrackNet/best_model_num_1.pt")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "results/prediction_visualizations/")
NUM_SAMPLES = 10  # Number of images to visualize

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
print("Loading model...")
model = EfficientCrackNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=False)['model_state_dict'])
model.eval()
print("Model loaded successfully!")

# Create simple args object for dataset
class Args:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.model_name = 'EfficientCrackNet'

args = Args()

# Load test dataset
print(f"\nLoading test dataset...")
test_dataset = DeepCrackDataset(args, data_part='test', subset_size=NUM_SAMPLES)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Loaded {len(test_dataset)} test images")
print("\nGenerating predictions...")

# Generate predictions
with torch.no_grad():
    for idx, (input_img, gt_mask) in enumerate(test_loader):
        input_img = input_img.to(device)

        # Get prediction
        pred_mask = model(input_img)
        pred_mask = torch.sigmoid(pred_mask)  # Apply sigmoid to get probabilities

        # Move to CPU for visualization
        input_img_cpu = input_img[0].cpu().permute(1, 2, 0).numpy()
        gt_mask_cpu = gt_mask[0, 0].cpu().numpy()
        pred_mask_cpu = pred_mask[0, 0].cpu().numpy()
        pred_mask_binary = (pred_mask_cpu > 0.5).astype(float)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Input image
        axes[0, 0].imshow(input_img_cpu)
        axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Ground truth mask
        axes[0, 1].imshow(gt_mask_cpu, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Predicted probability map
        axes[1, 0].imshow(pred_mask_cpu, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Predicted Probability Map', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # Binary predicted mask
        axes[1, 1].imshow(pred_mask_binary, cmap='gray')
        axes[1, 1].set_title('Predicted Mask', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(OUTPUT_DIR, f'prediction_{idx+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization {idx+1}/{NUM_SAMPLES}: {save_path}")

print(f"\n{'='*80}")
print(f"All visualizations saved to: {OUTPUT_DIR}")
print(f"{'='*80}")
