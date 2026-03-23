import torch
import os
import argparse
import csv
import numpy as np
from crack_detection.data.dataset import init_weights, DeepCrackDataset
from crack_detection.metrics import f1_score
from crack_detection.models.baselines import UNet_FCN, LMM_Net
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from crack_detection.models.efficientcracknet import EfficientCrackNet
from torch.utils.data import DataLoader
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Testing model
def eval(args, test_dataloaders):

    if args.model_name == 'UNet':
        model = UNet_FCN(args = args, scaler=2).to(device)
        model.apply(init_weights)
    elif args.model_name == 'LMM_Net':
        model = LMM_Net().to(device)
    elif args.model_name == 'EfficientCrackNet':
        model = EfficientCrackNet().to(device)

    model.load_state_dict(torch.load(f'./results/saved_models/{args.model_name}/best_model_num_{args.run_num}.pt', weights_only=False)['model_state_dict'])

    model.eval()
    f1_scores, recall_scores, precision_scores, iou_scores = 0.0, 0.0, 0.0, 0.0
    num_batch = 0.0
    per_image_results = []

    # --- diagnostics ---
    print(f"Dataset size : {len(test_dataloaders.dataset)} images")
    print(f"First image  : {test_dataloaders.dataset.image_path_list[0] if test_dataloaders.dataset.image_path_list else 'NONE'}")
    print(f"First mask   : {test_dataloaders.dataset.mask_path_list[0] if test_dataloaders.dataset.mask_path_list else 'NONE'}")
    print()

    # Setup output directory for prediction masks
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    image_paths = test_dataloaders.dataset.image_path_list
    img_idx = 0

    # metrics: IoU, BF score, F1 score, precision, recall
    for input_img, mask in test_dataloaders:
        input_img, mask = input_img.to(device), mask.to(device)
        with torch.no_grad():
            # Add batch to GPU
            output_mask= model(input_img)
            print(f"  output min={output_mask.min():.4f}  max={output_mask.max():.4f}  mean={output_mask.mean():.4f}")
            print(f"  mask        min={mask.min():.4f}  max={mask.max():.4f}  positive px={mask.sum():.0f}")
            output_mask[output_mask >= args.threshold] = 1.
            output_mask[output_mask < args.threshold] = 0.

            # Save prediction mask and side-by-side comparison for each image
            for b in range(output_mask.shape[0]):
                img_name = os.path.splitext(os.path.basename(image_paths[img_idx]))[0]

                orig = input_img[b].cpu().permute(1, 2, 0).numpy()
                orig = np.clip(orig, 0, 1)
                gt = mask[b, 0].cpu().numpy()
                pred = output_mask[b, 0].cpu().numpy()

                # Save raw binary prediction mask
                pred_png = PILImage.fromarray((pred * 255).astype(np.uint8))
                pred_png.save(os.path.join(output_dir, f'{img_name}_pred_mask.png'))

                # Build overlay: predicted cracks in red on top of input image
                overlay = orig.copy()
                overlay[pred == 1] = [1.0, 0.0, 0.0]  # red pixels where crack predicted

                # Save side-by-side comparison: input | ground truth | prediction | overlay
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(orig)
                axes[0].set_title('Input Image', fontsize=13)
                axes[0].axis('off')
                axes[1].imshow(gt, cmap='gray')
                axes[1].set_title('Ground Truth', fontsize=13)
                axes[1].axis('off')
                axes[2].imshow(pred, cmap='gray', vmin=0, vmax=1)
                axes[2].set_title('Prediction', fontsize=13)
                axes[2].axis('off')
                axes[3].imshow(overlay)
                axes[3].set_title('Overlay (red=crack)', fontsize=13)
                axes[3].axis('off')
                plt.suptitle(img_name, fontsize=11, y=1.01)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{img_name}_comparison.png'), dpi=150, bbox_inches='tight')
                plt.close()

                img_idx += 1

            y_true = mask.cpu().numpy().flatten().astype(int)
            y_pred = output_mask.cpu().numpy().flatten().astype(int)
            f1_s = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            intersection = np.diag(cm)
            ground_truth_set = cm.sum(axis=1)
            predicted_set = cm.sum(axis=0)
            union = ground_truth_set + predicted_set - intersection
            union_f = union.astype(np.float32)
            iou = np.where(union_f > 0, intersection / union_f, 0.0)
            batch_miou = np.mean(iou)
            iou_scores += batch_miou

            f1_scores += f1_s
            recall_scores += recall
            precision_scores += precision
            num_batch += 1.0

            # Per-image metrics
            img_name_for_metric = os.path.splitext(os.path.basename(image_paths[img_idx - 1]))[0]
            print(f"  [{img_name_for_metric}] F1={f1_s:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  mIoU={batch_miou:.4f}")
            per_image_results.append({
                'image': img_name_for_metric,
                'f1': round(f1_s, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'miou': round(batch_miou, 4),
            })

    test_f1_score = (f1_scores/num_batch)
    test_recall_score = (recall_scores/num_batch)
    test_precision_score = (precision_scores/num_batch)
    test_miou_score = (iou_scores/num_batch)

    print(f'Test F1 Score is: {round(test_f1_score, 2)}')
    print(f'Test Recall Score is: {round(test_recall_score, 2)}')
    print(f'Test Precision Score is: {round(test_precision_score, 2)}')
    print(f'Test mIoU Score is: {round(test_miou_score, 2)}')
    print(f'\nPrediction masks saved to: {output_dir}/')

    # Save per-image metrics to CSV
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'f1', 'precision', 'recall', 'miou'])
        writer.writeheader()
        writer.writerows(per_image_results)
    print(f'Per-image metrics saved to: {csv_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crack Segmentation Work')
    parser.add_argument('--data_dir', type=str, required=True, help='path to dataset root (must have test/images, test/masks)')
    parser.add_argument('--model_name', type=str, required=True, help='EfficientCrackNet, UNet, or LMM_Net')
    parser.add_argument('--data_name', type=str, required=True, help='dataset name (deepcrack)')
    parser.add_argument('--run_num', type=str, required=True, help='run number — maps to best_model_num_{run_num}.pt')
    parser.add_argument('--threshold', type=float, default=0.5, help='binarization threshold (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default='./results/real_eval_predictions', help='directory to save masks and comparisons')
    parser.add_argument('--subset_size', type=int, default=None, help='evaluate on N random test images only')

    args = parser.parse_args()

    if args.data_name == 'deepcrack':
        test_dataset = DeepCrackDataset(args, data_part='test', subset_size=args.subset_size)
        test_dataloaders = DataLoader(test_dataset, batch_size=1, num_workers=0)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval(args, test_dataloaders)

