import torch
import os
import argparse
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
from crack_detection.data.dataset import CustomImageDataset, init_weights, DeepCrackDataset
from crack_detection.metrics import f1_score, iou_score
from crack_detection.models.baselines import UNet_FCN, LMM_Net
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from crack_detection.models.efficientcracknet import EfficientCrackNet
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader
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

    model.load_state_dict(torch.load(f'./saved_models/{args.model_name}/best_model_num_{args.run_num}.pt', weights_only=False)['model_state_dict'])

    model.eval()
    f1_scores, recall_scores, precision_scores, iou_scores = 0.0, 0.0, 0.0, 0.0
    num_batch = 0.0

    # --- diagnostics ---
    print(f"Dataset size : {len(test_dataloaders.dataset)} images")
    print(f"First image  : {test_dataloaders.dataset.image_path_list[0] if test_dataloaders.dataset.image_path_list else 'NONE'}")
    print(f"First mask   : {test_dataloaders.dataset.mask_path_list[0] if test_dataloaders.dataset.mask_path_list else 'NONE'}")
    print()

    # Setup output directory for prediction masks
    output_dir = './real_eval_predictions'
    os.makedirs(output_dir, exist_ok=True)
    image_paths = test_dataloaders.dataset.image_path_list
    img_idx = 0

    # metrics: IoU, BF score, F1 score, precision, recall
    for input_img, mask in test_dataloaders:
        input_img, mask = input_img.to(device), mask.to(device)
        with torch.no_grad():
            # Add batch to GPU
            output_mask= model(input_img)
            print(f"  raw output  min={output_mask.min():.4f}  max={output_mask.max():.4f}  mean={output_mask.mean():.4f}")
            print(f"  mask        min={mask.min():.4f}  max={mask.max():.4f}  positive px={mask.sum():.0f}")
            output_mask[output_mask >= 0.5] = 1.
            output_mask[output_mask < 0.5] = 0.

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

                # Save side-by-side comparison: input | ground truth | prediction
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(orig)
                axes[0].set_title('Input Image', fontsize=13)
                axes[0].axis('off')
                axes[1].imshow(gt, cmap='gray')
                axes[1].set_title('Ground Truth', fontsize=13)
                axes[1].axis('off')
                axes[2].imshow(pred, cmap='gray')
                axes[2].set_title('Prediction', fontsize=13)
                axes[2].axis('off')
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
            iou_scores += np.mean(iou)

            f1_scores += f1_s
            recall_scores += recall
            precision_scores += precision
            num_batch += 1.0

    test_f1_score = (f1_scores/num_batch)
    test_recall_score = (recall_scores/num_batch)
    test_precision_score = (precision_scores/num_batch)
    test_miou_score = (iou_scores/num_batch)

    print(f'Test F1 Score is: {round(test_f1_score, 2)}')
    print(f'Test Recall Score is: {round(test_recall_score, 2)}')
    print(f'Test Precision Score is: {round(test_precision_score, 2)}')
    print(f'Test mIoU Score is: {round(test_miou_score, 2)}')
    print(f'\nPrediction masks saved to: {output_dir}/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crack Segmentation Work')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,help='Momentum in learning rate')
    parser.add_argument('--validate', type=bool, default=False, help='whether to validate the model or not')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--epochs', type=int, help='Num of epochs')
    parser.add_argument('--alpha', type=float, help='Reduction factor in Loss function')
    parser.add_argument('--optim_w_decay', type=float, default=2e-4)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--num_epochs_decay', type=int, default=5)
    parser.add_argument('--data_name', type=str, help='Dataset to be used')
    parser.add_argument('--rgb', type=bool, help='Is image RGB or not')
    parser.add_argument('--run_num', type=str, help='run number')
    parser.add_argument('--half', type=bool, default=False, help='use half Model size or not')
    parser.add_argument('--augment', type=bool, default=False, help='whether augment dataset or not')
    parser.add_argument('--subset_size', type=int, default=None, help='use random subset of data (e.g., 10)')

    args = parser.parse_args()

    if args.data_name == 'deepcrack':
        test_dataset = DeepCrackDataset(args, data_part='test', subset_size=args.subset_size)
        test_dataloaders = DataLoader(test_dataset, batch_size=1, num_workers=0)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval(args, test_dataloaders)

