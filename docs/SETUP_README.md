# EfficientCrackNet Setup Guide

This guide explains how to run the EfficientCrackNet model on your custom crack segmentation dataset.

## What Was Set Up

### 1. Repository Structure
The EfficientCrackNet repository has been cloned to:
```
C:\Users\Omar\OneDrive - aus.edu\Senior Design Project\Models\EfficientCrackNet\Crack-Segmentation\
```

### 2. Dataset Configuration
Your dataset is located at:
```
C:\Users\Omar\OneDrive - aus.edu\Senior Design Project\Dataset\Simulated Renders\0\
```

Structure:
- `train/images/` - 8000 training images (PNG format)
- `train/masks/` - 8000 training masks (PNG format)
- `test/images/` - 2000 test images (PNG format)
- `test/masks/` - 2000 test masks (PNG format)

### 3. Code Modifications
The `utils.py` file has been modified to work with your dataset structure:
- Changed from `train_img/*.jpg` to `train/images/*.png`
- Changed from `train_lab/*.png` to `train/masks/*.png`
- Same changes for test data

### 4. Dependencies Installed
All required Python packages have been installed:
- PyTorch 2.6.0 (with CUDA 12.4 support)
- torchvision
- numpy, Pillow, matplotlib
- scikit-learn
- tqdm, tensorboard, tifffile

## How to Use

### Option 1: Quick Start (Recommended)
Simply run the custom training script:
```bash
cd "C:\Users\Omar\OneDrive - aus.edu\Senior Design Project\Models\EfficientCrackNet\Crack-Segmentation"
python train_custom.py
```

This will train the EfficientCrackNet model with:
- 50 epochs
- Batch size: 8
- Learning rate: 0.001
- Alpha: 0.8 (for loss function)

### Option 2: Manual Training
Run the training script with custom parameters:
```bash
python main_dev.py --data_dir "C:\Users\Omar\OneDrive - aus.edu\Senior Design Project\Dataset\Simulated Renders\0" --model_name EfficientCrackNet --epochs 50 --alpha 0.8 --data_name deepcrack --run_num 1
```

### Testing the Setup
Before starting full training, verify everything works:
```bash
python test_dataloader.py
```

This will:
- Load your training and test datasets
- Display dataset sizes
- Show sample image/mask shapes
- Verify the DataLoader works correctly

### Evaluation
After training is complete, evaluate the model:
```bash
python eval_custom.py
```

Or manually:
```bash
python eval.py --data_dir "C:\Users\Omar\OneDrive - aus.edu\Senior Design Project\Dataset\Simulated Renders\0" --model_name EfficientCrackNet --data_name deepcrack --run_num 1
```

## Output Locations

### Trained Models
Saved to: `./saved_models/EfficientCrackNet/best_model_num_1.pt`

### Training Plots
Saved to: `./plots/EfficientCrackNet/run_1/loss_plots.jpg`

### TensorBoard Logs
Saved to: `./tensorboard/EfficientCrackNet/run_1/`

To view TensorBoard:
```bash
tensorboard --logdir=./tensorboard/EfficientCrackNet/run_1/
```

## Model Architecture
EfficientCrackNet is a lightweight semantic segmentation model that:
- Uses EfficientNet as the backbone
- Optimized for crack detection in concrete surfaces
- Input size: 192x256 pixels
- Output: Binary segmentation mask (crack vs. no crack)

## Training Parameters

You can modify these in `train_custom.py`:
- `EPOCHS`: Number of training epochs (default: 50)
- `BATCH_SIZE`: Batch size for training (default: 8)
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `ALPHA`: Weight for Dice and IoU loss (default: 0.8, decreases during training)
- `VALIDATE`: Whether to use validation split (default: False)

## GPU Support
The code automatically detects and uses CUDA if available. Your system has:
- PyTorch with CUDA 12.4 support installed
- Check if GPU is detected by running: `python -c "import torch; print(torch.cuda.is_available())"`

## Troubleshooting

### Out of Memory
If you run out of GPU memory, reduce `BATCH_SIZE` to 4 or 2.

### Slow Training
- Ensure CUDA is being used (check the output when training starts)
- Reduce `num_workers` in DataLoader if CPU is bottleneck
- Consider using a smaller subset of data for testing

### Different Dataset Structure
If you need to use a different dataset structure, modify the paths in `utils.py` around line 55-64.

## Next Steps
1. Run `python test_dataloader.py` to verify setup
2. Run `python train_custom.py` to start training
3. Monitor training progress in the console
4. After training, run `python eval_custom.py` to evaluate performance
5. View results in the plots directory

Good luck with your crack segmentation project!
