import torch
import os 
import pickle
import gc
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
from crack_detection.data.dataset import CustomImageDataset, DeepCrackDataset, save_plots, save_training_plot_only, save_checkpoint, init_weights
from crack_detection.models.baselines import UNet_FCN, LMM_Net
from crack_detection.models.efficientcracknet import EfficientCrackNet
from crack_detection.models.losses import BCELoss, DiceLoss, IoULoss
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


def run_with_validation(args, model, train_dataloaders, valid_dataloaders, plot_path):

    # Training loop
    epochs = []                
    epoch_train_loss = []
    epoch_valid_loss = []
    best_loss = np.inf
    best_path = f'./results/saved_models/{args.model_name}/best_model_num_{args.run_num}.pt'
    bce_loss = BCELoss()
    dice_loss = DiceLoss()
    miou_loss = IoULoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.optim_w_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_decay, patience=args.num_epochs_decay)
    print('------------  Training started! --------------')
    num_epochs = args.epochs
    alpha = args.alpha
    for epoch in tqdm(range(num_epochs)):
        model.train()

        b_train_loss = []
        if epoch % 20 == 0:
            alpha = max(0.0, alpha - 0.2)
        for i, (inputs, masks) in enumerate(train_dataloaders):
            inputs, masks = inputs.to(device), masks.to(device)
            
            optimizer.zero_grad()    
            outputs = model(inputs)

            loss = bce_loss(outputs, masks.to(device)) + (alpha * (dice_loss(outputs, masks.to(device)) + miou_loss(outputs, masks.to(device))))
            b_train_loss.append(loss.item())

            # Removed unnecessary CUDA cleanup - was causing 90% slowdown!
            # del inputs
            # del masks
            # torch.cuda.empty_cache()
            # gc.collect()

            loss.backward()
            optimizer.step()
        
        epoch_train_loss.append(np.mean(b_train_loss))
        lr_scheduler.step(np.mean(b_train_loss))
        print('Epoch: {}'.format(epoch+1))
        print('Training Loss: {}'.format(np.mean(b_train_loss)))

        model.eval()

        with torch.no_grad():
            b_valid_loss = []
            for i, (inputs, masks) in enumerate(valid_dataloaders):
                inputs = inputs.to(device)
                outputs = model(inputs)

                val_loss = bce_loss(outputs, masks.to(device))
                b_valid_loss.append(val_loss.item())

            print('Validation Loss {}'.format(np.mean(b_valid_loss)))
            print('-' * 40)
            epoch_valid_loss.append(np.mean(b_valid_loss))
            
            if np.mean(b_valid_loss) < best_loss:
                best_loss = np.mean(b_valid_loss)
                save_checkpoint(best_path, model, np.mean(b_valid_loss), args.validate)
        epochs.append(epoch+1)

    print("Training complete!")

    save_plots(epoch_train_loss, epoch_valid_loss, epochs, args)

    # writer.flush()

def run_without_validation(args, model, train_dataloaders, plot_path):

    # Training loop
    epochs = []                
    epoch_train_loss = []
    best_loss = np.inf
    best_path = './results/saved_models/{}'.format(args.model_name) + '/' + 'best_model_num_{}.pt'.format(args.run_num)
    bce_loss = BCELoss()
    dice_loss = DiceLoss()
    miou_loss = IoULoss()

    # if args.model_name == 'LMM_Net':
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.optim_w_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_decay, patience=args.num_epochs_decay)

    print('------------  Training started! --------------')

    num_epochs = args.epochs
    alpha = args.alpha
    for epoch in tqdm(range(num_epochs)):
        model.train()

        b_train_loss = []
        if epoch % 60 == 0:
            alpha -= 0.2

        for i, (inputs, labels) in enumerate(train_dataloaders):
            inputs, masks = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()    
            outputs = model(inputs)
            loss = bce_loss(outputs, masks.to(device)) + (alpha * (dice_loss(outputs, masks.to(device)) + miou_loss(outputs, masks.to(device))))
            b_train_loss.append(loss.item())

            # Removed unnecessary CUDA cleanup - was causing 90% slowdown!
            # del inputs
            # del labels
            # torch.cuda.empty_cache()
            # gc.collect()

            loss.backward()
            optimizer.step()

        if np.mean(b_train_loss) < best_loss:
            best_loss = np.mean(b_train_loss)
            save_checkpoint(best_path, model, np.mean(b_train_loss), args.validate)
        
        epoch_train_loss.append(np.mean(b_train_loss))
        lr_scheduler.step(np.mean(b_train_loss))
        print('Epoch: {}'.format(epoch+1))
        print('Training Loss: {}'.format(np.mean(b_train_loss)))

        epochs.append(epoch+1)

    print("Training complete!")
    save_training_plot_only(epoch_train_loss, epochs, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
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
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--persistent_workers', type=bool, default=False, help='keep workers alive between epochs')
    parser.add_argument('--pin_memory', type=bool, default=False, help='use pinned memory for faster transfer')
    parser.add_argument('--subset_size', type=int, default=None, help='use random subset of data (e.g., 200)')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # writer = SummaryWriter('./TB_runs/')
    plot_path= f'./results/plots/{args.model_name}/run_{args.run_num}/'
    model_path= f'./results/saved_models/{args.model_name}/run_{args.run_num}/'
    tensorboard_path = f'./results/tensorboard/{args.model_name}/run_{args.run_num}/'

    # Create directories if they don't exist
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    writer = SummaryWriter(tensorboard_path)
    
    if args.model_name == 'UNet':
        model = UNet_FCN(args = args, scaler=2).to(device)
        model.apply(init_weights)
    elif args.model_name == 'LMM_Net':
        model = LMM_Net().to(device)
    elif args.model_name == 'EfficientCrackNet':
        model = EfficientCrackNet().to(device)

    if args.data_name == 'deepcrack':
        train_dataset = DeepCrackDataset(args, data_part='train', subset_size=args.subset_size)
        train_dataloaders = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            pin_memory=args.pin_memory
        )

        valid_dataset = DeepCrackDataset(args, data_part='valid', subset_size=args.subset_size)
        valid_dataloaders = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            pin_memory=args.pin_memory
        )

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Cuda is availabe: ', torch.cuda.is_available())

    if args.validate:
        run_with_validation(args, model, train_dataloaders, valid_dataloaders, plot_path)
    else:
        run_without_validation(args, model, train_dataloaders, plot_path)

# python main_dev.py --data_dir C:/Users/jwkor/Documents/UNM/crack_segmentation/dataset/DeepCrack/ --validate True --model_name UNet --epochs 50 --alpha 0.8 --data_name deepcrack --run_num 1