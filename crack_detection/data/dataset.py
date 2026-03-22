import torch
import os
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from pathlib import Path
import glob
import random
import tifffile


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None, fused=None):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.fused = fused
        # gives list of entire path to each image along the img_dir
        self.image_path_list = sorted(glob.glob(os.path.join(self.img_dir, "*.png"))) if not self.fused else sorted(glob.glob(os.path.join(self.img_dir, "*.tif")))
        self.mask_path_list = sorted(glob.glob(os.path.join(self.mask_dir, "*.bmp")))

        self.img_dir = img_dir  # directory for train, valid, or test 
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        # TODO
        image = np.array(Image.open(self.image_path_list[idx])) if not self.fused else tifffile.imread(self.image_path_list[idx])
        image = torch.Tensor(np.moveaxis(image, [0,1,2], [2,1,0]))

        mask = np.array(Image.open(self.mask_path_list[idx]))
        mask = torch.LongTensor(np.where(mask == True, 1, 0))
        # label = self.label_idxs[idx]
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask
    

class DeepCrackDataset(Dataset):
    def __init__(self, args, data_part=None, subset_size=None):
        self.data_part = data_part
        self.augmentation_prob = 0.5
        self.args = args
        # gives list of entire path to each image along the img_dir
        if self.data_part == 'train':
            self.image_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "train/images/*.png")))
            self.mask_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "train/masks/*.png")))
        elif self.data_part == 'test':
            self.image_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "test/images/*.png")))
            self.mask_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "test/masks/*.png")))
        elif self.data_part == 'valid':
            # Use a subset of training data for validation if no separate validation set exists
            self.image_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "train/images/*.png")))
            self.mask_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "train/masks/*.png")))

        # Randomly select subset if specified
        if subset_size is not None and subset_size < len(self.image_path_list):
            import random
            # Set seed for reproducibility
            random.seed(42)
            indices = random.sample(range(len(self.image_path_list)), subset_size)
            indices.sort()  # Sort to maintain some order
            self.image_path_list = [self.image_path_list[i] for i in indices]
            self.mask_path_list = [self.mask_path_list[i] for i in indices]
            print(f"Using random subset: {subset_size} images from {data_part} set")
        
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        # TODO
        image = Image.open(self.image_path_list[idx])
        mask = Image.open(self.mask_path_list[idx])

        transformer = []
        mask_transformer = []
        if self.args.model_name == 'LMM_Net':
            image_width, image_height = 512, 512
        elif self.args.model_name == 'EfficientCrackNet':
            image_width, image_height = 512, 512  # Square to match 720x720 originals (no distortion)
            mask_width, mask_height = 512, 512

        transformer.append(transforms.Resize((image_height, image_width)))
        mask_transformer.append(transforms.Resize((mask_height, mask_width)))
        p_transform = random.random()

        
        if (self.data_part == 'train') and p_transform <= self.augmentation_prob:
            transformer = transforms.Compose(transformer)
            mask_transformer = transforms.Compose(mask_transformer)
            image = transformer(image)
            # mask = transformer(mask)
            mask = mask_transformer(mask)

            transformer = []
            # mask_transformer = transformer
            transformer.append(transforms.RandomInvert(p=0.05))
            transformer.append(transforms.ColorJitter(brightness=0.35,contrast=0.22,hue=0.02))
            transformer = transforms.Compose(transformer)
            image = transformer(image)

            if random.random() < self.augmentation_prob:
                image = F.hflip(image)
                mask = F.hflip(mask)

            if random.random() > self.augmentation_prob:
                image = F.vflip(image)
                mask = F.vflip(mask)
                
            transformer = []

        transformer.append(transforms.ToTensor())
        transformer = transforms.Compose(transformer)

        # print('Mid Image size:', image.size)
        image = transformer(image)
        # print('Post second resized Image size:', image.shape)
        mask = transformer(mask)

        
        # print(mask.shape)
        if mask.shape[0] > 1:
            transformer = transforms.Grayscale(num_output_channels=1)
            mask = transformer(mask)

        mask[mask < 0.5] = 0
        mask[mask > 0.5] = 1

        return image, mask
    

def save_training_plot_only(epoch_train_loss, epochs, args):
    plt.figure(figsize=(14, 5))

  # Accuracy plot
    plt.subplot(1, 2, 1)
    train_loss_plot, = plt.plot(epochs, epoch_train_loss, 'r')
    plt.title('Training Loss')
    plt.legend([train_loss_plot], ['Training Loss'])
    plt.savefig(f'./results/plots/{args.model_name}/run_{args.run_num}/loss_plots.jpg')


def save_plots(epoch_train_loss, epoch_valid_loss, epochs, args):
    plt.figure(figsize=(14, 5))

  # Accuracy plot
    plt.subplot(1, 2, 1)
    train_loss_plot, = plt.plot(epochs, epoch_train_loss, 'r')
    val_loss_plot, = plt.plot(epochs, epoch_valid_loss, 'b')
    plt.title('Training and Validation Loss')
    plt.legend([train_loss_plot, val_loss_plot], ['Training Loss', 'Validation Loss'])
    plt.savefig(f'./results/plots/{args.model_name}/run_{args.run_num}/loss_plots.jpg')


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

    
def save_checkpoint(save_path, model, loss, val_used=None):
    if save_path == None:
        return

    loss_txt = 'val_loss' if val_used else 'train_loss'
    state_dict = {'model_state_dict': model.state_dict(),
                  loss_txt: loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')