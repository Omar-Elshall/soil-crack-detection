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
        self.augmentation_prob = 0.7
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

        self.color_jitter = transforms.ColorJitter(
            brightness=0.35, contrast=0.22, saturation=0.3, hue=0.02
        )

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_path_list[idx]).convert('RGB')
        mask = Image.open(self.mask_path_list[idx])

        # Always resize to 512x512
        image = F.resize(image, (512, 512))
        mask = F.resize(mask, (512, 512), interpolation=transforms.InterpolationMode.NEAREST)

        if self.data_part == 'train' and random.random() <= self.augmentation_prob:
            # --- Spatial transforms: applied identically to image and mask ---

            # Random rotation ±30°
            if random.random() < 0.5:
                angle = random.uniform(-30, 30)
                image = F.rotate(image, angle, fill=0)
                mask = F.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST, fill=0)

            # Random zoom (resized crop)
            if random.random() < 0.4:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.7, 1.0), ratio=(0.9, 1.1)
                )
                image = F.resized_crop(image, i, j, h, w, (512, 512))
                mask = F.resized_crop(mask, i, j, h, w, (512, 512),
                                      interpolation=transforms.InterpolationMode.NEAREST)

            # Horizontal flip
            if random.random() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            # Vertical flip
            if random.random() < 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

            # --- Photometric transforms: image only ---

            # Color jitter (brightness, contrast, saturation, hue)
            image = self.color_jitter(image)

            # Random gamma shift (exposure)
            if random.random() < 0.4:
                gamma = random.uniform(0.6, 1.5)
                image = F.adjust_gamma(image, gamma)

            # Gaussian blur (simulates drone camera focus variation)
            if random.random() < 0.3:
                ks = random.choice([3, 5])
                image = F.gaussian_blur(image, kernel_size=ks)

            # Rare invert
            if random.random() < 0.05:
                image = F.invert(image)

        # Convert to tensor
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)

        if mask.shape[0] > 1:
            mask = transforms.Grayscale(num_output_channels=1)(mask)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

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