from torch.utils.data import DataLoader, random_split
import os
import torch
from torchvision import datasets, transforms
import lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms

from datasets import AlbumDataset
from augmentations import train_set_transforms, test_set_transforms

class CIFARDataModule(pl.LightningDataModule):

  def setup(self, stage):
    SEED = 8
    BATCH_SIZE = 512
     
    train = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    val = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

    # Get the dictionary with augmentations
    train_transforms = A.Compose(train_set_transforms.values())
    test_transforms = A.Compose(test_set_transforms.values())

    # Create custom Dataset to support albumentations library
    self.train_set = AlbumDataset('./data', train=True, download=True, transform=train_transforms)
    self.test_set = AlbumDataset('./data', train=False, download=True, transform=test_transforms)

    
  def train_dataloader(self):
    return DataLoader(self.train_set, batch_size=512)

  def val_dataloader(self):
    return DataLoader(self.test_set, batch_size=512)
