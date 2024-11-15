from dpmodel import *
from model_builder import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import List, Dict, Tuple
import albumentations as A
from PIL import Image
import time

category_to_keypoints = {
        'short sleeve top': 25,
        'short sleeve outwear': 31,
        'long sleeve top': 33,
        'long sleeve outwear': 39,
        'vest': 15,
        'sling': 15,
        'shorts': 10,
        'trousers': 14,
        'skirt': 8,
        'short sleeve dress': 29,
        'long sleeve dress': 37,
        'vest dress': 19,
        'sling dress': 19
    }


if __name__ == '__main__':
    img_dir='train/image'
    anno_dir='train/annos'
    dataset_dir = 'train/datasets'
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
    img_dir='train/image',
    anno_dir='train/annos',
    dataset_dir = 'train/datasets',
    #shuffle = True,
    category_to_keypoints = category_to_keypoints,
    sample = 5000,
    batch_size=32,  # Adjust based on your hardware
    num_workers=4,   # Adjust based on your hardware
    #timeout=300,

    )
    train_model(train_loader = train_loader, val_loader = val_loader,category_to_keypoints = category_to_keypoints,  num_epochs = 1)
    

