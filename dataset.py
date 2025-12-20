
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class RetinaMultiLabelDataset(Dataset):
    """
    Enhanced dataset with proper augmentation for retinal images
    """
    def __init__(self, csv_file, image_dir, transform=None, mode='train'):
        """
        Args:
            csv_file: Path to CSV with labels
            image_dir: Directory containing images
            transform: Torchvision transforms
            mode: 'train' | 'val' | 'test'
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row.iloc[0]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Get labels (handle test set without labels)
        if len(row) > 1:
            labels = torch.tensor(row[1:].values.astype("float32"))
        else:
            labels = torch.tensor([]) # Empty for test set

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, labels, img_name


def get_transforms(img_size=256, mode='train'):
    """
    Get transforms for different modes

    Args:
        img_size: Target image size
        mode: 'train' | 'val' | 'test'

    Returns:
        transforms.Compose object
    """

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomApply([
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])

    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_tta_transforms(img_size=256):
    """
    Get Test Time Augmentation transforms

    Returns list of transforms for TTA
    """
    base_transform = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    tta_transforms = [
        # Original
        transforms.Compose(base_transform),

        # Horizontal Flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        # Vertical Flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        # Both Flips
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        # Slight Rotation
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]

    return tta_transforms