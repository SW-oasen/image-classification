"""
Data Loading and Preprocessing for Butterfly Classification
==========================================================
PyTorch dataset classes and data loaders with augmentation.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import config
import warnings
warnings.filterwarnings('ignore')

class ButterflyDataset(Dataset):
    """Custom dataset for butterfly images"""
    
    def __init__(self, dataframe, label_encoder=None, transform=None, is_test=False):
        """
        Args:
            dataframe: pandas DataFrame with 'image_path' and 'label' columns
            label_encoder: sklearn LabelEncoder for labels
            transform: torchvision transforms
            is_test: whether this is test data (no labels)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        self.label_encoder = label_encoder
        
        if not is_test and label_encoder is not None:
            # Encode labels for training/validation
            self.labels = label_encoder.transform(dataframe['label'])
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = self.dataframe.iloc[idx]['image_path']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return {
                'image': image,
                'filename': self.dataframe.iloc[idx]['filename']
            }
        else:
            label = self.labels[idx]
            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.long),
                'filename': self.dataframe.iloc[idx]['filename']
            }

class DataManager:
    """Manages data loading and preprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        
        # Define transforms
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        self.test_transform = self._get_test_transforms()
        
    def _get_train_transforms(self):
        """Get training data transforms with augmentation"""
        transforms_list = [
            transforms.Resize((self.config.img_size, self.config.img_size)),
        ]
        
        if self.config.use_augmentation:
            transforms_list.extend([
                transforms.RandomRotation(self.config.rotation_degrees),
                transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob),
                transforms.RandomVerticalFlip(p=self.config.vertical_flip_prob),
                transforms.ColorJitter(
                    brightness=self.config.color_jitter,
                    contrast=self.config.color_jitter,
                    saturation=self.config.color_jitter,
                    hue=self.config.color_jitter/2
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10
                ),
            ])
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet standards
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(transforms_list)
    
    def _get_val_transforms(self):
        """Get validation data transforms (no augmentation)"""
        return transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_test_transforms(self):
        """Get test data transforms (same as validation)"""
        return self._get_val_transforms()
    
    def load_data(self):
        """Load and prepare all datasets"""
        print("📊 Loading butterfly dataset...")
        
        # Load CSV files
        train_df = pd.read_csv(self.config.train_csv)
        test_df = pd.read_csv(self.config.test_csv)
        #train_df = pd.read_csv(config.train_csv)
        #test_df = pd.read_csv(config.test_csv)

        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Number of classes: {train_df['label'].nunique()}")
        
        # Add full image paths
        train_df['image_path'] = train_df['filename'].apply(
            lambda x: self.config.train_dir / x
        )
        test_df['image_path'] = test_df['filename'].apply(
            lambda x: self.config.test_dir / x
        )
        
        # Check file existence
        print("🔍 Checking file existence...")
        train_exists = train_df['image_path'].apply(lambda x: x.exists())
        test_exists = test_df['image_path'].apply(lambda x: x.exists())
        
        missing_train = (~train_exists).sum()
        missing_test = (~test_exists).sum()
        
        if missing_train > 0:
            print(f"⚠️ Warning: {missing_train} training images not found")
            train_df = train_df[train_exists].reset_index(drop=True)
        
        if missing_test > 0:
            print(f"⚠️ Warning: {missing_test} test images not found")
            test_df = test_df[test_exists].reset_index(drop=True)
        
        print(f"✅ Valid files - Training: {len(train_df)}, Test: {len(test_df)}")
        
        # Encode labels
        self.label_encoder.fit(train_df['label'])
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"📋 Classes: {self.num_classes}")
        print(f"📋 Sample classes: {list(self.label_encoder.classes_[:10])}")
        
        # Split training data into train/validation
        train_data, val_data = train_test_split(
            train_df,
            test_size=self.config.val_split,
            stratify=train_df['label'],
            random_state=self.config.seed
        )
        
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        
        print(f"📊 Data splits:")
        print(f"   Training: {len(train_data)} samples")
        print(f"   Validation: {len(val_data)} samples")
        print(f"   Test: {len(test_df)} samples")
        
        # Create datasets
        train_dataset = ButterflyDataset(
            train_data,
            label_encoder=self.label_encoder,
            transform=self.train_transform,
            is_test=False
        )
        
        val_dataset = ButterflyDataset(
            val_data,
            label_encoder=self.label_encoder,
            transform=self.val_transform,
            is_test=False
        )
        
        test_dataset = ButterflyDataset(
            test_df,
            label_encoder=None,
            transform=self.test_transform,
            is_test=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        print("✅ Data loaders created successfully!")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'label_encoder': self.label_encoder,
            'num_classes': self.num_classes
        }
    
    def get_class_weights(self, train_df):
        """Calculate class weights for imbalanced dataset"""
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label']
        )
        
        # Convert to tensor
        class_weights = torch.FloatTensor(class_weights).to(self.config.device)
        return class_weights
    
    def analyze_class_distribution(self, train_df):
        """Analyze and print class distribution"""
        class_counts = train_df['label'].value_counts()
        
        print("\n📈 Class Distribution Analysis:")
        print(f"   Most frequent class: {class_counts.index[0]} ({class_counts.iloc[0]} samples)")
        print(f"   Least frequent class: {class_counts.index[-1]} ({class_counts.iloc[-1]} samples)")
        print(f"   Average samples per class: {class_counts.mean():.1f}")
        print(f"   Standard deviation: {class_counts.std():.1f}")
        print(f"   Imbalance ratio: {class_counts.max() / class_counts.min():.2f}")
        
        return class_counts
