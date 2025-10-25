"""
Configuration file for Generic Image Classification
==================================================
Central configuration management with automatic dataset detection.
"""

from pathlib import Path
import torch
import pandas as pd

class Config:
    """Generic configuration class for any image classification dataset"""
    
    def __init__(self):
        # Paths to be configured
        self.data_dir = Path("D:/Projects/DataScience/Portfolio/Image-Classification/Workspace/Image-Classification/data")

        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.train_csv = self.data_dir / "Training_set.csv"
        self.test_csv = self.data_dir / "Testing_set.csv"
        
        # Create CSV files if they don't exist. call _create_csv_files()

        # Auto-detect dataset characteristics
        self._analyze_dataset()
        
        # Output directories
        self.output_dir = Path("./output")
        self.models_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"
        
        # Create directories
        for directory in [self.output_dir, self.models_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model configuration (automatically optimized based on dataset)
        self.model_name = 'efficientnet_b0'  # Will be auto-optimized
        # self.num_classes = AUTOMATICALLY DETECTED!
        self.img_size = 224
        self.pretrained = True
        
        # Training configuration
        self.batch_size = 32
        self.num_epochs = 50
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.momentum = 0.9
        
        # Data loading
        self.num_workers = 4
        self.pin_memory = True if self.device.type == 'cuda' else False
        
        # Validation and testing
        self.val_split = 0.2
        self.test_batch_size = 64
        
        # Early stopping and checkpointing
        self.patience = 10
        
        # Model saving configuration
        self.save_best_model = True          # Keep this True
        self.save_checkpoint_every = 0       # Set to 0 to disable regular checkpoints
        self.max_checkpoints_to_keep = 1     # Only keep the best model
        self.cleanup_old_checkpoints = True  # Auto-cleanup old files
        
        # Data augmentation
        self.use_augmentation = True
        self.rotation_degrees = 30
        self.horizontal_flip_prob = 0.5
        self.vertical_flip_prob = 0.2
        self.color_jitter = 0.2
        
        # Mixed precision training
        self.use_mixed_precision = True if self.device.type == 'cuda' else False
        
        # Random seed
        self.seed = 42
        
        # Cache settings
        self.cache_features = True
        self.feature_cache_file = 'pytorch_features_cache.pt'
    
    
    # Create Training and Testing CSV, if they don't exist and the dataset is structured with folders with class names containing images  
    def _create_csv_files(self):
        # --- Create Training CSV ---
        if not self.train_csv.exists():
            print(f"Training CSV not found. Generating from {self.train_dir}...")
            if self.train_dir.exists():
                train_data = []
                # The class name is the name of the subdirectory
                for class_dir in sorted(self.train_dir.iterdir()):
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        for img_file in class_dir.glob('*.*'):
                            # Check for common image extensions
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                                train_data.append({'filename': f"{class_name}/{img_file.name}", 'label': class_name})
                
                if train_data:
                    train_df = pd.DataFrame(train_data)
                    train_df.to_csv(self.train_csv, index=False)
                    print(f"Successfully created {self.train_csv} with {len(train_df)} entries.")
                else:
                    print(f"No training images found or directory structure is incorrect.")
                    print("   Expected structure: data/train/<ClassName>/<image_file>")
            else:
                print(f"Training directory not found: {self.train_dir}")

        # --- Create Test CSV ---
        if not self.test_csv.exists():
            print(f"Test CSV not found. Generating from {self.test_dir}...")
            if self.test_dir.exists():
                test_files = [f.name for f in self.test_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']]

                if test_files:
                    test_df = pd.DataFrame({'filename': sorted(test_files)})
                    test_df.to_csv(self.test_csv, index=False)
                    print(f"Successfully created {self.test_csv} with {len(test_df)} entries.")
                else:
                    print("No test images found.")
            else:
                print(f"Test directory not found: {self.test_dir}")
      

    def _analyze_dataset(self):
        """Automatically analyze dataset and configure parameters"""
        try:
            import pandas as pd
            
            # Read training CSV to detect classes
            df = pd.read_csv(self.train_csv)
            unique_classes = sorted(df['label'].unique())
            
            # Automatically set number of classes
            self.num_classes = len(unique_classes)
            self.class_names = unique_classes
            
            # Analyze dataset size and balance
            total_images = len(df)
            class_counts = df['label'].value_counts()
            min_samples = class_counts.min()
            max_samples = class_counts.max()
            balance_ratio = min_samples / max_samples
            
            # Auto-optimize configuration based on dataset characteristics
            self._auto_optimize_config(total_images, self.num_classes, balance_ratio)
            
            print(f"\n AUTO-DETECTED DATASET CHARACTERISTICS:")
            print(f" Total Images: {total_images:,}")
            print(f" Number of Classes: {self.num_classes}")
            print(f" Dataset Balance Ratio: {balance_ratio:.3f}")
            print(f" Auto-optimized configuration applied!")
            
        except FileNotFoundError:
            print(f"Warning: Training CSV not found at {self.train_csv}")
            print("Please ensure the data directories are correct and CSVs can be generated.")
            self.num_classes = 75  # Fallback default
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        except Exception as e:
            print(f" Warning: Could not auto-analyze dataset: {e}")
            print(" Using default configuration")
            self.num_classes = 75
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
    
    def _auto_optimize_config(self, total_images, num_classes, balance_ratio):
        """Auto-optimize configuration based on dataset characteristics"""
        
        # Optimize model selection
        if num_classes <= 10 and total_images < 5000:
            self.model_name = 'efficientnet_b0'
            self.batch_size = 64
            self.num_epochs = 100
        elif num_classes <= 50 and total_images < 20000:
            self.model_name = 'efficientnet_b0'
            self.batch_size = 32
            self.num_epochs = 75
        elif num_classes <= 100:
            self.model_name = 'efficientnet_b1'
            self.batch_size = 32
            self.num_epochs = 50
        else:
            self.model_name = 'efficientnet_b2'
            self.batch_size = 16
            self.num_epochs = 50
            self.learning_rate = 0.0005
        
        # Adjust for imbalanced datasets
        if balance_ratio < 0.5:
            self.num_epochs = int(self.num_epochs * 1.2)
            self.learning_rate *= 0.8
            print(f" Detected imbalanced dataset - adjusted training parameters")
        
        # More augmentation for smaller datasets
        if total_images < 10000:
            self.use_augmentation = True
            self.rotation_degrees = 45
            self.color_jitter = 0.3
            print(f" Small dataset detected - enhanced data augmentation")
    
        
    def get_model_save_path(self, model_name=None):
        """Get the path to save the model"""
        if model_name is None:
            model_name = self.model_name
        return self.models_dir / f"best_{model_name}_butterfly_classifier.pth"
    
    def get_results_path(self, filename):
        """Get path for results files"""
        return self.results_dir / filename
    
    def get_logs_path(self, filename):
        """Get path for log files"""
        return self.logs_dir / filename
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print(" GENERIC IMAGE CLASSIFICATION CONFIGURATION ")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Classes: {self.num_classes}")
        print(f"Image size: {self.img_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Mixed precision: {self.use_mixed_precision}")
        print("="*60)
        print(f"Data augmentation: {self.use_augmentation}")
        print(f"Early stopping patience: {self.patience}")
        print("="*60)
