# PyTorch Image Classification Framework

This project provides a flexible and powerful framework for training and evaluating image classification models using PyTorch. It includes both a modular Python framework and a comprehensive Jupyter notebook implementation, specifically designed for butterfly species classification.

## Features

- **Modern Architectures**: Easily use powerful, pre-trained models like EfficientNet, ResNet, and Vision Transformers (ViT) via the `timm` library.
- **Centralized Configuration**: All hyperparameters, paths, and settings are managed in a single `config.py` file for easy tuning.
- **Complete Workflow**: Includes scripts for the entire machine learning pipeline:
    - `train_image_classifier.py`: End-to-end model training and validation.
    - `predict.py`: Generating predictions on a test set.
- **Interactive Notebook**: Complete Jupyter notebook implementation with step-by-step explanations.
- **Data Augmentation**: Built-in support for common data augmentation techniques to improve model robustness.
- **Advanced Training**: Supports mixed-precision training, learning rate scheduling, and early stopping.
- **GPU Optimization**: Optimized for NVIDIA GPUs with proper memory management and performance monitoring.
- **Checkpointing**: Automatically saves the best model and allows for resuming training from checkpoints.
- **Detailed Evaluation**: Generates classification reports, confusion matrices, and visualizations to assess model performance.

## Project Structure

```
Image-Classification/
├── data/
│   ├── train/
│   │   ├── image_train1.jpg
│   │   └── ...
│   ├── test/
│   │   ├── image_test1.jpg
│   │   └── ...
│   ├── Training_set.csv
│   └── Testing_set.csv
│
├── notebooks/
│   └── image_classification.ipynb  # Complete interactive notebook
│
├── pytorch/
│   ├── config.py                   # Main configuration file
│   ├── data_loader.py              # PyTorch Datasets and DataLoaders
│   ├── models.py                   # Model definitions (ImageClassifier)
│   ├── trainer.py                  # Core training and validation loop
│   ├── evaluator.py                # Model evaluation logic
│   ├── train_butterfly_classifier.py # Main script to start training
│   └── predict.py                  # Script to generate test predictions
│
└── output/
    ├── models/                     # Saved model checkpoints (.pth)
    ├── results/                    # Prediction CSVs and evaluation reports
    └── logs/                       # Training logs
```

## Dataset

**Source**: Kaggle ["butterfly-image-classification"](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)

The dataset contains:
- **Training set**: 12,594 images across 75 butterfly species
- **Test set**: 500 images for prediction
- **75 different butterfly species** to classify
- High-quality images with diverse poses, backgrounds, and lighting conditions

## Implementation Options

### Option 1: Jupyter Notebook (Recommended for Learning)

The `notebooks/image_classification.ipynb` provides a complete, interactive implementation with detailed explanations:

#### Notebook Contents:
1. **Setup & Installation** - Install required packages and import libraries
2. **Configuration** - Set up paths, model parameters, and training hyperparameters
3. **Data Loading & Preprocessing** - Custom dataset class with augmentation techniques
4. **Model Definition** - EfficientNet-based classifier with transfer learning
5. **Training Functions** - Optimized training and validation loops with GPU support
6. **Model Training** - Complete training loop with progress tracking
7. **Training Visualization** - Loss and accuracy curves
8. **Model Evaluation** - Detailed performance metrics and classification reports
9. **Confusion Matrix** - Visual analysis of model performance across species
10. **Test Set Predictions** - Generate predictions for Kaggle submission
11. **Prediction Analysis** - Distribution analysis of predictions

#### Key Features:
- **GPU Optimization**: Configured for NVIDIA GPUs with memory monitoring
- **Mixed Precision Training**: Faster training with reduced memory usage
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Transfer Learning**: Pre-trained EfficientNet backbone with custom classifier
- **Comprehensive Evaluation**: Classification reports, confusion matrices, and visualizations
- **Kaggle Ready**: Generates properly formatted submission files

### Option 2: Modular Python Framework

Use the structured Python modules in the `pytorch/` directory for production environments.

## Setup

### 1. Install Dependencies

It is recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

**Option A: Download from Kaggle**
```python
import kagglehub
path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")
```

**Option B: Manual Setup**
- Place training images in `data/train/`
- Place test images in `data/test/`
- Ensure `data/Training_set.csv` and `data/Testing_set.csv` are present with:
  - `Training_set.csv`: `filename` and `label` columns
  - `Testing_set.csv`: `filename` column

### 3. Hardware Requirements

**Recommended GPU Setup:**
- NVIDIA RTX 3060 or better
- 8GB+ VRAM for batch size 128
- CUDA-compatible PyTorch installation

**CPU Fallback:**
- The framework automatically falls back to CPU if GPU is unavailable
- Training will be significantly slower on CPU

## Usage

### Using the Jupyter Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook notebooks/image_classification.ipynb
   ```

2. **Run the Notebook**:
   - Execute cells sequentially
   - Monitor GPU usage and training progress
   - Adjust hyperparameters in the Configuration section
   - Generate predictions and submission files

3. **Key Configuration Options**:
   ```python
   # In the notebook Configuration section
   self.batch_size = 128      # Adjust based on GPU memory
   self.num_epochs = 30       # Training epochs
   self.learning_rate = 0.001 # Learning rate
   self.model_name = 'efficientnet_b0'  # Model architecture
   ```

### Using the Python Framework

1. **Configure the Project**:
   - Open `pytorch/config.py`
   - Set correct data paths and hyperparameters

2. **Train the Model**:
   ```bash
   cd pytorch
   python train_butterfly_classifier.py
   ```

3. **Generate Predictions**:
   ```bash
   python predict.py
   ```

## Model Performance

The framework achieves competitive results on the butterfly classification dataset:

- **Architecture**: EfficientNet-B0 with custom classifier head
- **Training Strategy**: Transfer learning with data augmentation
- **Validation Accuracy**: 85-90% (varies with hyperparameters)
- **Training Time**: ~2-3 hours on RTX 4070 Ti SUPER for 30 epochs

## Optimization Tips

### GPU Performance
- Use `batch_size=128` or higher for modern GPUs
- Enable mixed precision training for faster computation
- Set `pin_memory=True` for faster data transfer
- Monitor GPU memory usage to avoid OOM errors

### Training Improvements
- Increase `num_epochs` for better convergence
- Experiment with different EfficientNet variants (B1, B2, etc.)
- Try different learning rates and optimizers
- Implement learning rate scheduling for better results

## Output Files

The framework generates several output files:

- `output/models/best_model.pth`: Best model weights based on validation accuracy
- `output/results/submission.csv`: Test predictions in Kaggle format
- `output/results/classification_report.txt`: Detailed performance metrics
- `output/logs/training.log`: Training progress logs

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` in configuration
2. **Slow Training**: Ensure GPU is being used and `num_workers=0` on Windows
3. **File Not Found**: Check data paths in configuration
4. **Import Errors**: Ensure all requirements are installed

### Performance Monitoring

The notebook includes GPU monitoring functions:
```python
check_gpu_usage()  # Monitor GPU memory usage
```

## Contributing

Feel free to contribute improvements:
- Add new model architectures
- Implement advanced training techniques
- Improve data augmentation strategies
- Add evaluation metrics

## License

This project is open source and available under the MIT License.