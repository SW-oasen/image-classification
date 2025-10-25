"""
Main Training Script for Butterfly Classification with PyTorch
============================================================
Complete training pipeline with PyTorch deep learning.
"""

import torch
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_loader import DataManager
from models import create_model
from trainer import Trainer
from evaluator import Evaluator

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """Main training function"""
    print("=" * 43)
    print("  BUTTERFLY CLASSIFICATION WITH PYTORCH")
    print("=" * 43)
    
    # Initialize configuration
    config = Config()
    config.print_config()
    
    # Set random seeds
    set_seed(config.seed)
    print(f" Random seed set to {config.seed}")
    
    try:
        # Initialize data manager
        print("\n" + "="*60)
        print("DATA LOADING")
        print("="*60)
        
        data_manager = DataManager(config)
        data_dict = data_manager.load_data()
        
        train_loader = data_dict['train_loader']
        val_loader = data_dict['val_loader']
        test_loader = data_dict['test_loader']
        label_encoder = data_dict['label_encoder']
        num_classes = data_dict['num_classes']
        
        # Update config with actual number of classes
        config.num_classes = num_classes
        
        # Create model
        print("\n" + "="*60)
        print(" MODEL CREATION")
        print("="*60)
        
        model = create_model(config)
        
        # Check for existing checkpoint
        checkpoint_path = config.get_model_save_path()
        if checkpoint_path.exists():
            print(f"Found existing model at {checkpoint_path}")
            response = input("Do you want to resume training? (y/n): ").lower().strip()
            
            if response == 'y':
                from trainer import resume_training
                trainer = resume_training(model, config, train_loader, val_loader, 
                                        label_encoder, checkpoint_path)
            else:
                trainer = Trainer(model, train_loader, val_loader, config, label_encoder)
        else:
            trainer = Trainer(model, train_loader, val_loader, config, label_encoder)
        
        # Start training
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        
        training_results = trainer.train()
        
        # Load best model for evaluation
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        # Load best model
        best_model_path = config.get_model_save_path()
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f" Loaded best model with validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        # Create evaluator
        evaluator = Evaluator(model, config, label_encoder)
        
        # Evaluate on validation set
        val_results = evaluator.evaluate_model(val_loader, "Validation")
        
        # Make test predictions
        test_predictions = evaluator.predict_test_set(test_loader)
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"   Final Results:")
        print(f"   Best Validation Accuracy: {training_results['best_val_acc']:.2f}%")
        print(f"   Final Validation Accuracy: {val_results['accuracy']*100:.2f}%")
        print(f"   Top-3 Accuracy: {val_results['top3_accuracy']*100:.2f}%")
        print(f"   Top-5 Accuracy: {val_results['top5_accuracy']*100:.2f}%")
        print(f"   Total Training Time: {training_results['total_time']/3600:.2f} hours")
        print(f"   Test Predictions: {len(test_predictions)} samples")
        
        # Save final summary
        summary_path = config.get_results_path('training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Image Classification Training Summary\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Model: {config.model_name}\n")
            f.write(f"Image Size: {config.img_size}\n")
            f.write(f"Batch Size: {config.batch_size}\n")
            f.write(f"Learning Rate: {config.learning_rate}\n")
            f.write(f"Number of Classes: {config.num_classes}\n")
            f.write(f"Device: {config.device}\n\n")
            f.write("Results:\n")
            f.write(f"  Best Validation Accuracy: {training_results['best_val_acc']:.2f}%\n")
            f.write(f"  Final Validation Accuracy: {val_results['accuracy']*100:.2f}%\n")
            f.write(f"  Top-3 Accuracy: {val_results['top3_accuracy']*100:.2f}%\n")
            f.write(f"  Top-5 Accuracy: {val_results['top5_accuracy']*100:.2f}%\n")
            f.write(f"  Total Training Time: {training_results['total_time']/3600:.2f} hours\n")
            f.write(f"  Test Predictions: {len(test_predictions)} samples\n")
        
        print(f" Training summary saved to {summary_path}")
        
        return {
            'model': model,
            'training_results': training_results,
            'val_results': val_results,
            'test_predictions': test_predictions,
            'config': config
        }
        
    except Exception as e:
        print(f"\n Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    
    if results is not None:
        print("\n All done! Check the output directory for results.")
    else:
        print("\n Training failed. Please check the error messages above.")
