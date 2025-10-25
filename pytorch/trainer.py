"""
Training Module for Butterfly Classification
===========================================
Training loop with mixed precision, early stopping, and comprehensive logging.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

class Trainer:
    """Main training class"""
    
    def __init__(self, model, train_loader, val_loader, config, label_encoder):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.label_encoder = label_encoder
        self.device = config.device
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_loss_function()
        self._setup_scheduler()
        self._setup_mixed_precision()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        
        # Early stopping
        self.patience_counter = 0
        
        print(" Trainer initialized successfully!")
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        print(f" Optimizer: AdamW (lr={self.config.learning_rate}, wd={self.config.weight_decay})")
    
    def _setup_loss_function(self):
        """Setup loss function"""
        # You could add class weights here if needed
        self.criterion = nn.CrossEntropyLoss()
        print(" Loss function: CrossEntropyLoss")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        print(" Scheduler: ReduceLROnPlateau")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
            print(" Mixed precision training enabled")
        else:
            self.scaler = None
            print(" Standard precision training")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.use_mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.config.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print(" STARTING BUTTERFLY CLASSIFICATION TRAINING")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            print(f"\n Epoch {epoch+1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.learning_rates.append(current_lr)
            
            # Print epoch results
            print(f" Results:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.2e}")
            
            # Save best model (keep this)
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                if self.config.save_best_model:
                    save_path = self.config.get_model_save_path()
                    self._save_checkpoint(save_path, is_best=True)
                    print(f" New best model saved! Validation accuracy: {val_acc:.2f}%")
    
            # Only save regular checkpoints if configured (modify this section)
            if (self.config.save_checkpoint_every > 0 and 
                (epoch + 1) % self.config.save_checkpoint_every == 0):
                checkpoint_path = self.config.models_dir / f"checkpoint_epoch_{epoch+1}.pth"
                self._save_checkpoint(checkpoint_path, is_best=False)
                
                # Cleanup old checkpoints if enabled
                if self.config.cleanup_old_checkpoints:
                    self._cleanup_old_checkpoints()
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\n Early stopping triggered after {epoch+1} epochs")
                print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
                break
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save final results
        self._save_training_results()
        self._plot_training_curves()
        
        return {
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'total_time': total_time
        }
    
    def _save_checkpoint(self, filepath, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'config': self.config,
            'label_encoder': self.label_encoder
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def _save_training_results(self):
        """Save training results to CSV"""
        results_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accs,
            'val_acc': self.val_accs,
            'learning_rate': self.learning_rates
        })
        
        results_path = self.config.get_results_path('training_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f" Training results saved to {results_path}")
    
    def _plot_training_curves(self):
        """Plot and save training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accs, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(epochs, self.learning_rates, 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Best accuracy line
        ax4.axhline(y=self.best_val_acc, color='r', linestyle='--', 
                   label=f'Best Val Acc: {self.best_val_acc:.2f}%')
        ax4.plot(epochs, self.val_accs, 'r-', alpha=0.7)
        ax4.set_title('Best Validation Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.config.get_results_path('training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Training curves saved to {plot_path}")

def resume_training(model, config, train_loader, val_loader, label_encoder, checkpoint_path):
    """Resume training from checkpoint"""
    print(f" Resuming training from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, label_encoder)
    
    # Load training state
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    trainer.best_val_acc = checkpoint['best_val_acc']
    trainer.train_losses = checkpoint['train_losses']
    trainer.val_losses = checkpoint['val_losses']
    trainer.train_accs = checkpoint['train_accs']
    trainer.val_accs = checkpoint['val_accs']
    
    if 'scaler_state_dict' in checkpoint and trainer.scaler is not None:
        trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Adjust epochs
    completed_epochs = checkpoint['epoch']
    config.num_epochs = max(config.num_epochs, completed_epochs + 10)
    
    print(f"   Resuming from epoch {completed_epochs}")
    print(f"   Best validation accuracy so far: {trainer.best_val_acc:.2f}%")
    
    return trainer
