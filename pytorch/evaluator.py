"""
Evaluation and Prediction Module
===============================
Model evaluation, test predictions, and results analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import top_k_accuracy_score

class Evaluator:
    """Model evaluation and prediction class"""
    
    def __init__(self, model, config, label_encoder):
        self.model = model
        self.config = config
        self.label_encoder = label_encoder
        self.device = config.device
        
    def evaluate_model(self, data_loader, dataset_name="Validation"):
        """Comprehensive model evaluation"""
        print(f"\n🔍 Evaluating model on {dataset_name} set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_filenames = []
        
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                filenames = batch['filename']
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                total_loss += loss.item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_filenames.extend(filenames)
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        top3_accuracy = top_k_accuracy_score(all_labels, np.array(all_probabilities), k=3)
        top5_accuracy = top_k_accuracy_score(all_labels, np.array(all_probabilities), k=5)
        
        print(f"📊 {dataset_name} Results:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
        print(f"   Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
        
        # Generate detailed report
        self._generate_classification_report(all_labels, all_predictions, dataset_name)
        self._plot_confusion_matrix(all_labels, all_predictions, dataset_name)
        self._analyze_misclassifications(all_labels, all_predictions, all_filenames, 
                                       all_probabilities, dataset_name)
        
        return {
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'top5_accuracy': top5_accuracy,
            'avg_loss': avg_loss,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'filenames': all_filenames
        }
    
    def predict_test_set(self, test_loader):
        """Make predictions on test set"""
        print("\n🔮 Making predictions on test set...")
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_filenames = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                images = batch['image'].to(self.device)
                filenames = batch['filename']
                
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_filenames.extend(filenames)
        
        # Convert predictions to class names
        predicted_classes = self.label_encoder.inverse_transform(all_predictions)
        
        # Calculate confidence scores
        max_probabilities = np.max(all_probabilities, axis=1)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'filename': all_filenames,
            'predicted_label': predicted_classes,
            'confidence': max_probabilities
        })
        
        # Add top-3 predictions
        top3_indices = np.argsort(all_probabilities, axis=1)[:, -3:][:, ::-1]
        for i in range(3):
            col_name = f'top_{i+1}_prediction'
            conf_name = f'top_{i+1}_confidence'
            submission_df[col_name] = self.label_encoder.inverse_transform(top3_indices[:, i])
            submission_df[conf_name] = [probs[idx] for probs, idx in zip(all_probabilities, top3_indices[:, i])]
        
        # Save predictions
        submission_path = self.config.get_results_path('test_predictions.csv')
        submission_df.to_csv(submission_path, index=False)
        
        print(f"✅ Test predictions saved to {submission_path}")
        print(f"   Total predictions: {len(submission_df)}")
        print(f"   Average confidence: {max_probabilities.mean():.3f}")
        print(f"   Confidence std: {max_probabilities.std():.3f}")
        
        return submission_df
    
    def _generate_classification_report(self, y_true, y_pred, dataset_name):
        """Generate and save classification report"""
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        
        print(f"\n📋 Classification Report ({dataset_name}):")
        print(report)
        
        # Save report
        report_path = self.config.get_results_path(f'classification_report_{dataset_name.lower()}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - {dataset_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        
        print(f"📄 Classification report saved to {report_path}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, dataset_name):
        """Plot and save confusion matrix"""
        class_names = self.label_encoder.classes_
        cm = confusion_matrix(y_true, y_pred)
        
        # For large number of classes, show only top predicted classes
        if len(class_names) > 20:
            # Get most frequent predictions
            unique_preds, counts = np.unique(y_pred, return_counts=True)
            top_classes_idx = unique_preds[np.argsort(counts)[-20:]]
            
            # Filter confusion matrix
            cm_filtered = cm[np.ix_(top_classes_idx, top_classes_idx)]
            class_names_filtered = [class_names[i] for i in top_classes_idx]
            
            plt.figure(figsize=(16, 12))
            sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names_filtered,
                       yticklabels=class_names_filtered)
            plt.title(f'Confusion Matrix - {dataset_name} (Top 20 Predicted Classes)')
        else:
            plt.figure(figsize=(20, 16))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - {dataset_name}')
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        cm_path = self.config.get_results_path(f'confusion_matrix_{dataset_name.lower()}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Confusion matrix saved to {cm_path}")
    
    def _analyze_misclassifications(self, y_true, y_pred, filenames, probabilities, dataset_name):
        """Analyze misclassified samples"""
        class_names = self.label_encoder.classes_
        
        # Find misclassified samples
        misclassified_mask = np.array(y_true) != np.array(y_pred)
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            print("🎉 Perfect classification - no misclassifications!")
            return
        
        print(f"\n🔍 Misclassification Analysis ({dataset_name}):")
        print(f"   Total misclassified: {len(misclassified_indices)}")
        print(f"   Misclassification rate: {len(misclassified_indices)/len(y_true)*100:.2f}%")
        
        # Analyze worst misclassifications (lowest confidence on wrong predictions)
        misclass_data = []
        for idx in misclassified_indices:
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            confidence = probabilities[idx][y_pred[idx]]
            filename = filenames[idx]
            
            misclass_data.append({
                'filename': filename,
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': confidence
            })
        
        # Sort by confidence (most confident wrong predictions first)
        misclass_df = pd.DataFrame(misclass_data)
        misclass_df = misclass_df.sort_values('confidence', ascending=False)
        
        # Save misclassification analysis
        misclass_path = self.config.get_results_path(f'misclassifications_{dataset_name.lower()}.csv')
        misclass_df.to_csv(misclass_path, index=False)
        
        print(f"📋 Misclassification analysis saved to {misclass_path}")
        print(f"   Top 5 most confident wrong predictions:")
        for _, row in misclass_df.head().iterrows():
            print(f"     {row['filename']}: {row['true_class']} → {row['predicted_class']} ({row['confidence']:.3f})")
    
    def analyze_class_performance(self, y_true, y_pred, probabilities):
        """Analyze per-class performance"""
        class_names = self.label_encoder.classes_
        
        # Calculate per-class accuracy
        class_accuracies = []
        class_counts = []
        
        for i, class_name in enumerate(class_names):
            class_mask = np.array(y_true) == i
            class_count = np.sum(class_mask)
            
            if class_count > 0:
                class_correct = np.sum((np.array(y_pred)[class_mask]) == i)
                class_accuracy = class_correct / class_count
            else:
                class_accuracy = 0.0
            
            class_accuracies.append(class_accuracy)
            class_counts.append(class_count)
        
        # Create performance DataFrame
        performance_df = pd.DataFrame({
            'class_name': class_names,
            'sample_count': class_counts,
            'accuracy': class_accuracies
        })
        
        performance_df = performance_df.sort_values('accuracy', ascending=False)
        
        # Save performance analysis
        perf_path = self.config.get_results_path('class_performance.csv')
        performance_df.to_csv(perf_path, index=False)
        
        print(f"\n📈 Class Performance Analysis:")
        print(f"   Best performing class: {performance_df.iloc[0]['class_name']} ({performance_df.iloc[0]['accuracy']:.3f})")
        print(f"   Worst performing class: {performance_df.iloc[-1]['class_name']} ({performance_df.iloc[-1]['accuracy']:.3f})")
        print(f"   Average class accuracy: {np.mean(class_accuracies):.3f}")
        print(f"   Performance saved to {perf_path}")
        
        return performance_df

def load_and_evaluate(checkpoint_path, test_loader, config, label_encoder):
    """Load model and evaluate on test set"""
    print(f"📥 Loading model from {checkpoint_path}")
    
    # Load model
    from models import ButterflyClassifier
    model = ButterflyClassifier(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=False  # Don't load pretrained weights
    ).to(config.device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded successfully!")
    print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 'Unknown')}")
    
    # Create evaluator and make predictions
    evaluator = Evaluator(model, config, label_encoder)
    predictions = evaluator.predict_test_set(test_loader)
    
    return predictions, evaluator
