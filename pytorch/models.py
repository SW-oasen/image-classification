"""
Model Definitions for Image Classification
==============================================
PyTorch model architectures with pre-trained backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

class ImageClassifier(nn.Module):
    """Image classifier with various backbone options"""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=75, pretrained=True, dropout_rate=0.5):
        super(ImageClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name.startswith('efficientnet'):
            self.backbone = self._create_efficientnet(model_name, pretrained)
        elif model_name.startswith('resnet'):
            self.backbone = self._create_resnet(model_name, pretrained)
        elif model_name.startswith('vit'):
            self.backbone = self._create_vit(model_name, pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _create_efficientnet(self, model_name, pretrained):
        """Create EfficientNet backbone"""
        if pretrained:
            model = timm.create_model(model_name, pretrained=True)
        else:
            model = timm.create_model(model_name, pretrained=False)
        
        # Remove the classifier
        if hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        elif hasattr(model, 'head'):
            model.head = nn.Identity()
        
        return model
    
    def _create_resnet(self, model_name, pretrained):
        """Create ResNet backbone"""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Remove the classifier
        model.fc = nn.Identity()
        return model
    
    def _create_vit(self, model_name, pretrained):
        """Create Vision Transformer backbone"""
        if pretrained:
            model = timm.create_model(model_name, pretrained=True)
        else:
            model = timm.create_model(model_name, pretrained=False)
        
        # Remove the classifier
        if hasattr(model, 'head'):
            model.head = nn.Identity()
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        
        return model
    
    def _get_feature_dim(self):
        """Get the feature dimension of the backbone"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            return features.shape[1]
    
    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        
        # Global average pooling if needed
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features without classification"""
        with torch.no_grad():
            features = self.backbone(x)
            if len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            return features

class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, models_list, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)
        
        if weights is None:
            self.weights = [1.0 / len(models_list)] * len(models_list)
        else:
            self.weights = weights
        
        # Ensure weights sum to 1
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
    
    def forward(self, x):
        """Forward pass through ensemble"""
        outputs = []
        
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                outputs.append(output)
        
        # Weighted average
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += self.weights[i] * output
        
        return ensemble_output

def create_model(config):
    """Factory function to create models"""
    print(f"🏗️ Creating {config.model_name} model...")
    
    model = ImageClassifier(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained
    )
    
    model = model.to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model

def load_pretrained_model(model, checkpoint_path, device):
    """Load pretrained model weights"""
    print(f"📥 Loading model from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model weights loaded successfully!")
            
            if 'epoch' in checkpoint:
                print(f"   Checkpoint from epoch: {checkpoint['epoch']}")
            if 'best_val_acc' in checkpoint:
                print(f"   Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
                
        else:
            model.load_state_dict(checkpoint)
            print("✅ Model weights loaded successfully!")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e
    
    return model

def save_model(model, optimizer, epoch, val_acc, loss, filepath, best_acc=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'loss': loss,
        'best_val_acc': best_acc if best_acc is not None else val_acc
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Model saved to {filepath}")

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }
