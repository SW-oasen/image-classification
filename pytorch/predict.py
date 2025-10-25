"""
Test Prediction Generator
========================
Generate predictions for test images using the trained image classification model.

This script:
1. Loads the best trained model
2. Processes all test images
3. Generates predictions with confidence scores
4. Saves results to CSV file
5. Shows prediction statistics

Usage:
    python predict_simple.py

Output:
    - test_predictions.csv with filename, predicted_class, confidence
    - Console output with statistics and sample predictions
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')


def generate_test_predictions():
    """
    Generate predictions for all test images using the trained model.
    
    Returns:
        pandas.DataFrame: Results with filename, predicted_class, confidence
        None: If prediction generation fails
    """
    
    print("🚀 IMAGE CLASSIFICATION - TEST PREDICTIONS")
    print("=" * 55)
    
    # Use config to get dataset-specific paths
    from config import Config
    config = Config()
    
    # File paths using dataset-specific directories
    model_file = config.models_dir / f"best_{config.model_name}_classifier.pth"
    test_csv_file = config.test_csv
    test_images_dir = config.test_dir
    output_file = config.results_dir / "test_predictions.csv"
    
    # Validate required files exist
    paths_to_check = [
        (model_file, "Trained model"),
        (test_csv_file, "Test CSV file"),
        (test_images_dir, "Test images directory")
    ]
    
    print("📁 Checking required files...")
    for file_path, description in paths_to_check:
        abs_path = file_path.resolve()  # Get absolute path
        if not file_path.exists():
            print(f"❌ {description} not found: {abs_path}")
            return None
        print(f"✅ {description}: Found at {abs_path}")
    
    # Load test file list
    test_df = pd.read_csv(test_csv_file)
    print(f"📊 Found {len(test_df)} test images to process")
    
    # Load class names from training data
    train_csv_file = config.train_csv
    if train_csv_file.exists():
        train_df = pd.read_csv(train_csv_file)
        class_names = sorted(train_df['label'].unique())
        print(f"🏷️ Loaded {len(class_names)} class names")
    else:
        print("⚠️ Warning: Using fallback class names")
        class_names = [f"class_{i}" for i in range(config.num_classes)]
        
    # Setup device and image preprocessing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Using device: {device}")
    
    # Image preprocessing pipeline (same as training)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet standards
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load the trained model
    print("🤖 Loading trained model...")
    try:
        # Import the same model architecture used during training
        from models import ImageClassifier
        
        # Create model with same architecture as training
        model = ImageClassifier(
            model_name='efficientnet_b0',
            num_classes=len(class_names),
            pretrained=False
        )
        
        # Load trained weights - the file contains full checkpoint, extract model_state_dict
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        
        # Check if it's a full checkpoint or just model weights
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print("✅ Loaded from checkpoint format")
        else:
            model_state = checkpoint
            print("✅ Loaded from model weights format")
            
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()  # Set to evaluation mode
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None    # Generate predictions for all test images
    print("\n🔍 Processing test images...")
    predictions = []
    confidences = []
    processed_count = 0
    
    for idx, row in test_df.iterrows():
        filename = row['filename']
        image_path = Path(test_images_dir) / filename
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = image_transform(image).unsqueeze(0).to(device)
            
            # Generate prediction
            with torch.no_grad():
                model_output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(model_output, dim=1)
                predicted_class_index = torch.argmax(model_output, dim=1).item()
                confidence_score = probabilities[0][predicted_class_index].item()
            
            # Convert index to class name
            predicted_class_name = class_names[predicted_class_index]
            
            predictions.append(predicted_class_name)
            confidences.append(confidence_score)
            processed_count += 1
            
            # Progress indicator
            if processed_count % 100 == 0:
                print(f"   ✓ Processed {processed_count}/{len(test_df)} images")
                
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            predictions.append("ERROR")
            confidences.append(0.0)
    
    print(f"✅ Completed processing {processed_count} images")
    
    # Create results dataframe
    results_dataframe = pd.DataFrame({
        'filename': test_df['filename'],
        'predicted_class': predictions,
        'confidence': confidences
    })
    
    # Save predictions to CSV
    Path(output_file).parent.mkdir(exist_ok=True)
    results_dataframe.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Display statistics and analysis
    show_prediction_analysis(results_dataframe)
    
    return results_dataframe


def show_prediction_analysis(results_df):
    """Display analysis of prediction results"""
    
    print("\n📊 PREDICTION ANALYSIS")
    print("=" * 30)
    
    # Basic statistics
    valid_predictions = results_df[results_df['predicted_class'] != 'ERROR']
    confidence_scores = valid_predictions['confidence']
    
    print(f"📈 Confidence Statistics:")
    print(f"   Total predictions: {len(results_df)}")
    print(f"   Successful predictions: {len(valid_predictions)}")
    print(f"   Average confidence: {confidence_scores.mean():.3f}")
    print(f"   Minimum confidence: {confidence_scores.min():.3f}")
    print(f"   Maximum confidence: {confidence_scores.max():.3f}")
    print(f"   Std deviation: {confidence_scores.std():.3f}")
    
    # Confidence distribution
    high_conf = len(confidence_scores[confidence_scores >= 0.9])
    med_conf = len(confidence_scores[(confidence_scores >= 0.7) & (confidence_scores < 0.9)])
    low_conf = len(confidence_scores[confidence_scores < 0.7])
    
    print(f"\n🎯 Confidence Distribution:")
    print(f"   High confidence (≥90%): {high_conf} images ({high_conf/len(valid_predictions)*100:.1f}%)")
    print(f"   Medium confidence (70-90%): {med_conf} images ({med_conf/len(valid_predictions)*100:.1f}%)")
    print(f"   Low confidence (<70%): {low_conf} images ({low_conf/len(valid_predictions)*100:.1f}%)")
    
    # Top predicted classes
    print(f"\n🏷️ Top 10 Predicted Classes:")
    top_classes = valid_predictions['predicted_class'].value_counts().head(10)
    for rank, (class_name, count) in enumerate(top_classes.items(), 1):
        percentage = (count / len(valid_predictions)) * 100
        print(f"   {rank:2d}. {class_name}: {count} images ({percentage:.1f}%)")
    
    # Sample high-confidence predictions
    print(f"\n🎯 Sample High-Confidence Predictions:")
    high_conf_samples = valid_predictions[valid_predictions['confidence'] >= 0.95].head(5)
    for _, row in high_conf_samples.iterrows():
        print(f"   {row['filename']} → {row['predicted_class']} ({row['confidence']:.3f})")
    
    # Sample low-confidence predictions (might need review)
    low_conf_samples = valid_predictions[valid_predictions['confidence'] < 0.7].head(3)
    if len(low_conf_samples) > 0:
        print(f"\n⚠️ Low-Confidence Predictions (may need review):")
        for _, row in low_conf_samples.iterrows():
            print(f"   {row['filename']} → {row['predicted_class']} ({row['confidence']:.3f})")


def main():
    """Main execution function"""
    
    print("Starting image classification test prediction generation...\n")
    
    try:
        results = generate_test_predictions()
        
        if results is not None:
            print(f"\n🎉 SUCCESS: Test predictions generated successfully!")
            print(f"📁 Results saved to: output/results/test_predictions.csv")
            print(f"📊 Generated {len(results)} predictions")
            
            # Show file size
            results_file = Path("../output/results/test_predictions.csv")
            if results_file.exists():
                file_size = results_file.stat().st_size / 1024  # KB
                print(f"💾 File size: {file_size:.1f} KB")
                
        else:
            print(f"\n❌ FAILED: Could not generate test predictions")
            print(f"💡 Check that all required files are in place")
            
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
