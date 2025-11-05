"""
Test to diagnose model bias issue
"""
import os
import sys
import django
import torch
import numpy as np
from pathlib import Path

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.ml.resnet_classifier import resnet_loader
from PIL import Image
import torchvision.transforms as transforms

# Face shape mapping
FACE_SHAPES = {
    0: "heart",
    1: "oblong",
    2: "oval",
    3: "round",
    4: "square"
}

def test_random_inputs():
    """Test with random inputs to see if model is stuck"""
    print("=" * 60)
    print("TEST 1: Random Input Tensors")
    print("=" * 60)
    
    # Load model
    if not resnet_loader.is_loaded():
        resnet_loader.load_model()
    
    for i in range(5):
        # Create random tensor
        random_tensor = torch.randn(1, 3, 224, 224)
        result = resnet_loader.predict(random_tensor)
        
        if result:
            pred_class = result['predicted_class']
            confidence = result['confidence']
            probs = result['probabilities']
            
            print(f"\nRandom Input {i+1}:")
            print(f"  Predicted: {FACE_SHAPES[pred_class]} (conf: {confidence:.3f})")
            print(f"  Probabilities: {', '.join([f'{FACE_SHAPES[j]}={probs[j]:.3f}' for j in range(5)])}")


def test_with_images():
    """Test with actual uploaded images"""
    print("\n" + "=" * 60)
    print("TEST 2: Actual Images")
    print("=" * 60)
    
    uploads_dir = Path("media/uploads/2025/11")
    if not uploads_dir.exists():
        print("No uploads directory found")
        return
    
    images = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.jpeg")) + list(uploads_dir.glob("*.png"))
    
    if not images:
        print("No images found")
        return
    
    # Image preprocessing (same as in face_analyzer)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    for img_path in images[:5]:  # Test first 5 images
        try:
            # Load and preprocess
            pil_img = Image.open(img_path).convert('RGB')
            tensor = transform(pil_img).unsqueeze(0)
            
            # Get prediction
            result = resnet_loader.predict(tensor)
            
            if result:
                pred_class = result['predicted_class']
                confidence = result['confidence']
                probs = result['probabilities']
                
                print(f"\n{img_path.name}:")
                print(f"  Image size: {pil_img.size}")
                print(f"  Tensor stats: min={tensor.min():.3f}, max={tensor.max():.3f}, mean={tensor.mean():.3f}")
                print(f"  Predicted: {FACE_SHAPES[pred_class]} (conf: {confidence:.3f})")
                print(f"  Probabilities: {', '.join([f'{FACE_SHAPES[j]}={probs[j]:.3f}' for j in range(5)])}")
        except Exception as e:
            print(f"\nError with {img_path.name}: {e}")


def test_extreme_inputs():
    """Test with extreme inputs"""
    print("\n" + "=" * 60)
    print("TEST 3: Extreme Inputs")
    print("=" * 60)
    
    test_cases = [
        ("All zeros", torch.zeros(1, 3, 224, 224)),
        ("All ones", torch.ones(1, 3, 224, 224)),
        ("All -1", torch.full((1, 3, 224, 224), -1.0)),
        ("All 10", torch.full((1, 3, 224, 224), 10.0)),
    ]
    
    for name, tensor in test_cases:
        result = resnet_loader.predict(tensor)
        if result:
            pred_class = result['predicted_class']
            confidence = result['confidence']
            probs = result['probabilities']
            
            print(f"\n{name}:")
            print(f"  Predicted: {FACE_SHAPES[pred_class]} (conf: {confidence:.3f})")
            print(f"  Probabilities: {', '.join([f'{FACE_SHAPES[j]}={probs[j]:.3f}' for j in range(5)])}")


if __name__ == '__main__':
    print("Testing ResNet50 Face Shape Classifier for Bias Issues\n")
    
    test_random_inputs()
    test_with_images()
    test_extreme_inputs()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print("""
If all tests show similar probability distributions (especially favoring 'square'),
the model has a bias issue from training. This could be due to:

1. Imbalanced training data (too many square faces)
2. Poor feature extraction (model not learning meaningful features)
3. Overfitting to the validation set
4. Incorrect preprocessing during training vs inference

Solution: The model needs to be retrained with:
- Balanced dataset across all 5 face shapes
- Data augmentation
- Proper validation strategy
- Verified preprocessing pipeline
    """)
