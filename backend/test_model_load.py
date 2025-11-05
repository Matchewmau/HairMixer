"""Test if the ResNet50 model loads correctly with the updated architecture"""
import sys
import torch
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from hairmixer_app.ml.resnet_classifier import ResNet50FaceShapeClassifier

def test_model_load():
    print("🧪 Testing ResNet50 model loading...")
    print("-" * 60)
    
    # Model path
    model_path = Path(__file__).parent / 'hairmixer_app' / 'ml' / 'models' / 'resnet50_best_model.pth'
    print(f"Model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("❌ Model file not found!")
        return
    
    # Create model
    print("\n1️⃣ Creating model instance...")
    model = ResNet50FaceShapeClassifier(num_classes=5)
    print("✅ Model instance created")
    
    # Load checkpoint
    print("\n2️⃣ Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✅ Checkpoint loaded")
    print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
    
    # Extract state dict
    state = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
    print(f"   Total state dict keys: {len(state.keys())}")
    
    # Show sample of FC layer keys
    fc_keys = [k for k in state.keys() if 'fc' in k]
    print(f"   FC layer keys in checkpoint: {fc_keys}")
    
    # Test load with strict=True
    print("\n3️⃣ Loading weights (strict=True)...")
    try:
        missing, unexpected = model.load_state_dict(state, strict=True)
        print("✅ Model loaded successfully with strict=True!")
        print(f"   Missing keys: {missing if missing else 'none'}")
        print(f"   Unexpected keys: {unexpected if unexpected else 'none'}")
        
        # Set to eval mode
        model.eval()
        print("\n✅ Model ready for inference!")
        
        # Test inference
        print("\n4️⃣ Testing inference...")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            confidence = torch.max(probs)
        
        print(f"✅ Inference successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Predicted class: {pred_class.item()}")
        print(f"   Confidence: {confidence.item():.4f}")
        print(f"   All probabilities: {probs.numpy()[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model_load()
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS! Model loads and runs correctly!")
    else:
        print("💥 FAILED! Model has issues!")
