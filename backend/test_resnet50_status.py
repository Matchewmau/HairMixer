"""
Test script to check ResNet50 initialization status
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
import django
django.setup()

from hairmixer_app.ml.face_analyzer import FacialFeatureAnalyzer
import logging

# Set up logging to see debug info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_resnet50_status():
    """Test if ResNet50 is properly loaded"""
    print("🔍 Testing ResNet50 initialization...")
    
    try:
        # Force a new instance to see initialization logs
        analyzer = FacialFeatureAnalyzer()
        
        print("\n📊 Analyzer Status:")
        print(f"  - Detector Type: {getattr(analyzer, 'detector_type', 'Unknown')}")
        print(f"  - Has Feature Extractor: {hasattr(analyzer, 'feature_extractor')}")
        print(f"  - Feature Extractor Value: {getattr(analyzer, 'feature_extractor', None)}")
        print(f"  - Shape Classifier: {getattr(analyzer, 'shape_classifier', None)}")
        print(f"  - Device: {getattr(analyzer, 'device', 'Unknown')}")
        
        # Test ResNet50 loading directly
        print("\n🔧 Testing ResNet50 loading directly...")
        try:
            from torchvision.models import resnet50
            import torch
            import torch.nn as nn
            
            print("  - Importing ResNet50...")
            model = resnet50(pretrained=True)
            print(f"  - ResNet50 loaded: {model is not None}")
            
            # Test removing final layer
            print("  - Removing final classification layer...")
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            print(f"  - Feature extractor created: {feature_extractor is not None}")
            
            # Test device placement
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  - Moving to device: {device}")
            feature_extractor.to(device)
            feature_extractor.eval()
            print("  - Model ready for inference")
            
            print("✅ ResNet50 can be loaded manually!")
            
            # Now test why analyzer._load_shape_classifier() might be failing
            print("\n🔍 Testing analyzer._load_shape_classifier()...")
            try:
                analyzer._load_shape_classifier()
                print("✅ _load_shape_classifier() succeeded")
                print(f"  - Feature extractor after load: {analyzer.feature_extractor}")
                print(f"  - Shape classifier after load: {analyzer.shape_classifier}")
            except Exception as e:
                print(f"❌ _load_shape_classifier() failed: {str(e)}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"❌ Manual ResNet50 loading failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error testing ResNet50: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_resnet50_status()
