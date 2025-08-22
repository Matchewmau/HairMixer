#!/usr/bin/env python3
"""
Direct test script for face detection functionality
"""
import os
import sys
import django
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.ml.face_analyzer import FacialFeatureAnalyzer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Also set up Django logging
from django.conf import settings
import logging as django_logging
django_logging.getLogger('hairmixer_app.ml.face_analyzer').setLevel(logging.INFO)

def test_face_detection():
    """Test face detection with actual uploaded images"""
    
    # Test image path
    test_image = "uploads/2025/08/best-haircuts-for-every-face-shape-277864-1551404529439-main.jpg"
    test_image_path = backend_dir / test_image
    
    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    print(f"🔍 Testing face detection with: {test_image}")
    print(f"📁 Full path: {test_image_path}")
    print(f"📊 File exists: {test_image_path.exists()}")
    print(f"📊 File size: {test_image_path.stat().st_size} bytes")
    
    try:
        # Initialize analyzer
        print("\n🤖 Initializing FacialFeatureAnalyzer...")
        analyzer = FacialFeatureAnalyzer()
        
        print(f"🔧 Detector type: {analyzer.detector_type}")
        print(f"🔧 MediaPipe available: {hasattr(analyzer, 'face_detector') and analyzer.face_detector is not None}")
        print(f"🔧 ResNet available: {hasattr(analyzer, 'feature_extractor') and analyzer.feature_extractor is not None}")
        
        # Test face detection
        print("\n🎯 Running face detection...")
        
        # Add more detailed logging
        import cv2
        test_img = cv2.imread(str(test_image_path))
        print(f"📊 Image loaded successfully: {test_img is not None}")
        if test_img is not None:
            print(f"📊 Image shape: {test_img.shape}")
            print(f"📊 Image data type: {test_img.dtype}")
        
        result, error = analyzer.detect_and_analyze_face(str(test_image_path))
        
        print(f"\n📋 Results:")
        if result:
            print("✅ Face detection successful!")
            print(f"   - Face detected: {result.get('face_detected', False)}")
            print(f"   - Confidence: {result.get('confidence', 'N/A')}")
            print(f"   - Face shape: {result.get('face_shape', 'N/A')}")
            print(f"   - Detection method: {result.get('detection_method', 'N/A')}")
            print(f"   - Face box: {result.get('face_box', 'N/A')}")
            if 'quality_metrics' in result:
                print(f"   - Quality metrics: {result['quality_metrics']}")
            if 'facial_features' in result:
                print(f"   - Facial features: {result['facial_features']}")
        else:
            print("❌ Face detection failed!")
            print(f"   - Error: {error}")
        
        # Test another image if available
        test_image2 = "uploads/2025/08/7d9a66a024b2b8d2d48106672eb9579b.jpg"
        test_image_path2 = backend_dir / test_image2
        
        if test_image_path2.exists():
            print(f"\n🔍 Testing second image: {test_image2}")
            result2, error2 = analyzer.detect_and_analyze_face(str(test_image_path2))
            
            if result2:
                print("✅ Second image detection successful!")
                print(f"   - Face shape: {result2.get('face_shape', 'N/A')}")
                print(f"   - Confidence: {result2.get('confidence', 'N/A')}")
            else:
                print("❌ Second image detection failed!")
                print(f"   - Error: {error2}")
        
    except Exception as e:
        print(f"💥 Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_detection()
