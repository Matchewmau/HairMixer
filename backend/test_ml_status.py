#!/usr/bin/env python3
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
sys.path.append('.')
django.setup()

print("🧪 Testing ML Components Status...\n")

# Test 1: Basic ML libraries
print("1️⃣ Testing Basic ML Libraries:")
try:
    import cv2
    print("✅ OpenCV available:", cv2.__version__)
except ImportError as e:
    print("❌ OpenCV not available:", str(e))

try:
    import numpy as np
    print("✅ NumPy available:", np.__version__)
except ImportError as e:
    print("❌ NumPy not available:", str(e))

try:
    from sklearn import __version__
    print("✅ Scikit-learn available:", __version__)
except ImportError as e:
    print("❌ Scikit-learn not available:", str(e))

print("\n2️⃣ Testing HairMixer ML Components:")

# Test 2: Preprocess module
try:
    from hairmixer_app.ml.preprocess import read_image, detect_face, validate_image_quality
    print("✅ Preprocess module available")
    
    # Test basic functionality
    import tempfile
    from PIL import Image
    
    # Create a test image
    test_img = Image.new('RGB', (100, 100), color='red')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_img.save(tmp.name)
        
        # Test read_image
        img_array = read_image(tmp.name)
        print("  ✅ read_image works:", img_array.shape if img_array is not None else "Failed")
        
        # Test detect_face
        face_detected, faces = detect_face(img_array)
        print("  ✅ detect_face works:", face_detected, "faces found:", len(faces) if faces else 0)
        
        # Test validate_image_quality
        quality = validate_image_quality(img_array)
        print("  ✅ validate_image_quality works:", quality.get('is_valid', 'Unknown'))
        
    os.unlink(tmp.name)
    
except Exception as e:
    print("❌ Preprocess module failed:", str(e))

# Test 3: Model module
try:
    from hairmixer_app.ml.model import load_model
    print("✅ Model module available")
    
    # Try to load model
    model = load_model()
    print("  ✅ Model loading:", "Success" if model else "Failed (expected for now)")
    
except Exception as e:
    print("❌ Model module failed:", str(e))

# Test 4: Face analyzer
try:
    from hairmixer_app.ml.face_analyzer import FacialFeatureAnalyzer
    analyzer = FacialFeatureAnalyzer()
    print("✅ Face analyzer available")
except Exception as e:
    print("❌ Face analyzer failed:", str(e))

# Test 5: Recommendation engine
try:
    from hairmixer_app.logic.recommendation_engine import EnhancedRecommendationEngine
    engine = EnhancedRecommendationEngine()
    print("✅ Recommendation engine available")
except Exception as e:
    print("❌ Recommendation engine failed:", str(e))

# Test 6: Views availability flags
try:
    from hairmixer_app.views import (
        ML_MODEL_AVAILABLE,
        ML_PREPROCESS_AVAILABLE, 
        RECOMMENDATION_ENGINE_AVAILABLE
    )
    print("\n3️⃣ Views Import Status:")
    print("  ML_MODEL_AVAILABLE:", ML_MODEL_AVAILABLE)
    print("  ML_PREPROCESS_AVAILABLE:", ML_PREPROCESS_AVAILABLE)
    print("  RECOMMENDATION_ENGINE_AVAILABLE:", RECOMMENDATION_ENGINE_AVAILABLE)
except Exception as e:
    print("❌ Views import failed:", str(e))

print("\n🏁 ML Status Test Complete!")