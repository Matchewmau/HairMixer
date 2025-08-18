#!/usr/bin/env python3

print("Testing imports step by step...")

# Test 1: Basic Django
try:
    import django
    print("✅ Django imported successfully")
except ImportError as e:
    print(f"❌ Django import failed: {e}")

# Test 2: Basic ML libraries
try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import numpy
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

# Test 3: App components
try:
    import sys
    sys.path.append('.')
    from hairmixer_app.ml.preprocess import read_image
    print("✅ Preprocess module imported successfully")
except Exception as e:
    print(f"❌ Preprocess import failed: {e}")

try:
    from hairmixer_app.ml.model import load_model
    print("✅ Model module imported successfully")
except Exception as e:
    print(f"❌ Model import failed: {e}")

try:
    from hairmixer_app import views
    print("✅ Views imported successfully")
except Exception as e:
    print(f"❌ Views import failed: {e}")

print("\nDiagnostic completed!")