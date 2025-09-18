#!/usr/bin/env python3

from importlib.util import find_spec

print("Testing module availability step by step...")

def check_module(name: str, label: str = None):
    label = label or name
    try:
        available = find_spec(name) is not None
        if available:
            print(f"✅ {label} available")
        else:
            print(f"❌ {label} NOT available")
    except Exception as e:
        print(f"❌ {label} check failed: {e}")

# Test 1: Basic Django
check_module("django", "Django")

# Test 2: Basic ML libraries
check_module("cv2", "OpenCV (cv2)")
check_module("numpy", "NumPy")

# Test 3: App components
check_module("hairmixer_app.ml.preprocess", "Preprocess module")
check_module("hairmixer_app.ml.model", "Model module")
check_module("hairmixer_app.views", "Views module")

print("\nDiagnostic completed!")