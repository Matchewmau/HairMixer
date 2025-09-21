#!/usr/bin/env python3
"""
Test the face detection API endpoint
"""
import requests
from pathlib import Path

def test_api_endpoint():
    """Test the face detection API"""
    
    # Test image path
    test_image = Path("uploads/2025/08/best-haircuts-for-every-face-shape-277864-1551404529439-main.jpg")
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"🔍 Testing API endpoint with: {test_image}")
    
    try:
        # Test the upload endpoint
        url = "http://127.0.0.1:8000/api/upload/"
        
        with open(test_image, 'rb') as f:
            files = {'image': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post(url, files=files, timeout=30)
        
        print(f"📋 API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"   - Face detected: {result.get('face_detected', False)}")
            print(f"   - Face shape: {result.get('face_shape', {}).get('shape', 'N/A')}")
            print(f"   - Confidence: {result.get('confidence', 'N/A')}")
            print(f"   - Detection method: {result.get('detection_method', 'N/A')}")
            print(f"   - Quality score: {result.get('quality_metrics', {}).get('overall_quality', 'N/A')}")
        else:
            print("❌ API call failed!")
            print(f"   - Status: {response.status_code}")
            print(f"   - Response: {response.text}")
        
    except Exception as e:
        print(f"💥 Exception occurred: {str(e)}")

if __name__ == "__main__":
    test_api_endpoint()
