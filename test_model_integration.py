"""
Test script for the new hairstyle model integration
"""
import os
import django

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.ml.hairstyle_model_recommender import get_hairstyle_model_recommender

def test_model():
    print("=" * 80)
    print("Testing Hairstyle Model Integration")
    print("=" * 80)
    
    # Get recommender instance
    recommender = get_hairstyle_model_recommender()
    
    # Print model info
    info = recommender.get_model_info()
    print("\n✅ Model Info:")
    print(f"   Loaded: {info['loaded']}")
    print(f"   Features: {info['n_features']}")
    print(f"   Classes: {info['n_classes']}")
    print(f"   Model Type: {info['model_type']}")
    
    print("\n✅ Available Hairstyle Classes:")
    for i, name in enumerate(info['classes'][:10], 1):
        print(f"   {i}. {name}")
    print(f"   ... and {info['n_classes'] - 10} more")
    
    # Test prediction with sample preferences
    print("\n" + "=" * 80)
    print("Testing Prediction with Sample Preferences")
    print("=" * 80)
    
    test_cases = [
        {
            'name': 'Female - Professional - Medium Hair',
            'prefs': {
                'gender': 'female',
                'hair_type': 'wavy',
                'hair_length': 'medium',
                'hair_color': 'brown',
                'lifestyle': 'professional',
                'maintenance': 'medium',
                'volume': 'medium',
                'styling_maintenance': 'medium',
                'styling_preference': 'polished',
                'hair_condition': ['good'],
                'hair_thickness': 'medium',
                'hair_texture_detail': 'normal',
                'wants_bangs': False,
                'occasions': ['work', 'casual']
            },
            'face_shape': 'oval'
        },
        {
            'name': 'Male - Active - Short Hair',
            'prefs': {
                'gender': 'male',
                'hair_type': 'straight',
                'hair_length': 'short',
                'hair_color': 'black',
                'lifestyle': 'active',
                'maintenance': 'low',
                'volume': 'medium',
                'styling_maintenance': 'low',
                'styling_preference': 'natural',
                'hair_condition': [],
                'hair_thickness': 'thick',
                'hair_texture_detail': 'normal',
                'wants_bangs': False,
                'occasions': ['casual', 'work']
            },
            'face_shape': 'square'
        },
        {
            'name': 'Female - Creative - Long Hair',
            'prefs': {
                'gender': 'female',
                'hair_type': 'curly',
                'hair_length': 'long',
                'hair_color': 'auburn',
                'lifestyle': 'creative',
                'maintenance': 'high',
                'volume': 'high',
                'styling_maintenance': 'high',
                'styling_preference': 'glamorous',
                'hair_condition': ['frizzy', 'dry_ends'],
                'hair_thickness': 'thick',
                'hair_texture_detail': 'coarse',
                'wants_bangs': True,
                'occasions': ['party', 'casual']
            },
            'face_shape': 'heart'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 Test Case: {test_case['name']}")
        print(f"   Face Shape: {test_case['face_shape']}")
        print(f"   Preferences: {test_case['prefs']['gender']}, {test_case['prefs']['hair_type']}, {test_case['prefs']['hair_length']}")
        
        results = recommender.predict_top_k(
            test_case['prefs'],
            test_case['face_shape'],
            k=10
        )
        
        if results:
            print(f"\n   ✅ Top 10 Recommendations:")
            for r in results:
                print(f"      {r['rank']:2d}. {r['hairstyle_name']:<30} ({r['confidence']*100:5.2f}%)")
        else:
            print("   ❌ No recommendations generated")
    
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)

if __name__ == '__main__':
    test_model()
