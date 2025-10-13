"""
Test script to verify the new RF No-Family hairstyle recommender integration.

This script tests:
1. Model loading
2. Feature encoding
3. Prediction generation
4. Database lookup

Usage:
    python test_rf_recommender.py
"""

import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.ml.hairstyle_recommender import get_hairstyle_recommender
from hairmixer_app.models import Hairstyle, HairstyleCategory


def test_model_loading():
    """Test if the RF model loads successfully"""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)
    
    recommender = get_hairstyle_recommender()
    
    if recommender.is_available():
        print("✓ Model loaded successfully")
        info = recommender.get_model_info()
        print(f"  - Model type: {info.get('model_type')}")
        print(f"  - Features: {info.get('n_features')}")
        print(f"  - Classes: {info.get('n_classes')}")
        if info.get('metadata'):
            metadata = info['metadata']
            print(f"  - Top-10 Accuracy: {metadata.get('top_10_accuracy', 0)*100:.1f}%")
        return True
    else:
        print("✗ Model failed to load")
        return False


def test_feature_encoding():
    """Test feature encoding with sample preferences"""
    print("\n" + "="*60)
    print("TEST 2: Feature Encoding")
    print("="*60)
    
    recommender = get_hairstyle_recommender()
    
    if not recommender.is_available():
        print("✗ Model not available")
        return False
    
    # Sample user preferences
    test_prefs = {
        'gender': 'female',
        'hair_type': 'wavy',
        'hair_length': 'medium',
        'hair_color': 'brown',
        'lifestyle': 'active',
        'maintenance': 'low',
        'volume': 'medium',
        'hair_thickness': 'medium',
        'hair_texture_detail': 'normal',
        'styling_maintenance': 'low',
        'styling_preference': 'natural',
        'occasions': ['work', 'casual'],
        'wants_bangs': False,
        'hair_condition': 'none'
    }
    
    print("Sample preferences:")
    for key, value in test_prefs.items():
        print(f"  - {key}: {value}")
    
    try:
        # Test encoding
        feature_values = {}
        feature_values['faceshape'] = 'oval'
        for key, value in test_prefs.items():
            if key == 'hair_thickness':
                feature_values['thickness'] = value
            elif key == 'hair_texture_detail':
                feature_values['texture_detail'] = value
            elif key == 'occasions':
                feature_values['occasions'] = value[0] if value else 'casual'
            elif key == 'wants_bangs':
                feature_values['wants_bangs'] = 'true' if value else 'false'
            else:
                feature_values[key] = value
        
        encoded = recommender._encode_features(feature_values)
        
        if encoded is not None:
            print(f"✓ Features encoded successfully")
            print(f"  - Encoded shape: {encoded.shape}")
            print(f"  - Encoded values: {encoded[0][:5]}... (first 5)")
            return True
        else:
            print("✗ Feature encoding failed")
            return False
            
    except Exception as e:
        print(f"✗ Error during encoding: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """Test generating predictions"""
    print("\n" + "="*60)
    print("TEST 3: Prediction Generation")
    print("="*60)
    
    recommender = get_hairstyle_recommender()
    
    if not recommender.is_available():
        print("✗ Model not available")
        return False
    
    # Sample user preferences
    test_prefs = {
        'gender': 'female',
        'hair_type': 'wavy',
        'hair_length': 'medium',
        'hair_color': 'brown',
        'lifestyle': 'active',
        'maintenance': 'low',
        'volume': 'medium',
        'hair_thickness': 'medium',
        'hair_texture_detail': 'normal',
        'styling_maintenance': 'low',
        'styling_preference': 'natural',
        'occasions': ['work', 'casual'],
        'wants_bangs': False,
        'hair_condition': 'none'
    }
    
    try:
        recommendations = recommender.predict_top_k(
            test_prefs,
            face_shape='oval',
            k=10
        )
        
        if recommendations:
            print(f"✓ Generated {len(recommendations)} recommendations")
            print("\nTop 5 recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(
                    f"  {i}. {rec['hairstyle_name']} "
                    f"(confidence: {rec['confidence']:.3f})"
                )
            return True
        else:
            print("✗ No recommendations generated")
            return False
            
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_lookup():
    """Test looking up hairstyles in database"""
    print("\n" + "="*60)
    print("TEST 4: Database Lookup")
    print("="*60)
    
    # Check if hairstyles exist in database
    total_hairstyles = Hairstyle.objects.count()
    active_hairstyles = Hairstyle.objects.filter(is_active=True).count()
    categories = HairstyleCategory.objects.count()
    
    print(f"Database statistics:")
    print(f"  - Total hairstyles: {total_hairstyles}")
    print(f"  - Active hairstyles: {active_hairstyles}")
    print(f"  - Categories: {categories}")
    
    if total_hairstyles == 0:
        print("\n⚠ Warning: No hairstyles in database!")
        print("  Run: python manage.py import_hairstyles_catalog")
        return False
    else:
        print("\n✓ Database has hairstyles")
        
        # Show sample hairstyles
        samples = Hairstyle.objects.filter(is_active=True)[:5]
        print("\nSample hairstyles:")
        for hs in samples:
            print(f"  - {hs.name}")
        return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RF NO-FAMILY MODEL INTEGRATION TEST")
    print("="*60)
    
    results = {
        'Model Loading': test_model_loading(),
        'Feature Encoding': test_feature_encoding(),
        'Prediction': test_prediction(),
        'Database Lookup': test_database_lookup()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
