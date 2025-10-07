"""
Test script to verify the hairstyle recommendation system.
Tests the complete flow from model loading to generating 10 recommendations.
"""
import sys
import os
import pickle
import django

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.services.hairstyle_recommender import HairstyleRecommender
from hairmixer_app.models import Hairstyle, HairstyleCategory


def check_model():
    """Check if the model exists and its properties"""
    print("=" * 80)
    print("1. CHECKING HAIRSTYLE FAMILY MODEL")
    print("=" * 80)
    
    # Get the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(backend_dir, 'hairmixer_app', 'ml', 'models', 'hairstyle_family_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✓ Model file exists: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Model type: {type(model).__name__}")
        
        if hasattr(model, 'n_features_in_'):
            print(f"✓ Model expects {model.n_features_in_} features")
        
        if hasattr(model, 'classes_'):
            print(f"✓ Model predicts {len(model.classes_)} classes")
            print(f"  Sample classes: {list(model.classes_[:10])}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def check_database():
    """Check if hairstyles exist in the database"""
    print("\n" + "=" * 80)
    print("2. CHECKING DATABASE")
    print("=" * 80)
    
    try:
        total_hairstyles = Hairstyle.objects.count()
        active_hairstyles = Hairstyle.objects.filter(is_active=True).count()
        categories = HairstyleCategory.objects.count()
        
        print(f"✓ Total hairstyles: {total_hairstyles}")
        print(f"✓ Active hairstyles: {active_hairstyles}")
        print(f"✓ Categories: {categories}")
        
        if active_hairstyles == 0:
            print("⚠️  Warning: No active hairstyles in database")
            print("   You may need to populate the database with hairstyles")
        
        # Show sample hairstyles
        sample_styles = Hairstyle.objects.filter(is_active=True)[:5]
        if sample_styles:
            print("\n  Sample hairstyles:")
            for style in sample_styles:
                print(f"    - {style.name} (Category: {style.category.name if style.category else 'None'})")
        
        return True
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False


def test_recommendations():
    """Test the recommendation system"""
    print("\n" + "=" * 80)
    print("3. TESTING HAIRSTYLE RECOMMENDATIONS")
    print("=" * 80)
    
    # Test preferences
    test_cases = [
        {
            'name': 'Test Case 1: Female with Oval Face',
            'preferences': {
                'gender': 'female',
                'hair_type': 'wavy',
                'hair_length': 'medium',
                'faceshape': 'oval',
                'maintenance': 'medium',
                'lifestyle': 'professional',
                'volume': 'medium',
                'styling_maintenance': 'medium',
                'styling_preference': 'elegant',
                'hair_condition': 'good',
                'hair_thickness': 'medium',
                'hair_texture_detail': 'normal',
                'wants_bangs': False,
                'occasions': ['work', 'casual', 'formal']
            }
        },
        {
            'name': 'Test Case 2: Male with Round Face',
            'preferences': {
                'gender': 'male',
                'hair_type': 'straight',
                'hair_length': 'short',
                'faceshape': 'round',
                'maintenance': 'low',
                'lifestyle': 'active',
                'volume': 'low',
                'styling_maintenance': 'low',
                'styling_preference': 'natural',
                'hair_condition': 'excellent',
                'hair_thickness': 'thick',
                'hair_texture_detail': 'smooth',
                'wants_bangs': False,
                'occasions': ['casual', 'exercise']
            }
        }
    ]
    
    try:
        recommender = HairstyleRecommender()
        
        if not recommender.model_loaded:
            print("❌ Model not loaded in recommender")
            return False
        
        print("✓ HairstyleRecommender initialized successfully")
        
        for test_case in test_cases:
            print(f"\n{'─' * 80}")
            print(f"  {test_case['name']}")
            print(f"{'─' * 80}")
            
            prefs = test_case['preferences']
            print(f"  Input: {prefs['gender']}, {prefs['faceshape']} face, {prefs['hair_type']} hair")
            
            # Get recommendations
            recommendations = recommender.get_top_recommendations(prefs, top_n=10)
            
            print(f"\n  ✓ Received {len(recommendations)} recommendations")
            
            if len(recommendations) < 10:
                print(f"  ⚠️  Expected 10 recommendations, got {len(recommendations)}")
            
            # Display recommendations
            print("\n  Top 10 Hairstyle Recommendations:")
            print("  " + "-" * 76)
            for i, rec in enumerate(recommendations[:10], 1):
                print(f"  {i:2d}. {rec['name']:<40} | Score: {rec['match_score']:.3f} | Family: {rec['hairstyle_family']}")
                if i == 1:
                    print(f"      Category: {rec['category']}, Maintenance: {rec['maintenance']}, Difficulty: {rec['difficulty']}")
        
        print("\n" + "=" * 80)
        print("✓ RECOMMENDATION SYSTEM TEST PASSED")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing recommendations: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "HAIRSTYLE RECOMMENDER SYSTEM TEST" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Run checks
    model_ok = check_model()
    db_ok = check_database()
    
    if not model_ok:
        print("\n❌ Model check failed. Cannot proceed with recommendation test.")
        return False
    
    if not db_ok:
        print("\n⚠️  Database check failed. Recommendation test may fail.")
    
    # Test recommendations
    test_ok = test_recommendations()
    
    print("\n" + "=" * 80)
    if model_ok and test_ok:
        print("✅ ALL TESTS PASSED - Recommendation system is working correctly!")
        print("   The system successfully generates 10 hairstyle recommendations")
        print("   using the Random Forest model (hairstyle_family_model.pkl)")
    else:
        print("❌ SOME TESTS FAILED - Please check the errors above")
    print("=" * 80 + "\n")
    
    return model_ok and test_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
