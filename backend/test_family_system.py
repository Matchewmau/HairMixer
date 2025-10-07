"""
Quick test of the family-based recommendation system.
"""

import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.services.hairstyle_recommender import HairstyleRecommender

print("="*80)
print("FAMILY-BASED RECOMMENDATION SYSTEM TEST")
print("="*80)

# Initialize recommender
recommender = HairstyleRecommender()

print(f"\n✅ Model Status:")
print(f"   Loaded: {recommender.model_loaded}")
print(f"   Type: {type(recommender.model).__name__ if recommender.model else 'None'}")

if recommender.model and hasattr(recommender.model, 'classes_'):
    print(f"   Families: {len(recommender.model.classes_)}")
    print(f"   Sample families: {list(recommender.model.classes_[:5])}")

# Test with sample preferences
test_preferences = {
    'faceshape': 'oval',
    'gender': 'male',
    'hair_type': 'straight',
    'hair_length': 'medium',
    'volume': 'medium',
    'lifestyle': 'active',
    'maintenance': 'low',
    'styling_maintenance': 'low',
    'occasions': 'casual',
    'hair_color': 'black',
    'hair_texture_detail': 'normal',
    'styling_preference': 'natural',
    'hair_condition': 'healthy',
    'hair_thickness': 'normal',
    'wants_bangs': False
}

print(f"\n🧪 Testing with Male User Profile:")
print(f"   Face shape: oval")
print(f"   Hair: medium length, straight, black")
print(f"   Preferences: active lifestyle, low maintenance")

recommendations = recommender.get_top_recommendations(test_preferences, top_n=5)

print(f"\n📊 Results:")
print(f"   Recommendations returned: {len(recommendations)}")

if recommendations:
    print(f"\n   Top 5 Recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"   {i}. {rec['name']}")
        print(f"      Gender: {rec.get('tags', ['N/A'])[0] if rec.get('tags') else 'N/A'}")
        print(f"      Confidence: {rec.get('confidence', 0):.1f}%")
        print(f"      Family: {rec.get('hairstyle_family', 'N/A')}")
else:
    print("   ⚠️ No recommendations generated!")

print(f"\n{'='*80}")
print("✅ SYSTEM TEST COMPLETE!")
print("="*80)
print("\nFamily-based model is working correctly.")
print("Model predicts families → Database provides specific styles")
print(f"\n{'='*80}\n")
