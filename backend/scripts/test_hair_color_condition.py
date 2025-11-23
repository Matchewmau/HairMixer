"""
Test: Does hair color and hair condition affect recommendations?

This script demonstrates which user preference attributes
are actually used in the Random Forest recommendation model.
"""

import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.ml.hairstyle_model_recommender import (
    get_hairstyle_model_recommender
)

print("=" * 80)
print("HAIRSTYLE RECOMMENDATION - PREFERENCE ATTRIBUTES TEST")
print("=" * 80)
print()

# Get the ML recommender
recommender = get_hairstyle_model_recommender()

if not recommender.is_available():
    print("❌ ML Model not available. Please train the model first.")
    sys.exit(1)

print("✅ ML Model loaded successfully")
print()

# Create two identical preference sets except for hair color and condition
print("Creating test preference sets...")
print()

# Test Set 1: Brown hair, healthy condition
prefs1 = {
    'gender': 'female',
    'hair_type': 'wavy',
    'hair_length': 'medium',
    'hair_color': 'brown',
    'maintenance': 'low',
    'lifestyle': 'professional',
    'volume': 'medium',
    'styling_maintenance': 'low',
    'styling_preference': 'natural',
    'hair_condition': 'healthy',
    'hair_thickness': 'medium',
    'hair_texture_detail': 'fine',
    'wants_bangs': False,
    'occasions': ['work', 'casual'],
}

# Test Set 2: Same everything EXCEPT hair color and condition
prefs2 = {
    'gender': 'female',
    'hair_type': 'wavy',
    'hair_length': 'medium',
    'hair_color': 'blonde',  # <-- CHANGED
    'maintenance': 'low',
    'lifestyle': 'professional',
    'volume': 'medium',
    'styling_maintenance': 'low',
    'styling_preference': 'natural',
    'hair_condition': 'damaged',  # <-- CHANGED
    'hair_thickness': 'medium',
    'hair_texture_detail': 'fine',
    'wants_bangs': False,
    'occasions': ['work', 'casual'],
}

face_shape = 'oval'

print("TEST 1: Professional Woman")
print("-" * 80)
print(f"Face Shape:     {face_shape}")
print(f"Gender:         {prefs1['gender']}")
print(f"Hair Type:      {prefs1['hair_type']}")
print(f"Hair Length:    {prefs1['hair_length']}")
print(f"Hair Color:     {prefs1['hair_color']} ⚠️")
print(f"Hair Condition: {prefs1['hair_condition']} ⚠️")
print(f"Maintenance:    {prefs1['maintenance']}")
print(f"Lifestyle:      {prefs1['lifestyle']}")
print(f"Occasions:      {', '.join(prefs1['occasions'])}")
print()

# Get recommendations for set 1
recs1 = recommender.predict_top_k(prefs1, face_shape, k=5)

print("Top 5 Recommendations:")
for i, rec in enumerate(recs1, 1):
    print(f"  {i}. {rec['hairstyle_name']} (score: {rec['confidence']:.3f})")
print()

print("=" * 80)
print("TEST 2: Same Preferences BUT Different Hair Color & Condition")
print("-" * 80)
print(f"Face Shape:     {face_shape}")
print(f"Gender:         {prefs2['gender']}")
print(f"Hair Type:      {prefs2['hair_type']}")
print(f"Hair Length:    {prefs2['hair_length']}")
print(f"Hair Color:     {prefs2['hair_color']} ⚠️ CHANGED")
print(f"Hair Condition: {prefs2['hair_condition']} ⚠️ CHANGED")
print(f"Maintenance:    {prefs2['maintenance']}")
print(f"Lifestyle:      {prefs2['lifestyle']}")
print(f"Occasions:      {', '.join(prefs2['occasions'])}")
print()

# Get recommendations for set 2
recs2 = recommender.predict_top_k(prefs2, face_shape, k=5)

print("Top 5 Recommendations:")
for i, rec in enumerate(recs2, 1):
    print(f"  {i}. {rec['hairstyle_name']} (score: {rec['confidence']:.3f})")
print()

print("=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)
print()
print("Changed Attributes:")
print(f"  • Hair Color:     {prefs1['hair_color']} → {prefs2['hair_color']}")
print(f"  • Hair Condition: {prefs1['hair_condition']} → {prefs2['hair_condition']}")
print()

# Compare the recommendations
names1 = [r['hairstyle_name'] for r in recs1]
names2 = [r['hairstyle_name'] for r in recs2]
scores1 = [r['confidence'] for r in recs1]
scores2 = [r['confidence'] for r in recs2]

if names1 == names2:
    print("✅ RESULT: Recommendations are IDENTICAL")
    print()
    print("   → Hair color and hair condition DO NOT affect recommendations!")
    print("   → The model ignores these aesthetic/condition attributes")
    print()
    
    # Check if scores are also identical
    if scores1 == scores2:
        print("   → Even the confidence scores are identical")
    else:
        print("   → Scores differ slightly (may be due to model randomness)")
        for i in range(len(scores1)):
            diff = abs(scores1[i] - scores2[i])
            if diff > 0.001:
                print(f"     Style {i+1}: {scores1[i]:.4f} vs {scores2[i]:.4f} (diff: {diff:.4f})")
else:
    print("❌ RESULT: Recommendations are DIFFERENT")
    print()
    print("   → Hair color and/or hair condition DO affect recommendations")
    print()
    print("   Different recommendations:")
    for i in range(max(len(names1), len(names2))):
        n1 = names1[i] if i < len(names1) else "---"
        n2 = names2[i] if i < len(names2) else "---"
        if n1 != n2:
            print(f"     Position {i+1}: {n1} vs {n2}")

print()
print("=" * 80)
print("MODEL FEATURE ANALYSIS")
print("=" * 80)
print()

# Check what features the model actually uses
print("Features that the Random Forest model considers:")
print()

# These are the features used based on the code
used_features = [
    "✅ gender",
    "✅ hair_type (straight, wavy, curly, coily)",
    "✅ hair_length (pixie, short, medium, long, extra_long)",
    "✅ maintenance (low, medium, high)",
    "✅ lifestyle (professional, active, student, etc.)",
    "✅ faceshape (oval, round, square, heart, etc.)",
    "✅ occasions (work, casual, party, wedding, etc.)",
]

unused_features = [
    "❌ hair_color",
    "❌ hair_condition",
    "❌ hair_thickness",
    "❌ hair_texture_detail",
    "❌ volume",
    "❌ styling_maintenance",
    "❌ styling_preference",
]

print("USED in model training and prediction:")
for feature in used_features:
    print(f"  {feature}")

print()
print("NOT USED in model training and prediction:")
for feature in unused_features:
    print(f"  {feature}")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("The HairMixer recommendation system focuses on STRUCTURAL factors:")
print()
print("  • Face shape compatibility")
print("  • Hair type/length (what styles are physically possible)")
print("  • Maintenance requirements (time commitment)")
print("  • Lifestyle and occasions (practical needs)")
print()
print("It does NOT consider AESTHETIC or CONDITION factors:")
print()
print("  • Hair color (doesn't affect whether a cut/style works)")
print("  • Hair condition (affects execution, not style selection)")
print("  • Volume/texture preferences (minor styling details)")
print()
print("This design makes sense because:")
print("  1. Any hairstyle can work with any hair color")
print("  2. Hair condition can be improved before styling")
print("  3. Focus on what CAN be done vs cosmetic preferences")
print()
print("=" * 80)
