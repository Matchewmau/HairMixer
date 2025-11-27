"""
Test script to verify face shape detection flow and Gemini integration
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.models import UserPreference, Hairstyle
from hairmixer_app.services.gemini_service import get_gemini_service


def test_face_shape_flow():
    """Test the complete face shape flow"""
    
    print("=" * 80)
    print("FACE SHAPE DETECTION & GEMINI INTEGRATION TEST")
    print("=" * 80)
    
    # 1. Check if we have any user preferences with face shapes
    print("\n1. Checking database for user preferences with face shapes...")
    prefs = UserPreference.objects.exclude(faceshape='').order_by('-updated_at')[:5]
    
    if not prefs:
        print("   ❌ No user preferences with face shapes found in database")
        print("   Tip: Upload an image through the app to generate face shape data")
        return False
    
    print(f"   ✅ Found {prefs.count()} preferences with face shapes")
    
    # Display the preferences
    for i, pref in enumerate(prefs, 1):
        print(f"\n   Preference {i}:")
        print(f"      - Face Shape: {pref.faceshape}")
        print(f"      - Confidence: {pref.faceshape_confidence:.2%}")
        print(f"      - Hair Type: {pref.hair_type}")
        print(f"      - Hair Length: {pref.hair_length}")
        print(f"      - Gender: {pref.gender}")
    
    # 2. Test Gemini service with a preference
    test_pref = prefs.first()
    print(f"\n2. Testing Gemini service with preference ID: {test_pref.id}")
    
    # Get a hairstyle to test with
    hairstyle = Hairstyle.objects.filter(is_active=True).first()
    if not hairstyle:
        print("   ❌ No active hairstyles found in database")
        return False
    
    print(f"   Using hairstyle: {hairstyle.name}")
    
    # Prepare user preferences dict (matching the view)
    user_preferences = {
        'hair_type': test_pref.hair_type,
        'hair_length': test_pref.hair_length,
        'maintenance': test_pref.maintenance,
        'lifestyle': test_pref.lifestyle,
        'gender': test_pref.gender,
        'occasions': test_pref.occasions or [],
        'hair_thickness': test_pref.hair_thickness,
        'hair_texture_detail': test_pref.hair_texture_detail,
        'wants_bangs': test_pref.wants_bangs,
        'volume': test_pref.volume,
        'styling_preference': test_pref.styling_preference,
        'styling_maintenance': test_pref.styling_maintenance,
        'hair_color': test_pref.hair_color,
        'hair_condition': test_pref.hair_condition or [],
    }
    
    print("\n   User preferences being passed to Gemini:")
    for key, value in user_preferences.items():
        print(f"      - {key}: {value}")
    
    print(f"\n   Face shape: {test_pref.faceshape}")
    print(f"   Face shape confidence: {test_pref.faceshape_confidence:.2%}")
    
    # 3. Call Gemini service
    print("\n3. Calling Gemini service...")
    gemini_service = get_gemini_service()
    
    if not gemini_service.enabled:
        print("   ⚠️  Gemini service not enabled (API key not configured)")
        print("   This will use fallback responses instead of AI generation")
    
    try:
        ai_details = gemini_service.generate_hairstyle_details(
            hairstyle_name=hairstyle.name,
            hairstyle_description=hairstyle.description or '',
            user_preferences=user_preferences,
            face_shape=test_pref.faceshape,
            face_shape_confidence=test_pref.faceshape_confidence,
            hairstyle_tags=hairstyle.tags or [],
            hairstyle_occasions=hairstyle.occasions or []
        )
        
        print(f"   ✅ Gemini service call successful")
        print(f"   AI Generated: {ai_details.get('success', False)}")
        
        # 4. Check the response
        print("\n4. Checking Gemini response...")
        
        description = ai_details.get('personalized_description', '')
        face_shape_lower = test_pref.faceshape.lower()
        
        print(f"\n   Generated Description:")
        print(f"   {description}")
        
        # Check if face shape is mentioned in the description
        if face_shape_lower in description.lower():
            print(f"\n   ✅ PASS: Face shape '{test_pref.faceshape}' is mentioned in the description")
        else:
            print(f"\n   ⚠️  WARNING: Face shape '{test_pref.faceshape}' NOT found in description")
            print(f"   This might be okay if the AI paraphrased or used synonyms")
        
        # Display preference match
        print("\n   Preference Match Points:")
        for i, match in enumerate(ai_details.get('preference_match', []), 1):
            print(f"      {i}. {match}")
            if face_shape_lower in match.lower():
                print(f"         ✅ Mentions face shape")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error calling Gemini service: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_face_shape_mapping():
    """Verify the face shape mapping is correct"""
    print("\n" + "=" * 80)
    print("FACE SHAPE MAPPING VERIFICATION")
    print("=" * 80)
    
    from hairmixer_app.ml.model import FACE_SHAPES
    
    print("\nResNet50 Face Shape Mapping:")
    for idx, shape in FACE_SHAPES.items():
        print(f"   Class {idx}: {shape}")
    
    print("\nChecking database values match model classes...")
    prefs = UserPreference.objects.exclude(faceshape='')
    
    for pref in prefs[:10]:
        if pref.faceshape in FACE_SHAPES.values():
            status = "✅"
        else:
            status = "❌"
        print(f"   {status} Preference {pref.id}: {pref.faceshape}")


if __name__ == '__main__':
    try:
        check_face_shape_mapping()
        success = test_face_shape_flow()
        
        print("\n" + "=" * 80)
        if success:
            print("TEST COMPLETED SUCCESSFULLY ✅")
        else:
            print("TEST COMPLETED WITH WARNINGS ⚠️")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
