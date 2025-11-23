"""
Test script to check if Gemini AI Service is working properly
"""
import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.services.gemini_service import get_gemini_service

def test_gemini_service():
    """Test the Gemini service initialization and functionality"""
    
    print("=" * 60)
    print("Testing Gemini AI Service")
    print("=" * 60)
    
    # Get service instance
    service = get_gemini_service()
    
    # Check if service is enabled
    print(f"\n1. Service Enabled: {service.enabled}")
    print(f"   API Key Set: {'Yes' if service.api_key else 'No'}")
    print(f"   Model Name: {service.model_name}")
    
    if not service.enabled:
        print("\n❌ Gemini service is DISABLED")
        print("   Reason: No API key configured")
        print("\n   To enable Gemini AI:")
        print("   1. Get an API key from: https://makersuite.google.com/app/apikey")
        print("   2. Set environment variable: GEMINI_API_KEY=your_key_here")
        print("   3. Or create a .env file in backend folder with:")
        print("      GEMINI_API_KEY=your_key_here")
        return False
    
    print("\n✅ Gemini service is ENABLED")
    
    # Test hairstyle detail generation
    print("\n2. Testing AI Content Generation...")
    
    test_data = {
        'hairstyle_name': 'Bob Cut',
        'hairstyle_description': 'A classic short hairstyle that ends around jaw length',
        'user_preferences': {
            'hair_type': 'straight',
            'hair_length': 'short',
            'maintenance': 'low',
            'lifestyle': 'professional',
            'gender': 'female',
            'occasions': ['work', 'casual'],
            'hair_thickness': 'medium',
            'hair_texture_detail': 'fine',
            'wants_bangs': False,
        },
        'face_shape': 'oval',
        'face_shape_confidence': 0.85,
        'hairstyle_tags': ['short', 'classic', 'professional'],
        'hairstyle_occasions': ['work', 'formal', 'casual']
    }
    
    try:
        result = service.generate_hairstyle_details(**test_data)
        
        print(f"\n   Generation Success: {result.get('success', False)}")
        
        if result.get('success'):
            print("\n   ✅ AI Content Generated Successfully!")
            print("\n   Content Sections:")
            print(f"   - Personalized Description: {len(result.get('personalized_description', ''))} chars")
            print(f"   - Preference Matches: {len(result.get('preference_match', []))} items")
            print(f"   - Product Recommendations: {len(result.get('products', []))} items")
            print(f"   - Maintenance Steps: {len(result.get('maintenance_guide', []))} items")
            print(f"   - Styling Tips: {len(result.get('styling_tips', []))} items")
            
            print("\n   Sample Output:")
            print(f"   Description: {result.get('personalized_description', 'N/A')}")
            print(f"\n   Preference Match Examples:")
            for i, match in enumerate(result.get('preference_match', [])[:2], 1):
                print(f"   {i}. {match}")
            print(f"\n   Product Examples:")
            for i, product in enumerate(result.get('products', [])[:2], 1):
                print(f"   {i}. {product}")
            
            return True
        else:
            print("\n   ⚠️  Using Fallback Content (AI generation failed)")
            return False
            
    except Exception as e:
        print(f"\n   ❌ Error during generation: {str(e)}")
        return False

if __name__ == '__main__':
    success = test_gemini_service()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Gemini AI Service is WORKING CORRECTLY")
    else:
        print("⚠️  Gemini AI Service is using FALLBACK content")
    print("=" * 60)
