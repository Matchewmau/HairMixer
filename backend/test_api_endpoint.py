"""
Test the hairstyle detail endpoint with real data
"""
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000/api"

def test_hairstyle_detail_endpoint():
    print("=" * 70)
    print("Testing Hairstyle Detail API Endpoint with Gemini AI")
    print("=" * 70)
    
    # First, let's get a list of hairstyles to test with
    print("\n1. Fetching available hairstyles...")
    
    try:
        response = requests.get(f"{BASE_URL}/hairstyles/")
        response.raise_for_status()
        data = response.json()
        
        # Handle different response formats
        if isinstance(data, dict):
            hairstyles = data.get('results', []) or data.get('data', [])
            if not hairstyles and 'id' in data:
                hairstyles = [data]
        else:
            hairstyles = data
        
        if not hairstyles:
            print("   ❌ No hairstyles found in database")
            return
        
        print(f"   ✅ Found {len(hairstyles)} hairstyles")
        
        # Get the first hairstyle
        test_hairstyle = hairstyles[0]
        hairstyle_id = test_hairstyle['id']
        hairstyle_name = test_hairstyle['name']
        
        print(f"   Testing with: {hairstyle_name} (ID: {hairstyle_id})")
        
    except Exception as e:
        print(f"   ❌ Error fetching hairstyles: {str(e)}")
        return
    
    # Test the detail endpoint without user preferences
    print(f"\n2. Testing detail endpoint (no preferences)...")
    
    try:
        response = requests.get(f"{BASE_URL}/hairstyles/{hairstyle_id}/details/")
        response.raise_for_status()
        details = response.json()
        
        print(f"   ✅ API Response received")
        print(f"\n   Response Structure:")
        print(f"   - hairstyle: {'✓' if 'hairstyle' in details else '✗'}")
        print(f"   - face_shape: {details.get('face_shape', 'N/A')}")
        print(f"   - ai_generated: {details.get('ai_generated', False)}")
        print(f"   - personalized_description: {'✓' if details.get('personalized_description') else '✗'} ({len(details.get('personalized_description', ''))} chars)")
        print(f"   - preference_match: {len(details.get('preference_match', []))} items")
        print(f"   - recommended_products: {len(details.get('recommended_products', []))} items")
        print(f"   - maintenance_guide: {len(details.get('maintenance_guide', []))} items")
        print(f"   - styling_tips: {len(details.get('styling_tips', []))} items")
        
        print(f"\n   Sample Content:")
        print(f"   Description: {details.get('personalized_description', 'N/A')[:150]}...")
        
        if details.get('preference_match'):
            print(f"\n   First Preference Match:")
            print(f"   • {details['preference_match'][0]}")
        
        if details.get('recommended_products'):
            print(f"\n   First Product:")
            print(f"   • {details['recommended_products'][0]}")
        
        print("\n" + "=" * 70)
        if details.get('ai_generated'):
            print("✅ GEMINI AI IS WORKING - Generating personalized content!")
        else:
            print("⚠️  Using fallback content - Gemini AI may not be working")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to backend server")
        print("   Make sure Django server is running: python manage.py runserver")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

if __name__ == '__main__':
    test_hairstyle_detail_endpoint()
