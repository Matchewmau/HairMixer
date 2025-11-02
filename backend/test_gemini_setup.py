"""
Test script to verify Gemini API configuration
"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.services.gemini_service import get_gemini_service
from hairmixer_app.overlay import AdvancedOverlayProcessor, _GEMINI_AVAILABLE

print("=" * 60)
print("GEMINI CONFIGURATION TEST")
print("=" * 60)

# Test 1: Gemini API (for text generation)
print("\n1. Testing Gemini API (Text Generation):")
print("-" * 60)
gemini_service = get_gemini_service()
print(f"   ✓ Service enabled: {gemini_service.enabled}")
print(f"   ✓ Model name: {gemini_service.model_name}")
print(f"   ✓ API key configured: {bool(gemini_service.api_key)}")
if gemini_service.api_key:
    print(f"   ✓ API key format: {gemini_service.api_key[:10]}...")

# Test 2: Gemini WebAPI (for overlay generation)
print("\n2. Testing Gemini WebAPI (Overlay Generation):")
print("-" * 60)
processor = AdvancedOverlayProcessor()
print(f"   ✓ Gemini WebAPI available: {_GEMINI_AVAILABLE}")
print(f"   ✓ AI overlay enabled: {processor.ai_enabled}")
print(f"   ✓ 1PSID configured: {bool(processor.gemini_sid)}")
print(f"   ✓ 1PSIDTS configured: {bool(processor.gemini_sidts)}")
print(f"   ✓ Model: {processor.gemini_model}")

# Test 3: Overall status
print("\n3. Overall Status:")
print("-" * 60)
text_gen_ok = gemini_service.enabled
overlay_ok = (
    _GEMINI_AVAILABLE 
    and processor.ai_enabled 
    and bool(processor.gemini_sid) 
    and bool(processor.gemini_sidts)
)

if text_gen_ok and overlay_ok:
    print("   ✅ ALL SYSTEMS GO!")
    print("   - Text generation: READY")
    print("   - AI overlay generation: READY")
elif text_gen_ok:
    print("   ⚠️  PARTIAL SETUP")
    print("   - Text generation: READY")
    print("   - AI overlay generation: NOT CONFIGURED")
    print("   → Configure GEMINI_SECURE_1PSID and GEMINI_SECURE_1PSIDTS")
elif overlay_ok:
    print("   ⚠️  PARTIAL SETUP")
    print("   - Text generation: NOT CONFIGURED")
    print("   - AI overlay generation: READY")
    print("   → Configure GEMINI_API_KEY")
else:
    print("   ❌ NOT CONFIGURED")
    print("   → Configure all Gemini credentials in .env file")

print("\n" + "=" * 60)
print("Test completed. Restart Django server to apply any changes.")
print("=" * 60)
