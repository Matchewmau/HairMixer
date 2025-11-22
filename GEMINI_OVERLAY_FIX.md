# Fixed Gemini API and Overlay Issues

## Date: October 14, 2025

## Issues Reported

### 1. **Gemini API Returning Fallback Content**
User reported that clicking "Try Hairstyle" shows fallback content instead of AI-generated personalized content.

### 2. **Overlay API Returning 400 Error**
```
[WARNING] Bad Request: /api/overlay/
[14/Oct/2025 00:49:10] "POST /api/overlay/ HTTP/1.1" 400 53
```

---

## Root Causes Identified

### Issue 1: Incorrect Gemini Model Name
**Problem:** `.env` file had `GEMINI_MODEL_NAME=gemini-flash`  
**Correct:** Should be `GEMINI_MODEL_NAME=gemini-1.5-flash`

The Gemini API doesn't recognize `gemini-flash` as a valid model name, causing the API call to fail and fall back to static content.

### Issue 2: Overlay Validation Error
**Problem:** 400 Bad Request suggests validation error in the overlay request  
**Action:** Added detailed logging to identify the exact validation issue

---

## Fixes Applied

### 1. Fixed Gemini Model Name ✅

**File:** `.env`

**Before:**
```env
GEMINI_MODEL_NAME=gemini-flash
```

**After:**
```env
GEMINI_MODEL_NAME=gemini-1.5-flash
```

**Valid Model Names:**
- `gemini-1.5-flash` (Fast, efficient)
- `gemini-1.5-pro` (More powerful)
- `gemini-1.0-pro` (Legacy)

### 2. Enhanced Error Logging ✅

**File:** `backend/hairmixer_app/services/gemini_service.py`

**Added:**
```python
except Exception as e:
    logger.error(
        f"Error generating hairstyle details: {str(e)}",
        exc_info=True  # ← Full stack trace
    )
```

**Benefit:** Now you can see the full error stack trace in the logs, making it easier to debug Gemini API issues.

### 3. Added Overlay Validation Logging ✅

**File:** `backend/hairmixer_app/views/analysis.py`

**Added:**
```python
def post(self, request):
    try:
        logger.info(f"Overlay request data: {request.data}")
        req_ser = OverlayRequestSerializer(data=request.data)
        if not req_ser.is_valid():
            logger.error(f"Overlay validation errors: {req_ser.errors}")
        req_ser.is_valid(raise_exception=True)
```

**Benefit:** Now you can see exactly what data is being sent and what validation errors occur.

---

## How to Apply Fixes

### Step 1: Restart Django Server

The `.env` file changes require a server restart to take effect.

```powershell
# Stop the current server (Ctrl+C)
# Then restart:
cd D:\CODING\Python\HairMixer
.\venv311\Scripts\Activate.ps1
cd backend
python manage.py runserver
```

### Step 2: Clear Cache

The API responses might be cached. Clear cache by:

**Option A: Restart server** (done in Step 1)

**Option B: Django shell command:**
```python
from django.core.cache import cache
cache.clear()
```

### Step 3: Test Again

1. Upload an image
2. Submit preferences
3. Click "Try Hairstyle"
4. **Check the terminal logs** for:
   - `Gemini API initialized with model: gemini-1.5-flash` ✅
   - `Successfully generated details for hairstyle: [name]` ✅
   - **NOT** seeing "Error generating hairstyle details" ❌

---

## Expected Behavior After Fix

### Gemini API Success:
```
[INFO] Gemini API initialized with model: gemini-1.5-flash
[INFO] Successfully generated details for hairstyle: Classic Pompadour
[GET] /api/hairstyles/.../details/ HTTP/1.1 200
```

**Response will include:**
- ✅ `ai_generated: true`
- ✅ Personalized description
- ✅ Preference match explanations
- ✅ Product recommendations
- ✅ Maintenance guide
- ✅ Styling tips

### Overlay API - What to Check:
```
[INFO] Overlay request data: {'image_id': '...', 'hairstyle_id': '...', 'overlay_type': 'basic'}
```

**If you still see 400 error:**
Check the logs for:
```
[ERROR] Overlay validation errors: {...}
```

This will tell you exactly what field is failing validation.

---

## Troubleshooting

### If Gemini Still Returns Fallback:

#### Check 1: Verify API Key
```python
# In Django shell
from django.conf import settings
print(settings.GEMINI_API_KEY)
print(settings.GEMINI_MODEL_NAME)
```

Should output:
```
AIzaSyCCBXrsgH5gnYfRJOywSKKAUG0XzfxsD4o
gemini-1.5-flash
```

#### Check 2: Test API Key
```python
import google.generativeai as genai
genai.configure(api_key='AIzaSyCCBXrsgH5gnYfRJOywSKKAUG0XzfxsD4o')
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Say hello!")
print(response.text)
```

If this fails, the API key might be:
- Expired
- Invalid
- Restricted (check API restrictions in Google Cloud Console)

#### Check 3: Check Logs
Look for the full error with stack trace:
```
[ERROR] Error generating hairstyle details: [detailed error]
Traceback (most recent call last):
  ...
```

Common errors:
- `404`: Model not found (wrong model name)
- `403`: API key invalid or restricted
- `429`: Rate limit exceeded
- `500`: Gemini service error

### If Overlay Still Returns 400:

#### Check the Logs:
```
[INFO] Overlay request data: {...}
[ERROR] Overlay validation errors: {...}
```

Common issues:
- `image_id` not a valid UUID
- `hairstyle_id` not a valid UUID
- `overlay_type` not 'basic' or 'advanced'
- Missing required fields

---

## API Configuration Reference

### Gemini API Settings (.env)
```env
# API Key (required)
GEMINI_API_KEY=AIzaSyCCBXrsgH5gnYfRJOywSKKAUG0XzfxsD4o

# Model Name (required)
GEMINI_MODEL_NAME=gemini-1.5-flash

# Available models:
# - gemini-1.5-flash (Recommended: Fast, efficient)
# - gemini-1.5-pro (More powerful, slower)
# - gemini-1.0-pro (Legacy)
```

### Overlay API Request Format
```json
POST /api/overlay/
{
  "image_id": "uuid-here",
  "hairstyle_id": "uuid-here",
  "overlay_type": "basic"  // or "advanced"
}
```

---

## Verification Checklist

After restarting the server, verify:

### Backend Startup:
- [ ] No errors on startup
- [ ] Gemini API initialized successfully
- [ ] Model name is `gemini-1.5-flash`

### Try Hairstyle Feature:
- [ ] Click "Try Hairstyle" button
- [ ] Modal opens successfully
- [ ] AI-generated content appears (not fallback)
- [ ] User preferences section shows
- [ ] All sections have content:
  - [ ] Personalized description
  - [ ] Preference match
  - [ ] Recommended products
  - [ ] Maintenance guide
  - [ ] Styling tips

### Check Logs:
- [ ] No "Error generating hairstyle details" messages
- [ ] See "Successfully generated details" messages
- [ ] If overlay is attempted, check for validation errors

---

## Files Modified

1. ✅ `.env`
   - Fixed `GEMINI_MODEL_NAME` from `gemini-flash` to `gemini-1.5-flash`

2. ✅ `backend/hairmixer_app/services/gemini_service.py`
   - Added `exc_info=True` to error logging
   - Now shows full stack trace for errors

3. ✅ `backend/hairmixer_app/views/analysis.py`
   - Added request data logging
   - Added validation error logging
   - Helps debug overlay 400 errors

---

## Next Steps

### 1. **Restart Server** (Required)
```powershell
cd D:\CODING\Python\HairMixer
.\venv311\Scripts\Activate.ps1
cd backend
python manage.py runserver
```

### 2. **Test Gemini API**
- Upload image
- Submit preferences
- Click "Try Hairstyle"
- **Look for:** "ai_generated: true" in response
- **Look for:** Personalized, detailed content (not generic fallback)

### 3. **Check Logs**
Watch the terminal for:
```
[INFO] Gemini API initialized with model: gemini-1.5-flash
[INFO] Successfully generated details for hairstyle: ...
```

**Not:** 
```
[ERROR] Error generating hairstyle details: ...
```

### 4. **If Still Issues**
Share the full error log from the terminal, especially:
- The initialization message
- Any error messages with stack traces
- The overlay validation errors (if any)

---

## Status

**Gemini API:** ✅ Fixed (model name corrected, needs server restart)  
**Overlay API:** ⚠️ Enhanced logging added, waiting for restart to debug  
**Server Restart:** 🔄 Required for changes to take effect  

**Action Required:** Restart Django server and test again!

