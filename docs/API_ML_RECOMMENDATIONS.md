# HairMixer API Documentation

## ML Recommendations API

### Overview

The ML Recommendations API provides intelligent hairstyle suggestions using a trained Random Forest classifier combined with ResNet50 face shape detection.

---

## Endpoints

### 1. Get ML Recommendations

Generate top 10 hairstyle recommendations using ML model.

**Endpoint:** `POST /api/recommend/ml/`

**Authentication:** Not required (public endpoint)

**Rate Limiting:** 20 requests per hour per user

#### Request

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "preference_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| preference_id | UUID | Yes | User preference UUID from preferences endpoint |

#### Response

**Success (200):**
```json
{
  "recommendation_count": 10,
  "recommendations": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "name": "Layered Bob with Side-Swept Bangs",
      "description": "A versatile bob cut with soft layers that frame the face beautifully. Perfect for oval and heart-shaped faces.",
      "image_url": "/media/hairstyles/2025/10/bob_layers.jpg",
      "category": "Bob Cuts",
      "hairstyle_family": "Bob",
      "difficulty": "medium",
      "estimated_time": 30,
      "maintenance": "medium",
      "tags": [
        "professional",
        "versatile",
        "modern",
        "bangs"
      ],
      "match_score": 0.892,
      "confidence": 89.2
    },
    {
      "id": "223e4567-e89b-12d3-a456-426614174001",
      "name": "Beach Waves",
      "description": "Effortless wavy style perfect for casual occasions...",
      "image_url": "/media/hairstyles/2025/10/beach_waves.jpg",
      "category": "Wavy Styles",
      "hairstyle_family": "Waves",
      "difficulty": "easy",
      "estimated_time": 15,
      "maintenance": "low",
      "tags": [
        "casual",
        "natural",
        "easy"
      ],
      "match_score": 0.856,
      "confidence": 85.6
    }
    // ... 8 more recommendations
  ],
  "model_used": "hairstyle_family_model",
  "faceshape": "oval",
  "faceshape_confidence": 0.924
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| recommendation_count | integer | Number of recommendations returned (max 10) |
| recommendations | array | List of hairstyle recommendations |
| recommendations[].id | UUID | Hairstyle unique identifier |
| recommendations[].name | string | Hairstyle name |
| recommendations[].description | string | Detailed description |
| recommendations[].image_url | string | Path to hairstyle image |
| recommendations[].category | string | Hairstyle category |
| recommendations[].hairstyle_family | string | Predicted family by ML model |
| recommendations[].difficulty | string | Styling difficulty (easy, medium, hard, professional) |
| recommendations[].estimated_time | integer | Styling time in minutes |
| recommendations[].maintenance | string | Maintenance level (low, medium, high) |
| recommendations[].tags | array | Descriptive tags |
| recommendations[].match_score | float | ML model confidence (0.0-1.0) |
| recommendations[].confidence | float | Confidence percentage (0-100) |
| model_used | string | ML model identifier |
| faceshape | string | Detected face shape |
| faceshape_confidence | float | Face shape detection confidence |

**Error Responses:**

**400 Bad Request:**
```json
{
  "error": "preference_id is required"
}
```

**404 Not Found:**
```json
{
  "error": "User preferences not found"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Failed to generate ML recommendations",
  "details": "Error message details"
}
```

---

### 2. Save User Preferences

Save comprehensive user preferences for ML recommendations.

**Endpoint:** `POST /api/preferences/`

**Authentication:** Optional (can be anonymous)

#### Request

**Body:**
```json
{
  "faceshape": "oval",
  "gender": "female",
  "hair_type": "wavy",
  "hair_length": "medium",
  "hair_color": "brown",
  "volume": "medium",
  "hair_thickness": "medium",
  "hair_texture_detail": "normal",
  "styling_preference": "casual",
  "hair_condition": "good",
  "wants_bangs": true,
  "lifestyle": "professional",
  "maintenance": "medium",
  "styling_maintenance": "medium",
  "occasions": ["work", "casual", "date"],
  "hairstyle_family": "",
  "hairstyle_name": ""
}
```

**Parameters:**

| Field | Type | Required | Values |
|-------|------|----------|--------|
| faceshape | string | No* | oval, round, square, heart, oblong, diamond, triangle |
| gender | string | No | male, female, nb, other |
| hair_type | string | Yes | straight, wavy, curly, coily |
| hair_length | string | Yes | pixie, short, medium, long, extra_long |
| hair_color | string | No | Any color |
| volume | string | Yes | flat, light, medium, high |
| hair_thickness | string | Yes | fine, medium, thick, very_thick |
| hair_texture_detail | string | Yes | smooth, coarse, silky, frizzy, normal |
| styling_preference | string | Yes | natural, casual, polished, glamorous, edgy |
| hair_condition | string | No | excellent, good, fair, damaged |
| wants_bangs | boolean | No | true, false |
| lifestyle | string | Yes | active, professional, creative, casual |
| maintenance | string | Yes | low, medium, high |
| styling_maintenance | string | No | low, medium, high |
| occasions | array | Yes | work, casual, formal, date, exercise, travel, party, wedding |
| hairstyle_family | string | No | Any hairstyle family |
| hairstyle_name | string | No | Any hairstyle name |

*Auto-populated from face detection if not provided

#### Response

**Success (200/201):**
```json
{
  "success": true,
  "preference_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Preferences saved successfully"
}
```

---

### 3. Upload Image

Upload image for face shape detection and analysis.

**Endpoint:** `POST /api/upload/`

**Authentication:** Optional

**Content-Type:** `multipart/form-data`

#### Request

**Form Data:**
```
image: <file> (required)
```

**File Requirements:**
- Formats: JPG, JPEG, PNG
- Max size: 10 MB
- Min dimensions: 200x200 pixels
- Must contain a clearly visible face

#### Response

**Success (200/201):**
```json
{
  "success": true,
  "image_id": "789e4567-e89b-12d3-a456-426614174000",
  "image_url": "/media/uploads/2025/10/user_image.jpg",
  "face_detected": true,
  "face_count": 1,
  "face_shape": {
    "shape": "oval",
    "confidence": 0.924
  },
  "facial_features": {
    "jawline_width": 0.72,
    "cheek_width": 0.68,
    "forehead_width": 0.70
  },
  "processing_status": "completed"
}
```

---

## Complete Workflow Example

### Step 1: Upload Image

```javascript
const formData = new FormData();
formData.append('image', imageFile);

const uploadResponse = await fetch('http://localhost:8000/api/upload/', {
  method: 'POST',
  body: formData
});

const uploadData = await uploadResponse.json();
console.log('Detected face shape:', uploadData.face_shape);
```

### Step 2: Save Preferences

```javascript
const preferences = {
  faceshape: uploadData.face_shape.shape,  // Auto-filled
  hair_type: 'wavy',
  hair_length: 'medium',
  volume: 'medium',
  hair_thickness: 'medium',
  hair_texture_detail: 'normal',
  styling_preference: 'casual',
  wants_bangs: true,
  lifestyle: 'professional',
  maintenance: 'medium',
  occasions: ['work', 'casual']
};

const prefsResponse = await fetch('http://localhost:8000/api/preferences/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(preferences)
});

const prefsData = await prefsResponse.json();
```

### Step 3: Get ML Recommendations

```javascript
const mlResponse = await fetch('http://localhost:8000/api/recommend/ml/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    preference_id: prefsData.preference_id
  })
});

const recommendations = await mlResponse.json();
console.log('Top 10 recommendations:', recommendations.recommendations);
```

---

## Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request body format and required fields |
| 404 | Not Found | Verify resource IDs exist |
| 413 | Payload Too Large | Reduce image file size (max 10MB) |
| 429 | Too Many Requests | Wait before retrying (rate limit: 20/hour) |
| 500 | Internal Server Error | Contact support or check logs |

---

## Rate Limiting

All endpoints have rate limiting:

- **Authenticated users**: 20 requests/hour for recommendations, 10 uploads/hour
- **Anonymous users**: 10 requests/hour for recommendations, 5 uploads/hour

Rate limit headers:
```
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 15
X-RateLimit-Reset: 1696348800
```

---

## Best Practices

1. **Always provide all required fields** for best ML results
2. **Use high-quality face images** for accurate face shape detection
3. **Handle faceshape auto-population** - face shape will be filled from upload if not provided
4. **Cache recommendations** - results don't change frequently for same preferences
5. **Implement retry logic** for 500 errors with exponential backoff
6. **Validate file size** before upload to avoid 413 errors

---

## SDK Examples

### Python

```python
import requests

# Upload image
with open('user_photo.jpg', 'rb') as f:
    upload_response = requests.post(
        'http://localhost:8000/api/upload/',
        files={'image': f}
    )
upload_data = upload_response.json()

# Save preferences
preferences = {
    'faceshape': upload_data['face_shape']['shape'],
    'hair_type': 'wavy',
    'hair_length': 'medium',
    'volume': 'medium',
    'hair_thickness': 'medium',
    'hair_texture_detail': 'normal',
    'styling_preference': 'casual',
    'lifestyle': 'professional',
    'maintenance': 'medium',
    'occasions': ['work', 'casual']
}

prefs_response = requests.post(
    'http://localhost:8000/api/preferences/',
    json=preferences
)
prefs_data = prefs_response.json()

# Get ML recommendations
ml_response = requests.post(
    'http://localhost:8000/api/recommend/ml/',
    json={'preference_id': prefs_data['preference_id']}
)
recommendations = ml_response.json()

for rec in recommendations['recommendations']:
    print(f"{rec['name']}: {rec['confidence']}% match")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function getRecommendations() {
  // Upload image
  const formData = new FormData();
  formData.append('image', fs.createReadStream('user_photo.jpg'));
  
  const uploadRes = await axios.post(
    'http://localhost:8000/api/upload/',
    formData,
    { headers: formData.getHeaders() }
  );
  
  // Save preferences
  const preferences = {
    faceshape: uploadRes.data.face_shape.shape,
    hair_type: 'wavy',
    hair_length: 'medium',
    // ... other fields
  };
  
  const prefsRes = await axios.post(
    'http://localhost:8000/api/preferences/',
    preferences
  );
  
  // Get ML recommendations
  const mlRes = await axios.post(
    'http://localhost:8000/api/recommend/ml/',
    { preference_id: prefsRes.data.preference_id }
  );
  
  return mlRes.data;
}
```

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/Matchewmau/HairMixer
- Documentation: `/docs/ML_RECOMMENDATIONS.md`
- API Status: `GET /api/health/`

---

**Last Updated:** October 3, 2025  
**API Version:** 2.0  
**Model Version:** hairstyle_family_model v1.0
