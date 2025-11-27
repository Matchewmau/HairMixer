# Try Hairstyle Feature - Implementation Guide

## Overview

The "Try Hairstyle" feature provides users with an immersive, detailed preview of recommended hairstyles. When a user clicks "Try Hairstyle" on a recommendation card, a modal opens displaying:

1. **Visual Preview**: Overlay of the hairstyle on the user's photo
2. **Personalized Description**: AI-generated description tailored to their face shape and preferences
3. **Preference Match**: How the style aligns with their stated preferences
4. **Product Recommendations**: Specific products needed for the style
5. **Maintenance Guide**: Step-by-step maintenance instructions
6. **Styling Tips**: Professional tips for achieving and maintaining the look
7. **Navigation**: Back/Next buttons to browse through all recommended hairstyles

## Architecture

### Backend Components

#### 1. Gemini AI Service (`backend/hairmixer_app/services/gemini_service.py`)

**Purpose**: Integrates with Google's Gemini API to generate personalized hairstyle information.

**Key Features**:
- Generates AI-powered personalized descriptions
- Creates preference match explanations
- Suggests appropriate products
- Provides maintenance guides
- Offers professional styling tips
- Graceful fallback when API is unavailable

**Configuration**:
```python
# Environment variables needed
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-pro  # or gemini-pro-vision
```

**Usage Example**:
```python
from hairmixer_app.services.gemini_service import get_gemini_service

service = get_gemini_service()
details = service.generate_hairstyle_details(
    hairstyle_name="Layered Bob",
    hairstyle_description="A versatile bob with layers...",
    user_preferences={
        'hair_type': 'wavy',
        'maintenance': 'low',
        # ... other preferences
    },
    face_shape='oval',
    face_shape_confidence=0.92
)
```

#### 2. API View (`backend/hairmixer_app/views/hairstyle_detail_view.py`)

**Endpoint**: `GET /api/hairstyles/<uuid:hairstyle_id>/details/`

**Query Parameters**:
- `preference_id` (optional): UUID of user preferences
- `image_id` (optional): UUID of uploaded image

**Response Structure**:
```json
{
  "hairstyle": {
    "id": "uuid",
    "name": "Layered Bob",
    "description": "...",
    "image_url": "...",
    // ... other hairstyle fields
  },
  "face_shape": "oval",
  "face_shape_confidence": 0.92,
  "ai_generated": true,
  "personalized_description": "This layered bob will beautifully frame your oval face...",
  "preference_match": [
    "Matches your low maintenance preference",
    "Perfect for your wavy hair type",
    "Suits your casual lifestyle"
  ],
  "recommended_products": [
    "Volumizing Mousse - Adds body and texture",
    "Texturizing Spray - Creates definition",
    // ...
  ],
  "maintenance_guide": [
    "Wash hair 2-3 times per week",
    "Apply styling products to damp hair",
    // ...
  ],
  "styling_tips": [
    "Work with your natural wave pattern",
    "Use a diffuser for added volume",
    // ...
  ]
}
```

**Caching**: Results are cached for 1 hour per hairstyle/preference/image combination.

#### 3. URL Configuration

Added to `backend/hairmixer_app/urls.py`:
```python
path(
    'hairstyles/<uuid:hairstyle_id>/details/',
    views.HairstyleDetailWithAIView.as_view(),
    name='hairstyle_detail_ai',
),
```

### Frontend Components

#### 1. Updated Results Page (`frontend/src/pages/Results.js`)

**New State Variables**:
```javascript
const [showTryHairstyleModal, setShowTryHairstyleModal] = useState(false);
const [currentHairstyleIndex, setCurrentHairstyleIndex] = useState(0);
const [hairstyleDetails, setHairstyleDetails] = useState(null);
const [loadingDetails, setLoadingDetails] = useState(false);
const [detailsError, setDetailsError] = useState('');
```

**Key Functions**:

1. `handleTryHairstyle(index)`: Opens modal and loads details
2. `handleNextHairstyle()`: Navigate to next recommendation
3. `handlePrevHairstyle()`: Navigate to previous recommendation
4. `closeTryHairstyleModal()`: Close modal and cleanup

#### 2. API Service (`frontend/src/services/api.js`)

**New Method**:
```javascript
async getHairstyleDetailsWithAI(hairstyleId, preferenceId = null, imageId = null) {
  const params = new URLSearchParams();
  if (preferenceId) params.append('preference_id', preferenceId);
  if (imageId) params.append('image_id', imageId);
  const query = params.toString();
  return this.request(`/hairstyles/${hairstyleId}/details/${query ? '?' + query : ''}`);
}
```

## UI/UX Design

### Modal Layout

The modal is divided into two main columns on desktop (single column on mobile):

**Left Column**:
- Hairstyle overlay on user's photo (if available)
- Face shape analysis card

**Right Column** (scrollable):
- Personalized description
- Preference match details
- Recommended products
- Maintenance guide
- Professional styling tips

**Footer**:
- Previous button (left)
- Current position indicator (center)
- Next button (right)

### Visual Design Features

1. **Color-coded sections**: Each information type has a distinct color theme
   - Purple: Personalized description
   - Green: Preference match
   - Orange: Products
   - Blue: Maintenance
   - Pink: Styling tips

2. **Icons**: Each section has an appropriate emoji/icon for quick recognition

3. **Smooth transitions**: Loading states and animations for better UX

4. **Responsive design**: Works on mobile, tablet, and desktop

## Installation & Setup

### Backend Setup

1. **Install dependencies**:
```bash
cd backend
pip install google-generativeai==0.8.3
```

2. **Configure environment variables** in `.env`:
```bash
# Get your API key from https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-pro
```

3. **Run migrations** (if any new models were added):
```bash
python manage.py migrate
```

4. **Test the API endpoint**:
```bash
python manage.py runserver
# Visit: http://localhost:8000/api/hairstyles/<hairstyle-id>/details/
```

### Frontend Setup

1. **Install dependencies** (already included in package.json)

2. **Restart development server**:
```bash
cd frontend
npm start
```

## Testing

### Backend Testing

Test the Gemini service:
```python
from hairmixer_app.services.gemini_service import get_gemini_service

service = get_gemini_service()
result = service.generate_hairstyle_details(
    hairstyle_name="Test Style",
    hairstyle_description="A test hairstyle",
    user_preferences={'hair_type': 'wavy'},
    face_shape='oval'
)
print(result)
```

Test the API endpoint:
```bash
curl -X GET "http://localhost:8000/api/hairstyles/<uuid>/details/?preference_id=<uuid>&image_id=<uuid>"
```

### Frontend Testing

1. Upload a photo
2. Set preferences
3. View recommendations
4. Click "Try Hairstyle" on any recommendation
5. Verify modal opens with all sections populated
6. Test navigation between hairstyles
7. Verify overlay generation works

## Error Handling

### Backend

1. **Missing Gemini API Key**: Falls back to template-based responses
2. **API Rate Limits**: Graceful degradation with fallback content
3. **Invalid UUIDs**: Returns 404 with clear error message
4. **Missing Data**: Provides sensible defaults

### Frontend

1. **Network Errors**: Shows error message with retry option
2. **Missing Data**: Displays available information only
3. **Loading States**: Shows spinner during data fetch
4. **Navigation Errors**: Maintains current state on failure

## Performance Optimization

1. **Caching**: API responses cached for 1 hour
2. **Lazy Loading**: Details loaded only when modal opens
3. **Debouncing**: Navigation clicks debounced to prevent rapid requests
4. **Overlay Reuse**: Overlay generated once and reused

## Future Enhancements

1. **Save Favorites**: Allow users to save styles they like
2. **Share Feature**: Share hairstyle details with friends/stylists
3. **Video Tutorials**: Add links to video tutorials for complex styles
4. **Salon Finder**: Integrate with local salons that can do the style
5. **Cost Estimates**: Add approximate cost for products/salon visit
6. **Style Variations**: Show variations of the same style
7. **Before/After Gallery**: User-submitted photos with the style
8. **AR Try-On**: Real-time AR preview using phone camera

## Troubleshooting

### "AI features will be disabled" warning

**Cause**: Missing `GEMINI_API_KEY` in environment variables

**Solution**: 
1. Get API key from https://makersuite.google.com/app/apikey
2. Add to `.env` file
3. Restart server

### Modal doesn't open

**Cause**: JavaScript error or missing recommendations

**Solution**:
1. Check browser console for errors
2. Verify recommendations exist in state
3. Check network tab for API errors

### Overlay not showing

**Cause**: Image or hairstyle ID missing

**Solution**:
1. Verify user uploaded an image
2. Check that hairstyle has valid ID
3. Verify authentication (overlay requires auth)

### Slow loading

**Cause**: Gemini API response time

**Solution**:
1. Consider caching strategy
2. Show loading indicator sooner
3. Implement progressive loading

## API Rate Limits & Costs

**Gemini API**:
- Free tier: 60 requests per minute
- Paid tier: Higher limits available
- Cost: Check current pricing at Google AI Studio

**Recommendations**:
- Implement request queuing for high traffic
- Consider upgrading to paid tier for production
- Monitor usage via Google Cloud Console

## Security Considerations

1. **API Key Protection**: Never expose in client-side code
2. **Rate Limiting**: Implement on backend to prevent abuse
3. **Input Validation**: Sanitize all user inputs
4. **CORS**: Properly configure for production
5. **Authentication**: Consider requiring auth for AI features

## Documentation Updates

This feature requires updates to:
- [ ] API_DOCS.md - Add new endpoint documentation
- [ ] README.md - Mention new feature in feature list
- [ ] OVERLAY_SETUP.md - Reference Gemini API setup
- [ ] User guide - Add screenshots and usage instructions

## Support & Resources

- **Gemini API Documentation**: https://ai.google.dev/docs
- **React Documentation**: https://react.dev/
- **Django REST Framework**: https://www.django-rest-framework.org/
- **Issue Tracker**: [Your repository issues page]
