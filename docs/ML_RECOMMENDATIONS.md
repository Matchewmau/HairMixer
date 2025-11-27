# Enhanced User Preferences & ML Recommendations

## Overview

This module implements a comprehensive user preferences system with ML-powered hairstyle recommendations. It combines computer vision (ResNet50 for face shape detection) with machine learning (Random Forest classifier) to provide personalized hairstyle suggestions.

## Architecture

```
┌─────────────────┐
│  Image Upload   │ → ResNet50 Face Shape Detection
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  9-Step Wizard  │ → User Preference Collection
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Save Preferences│ → UserPreference Model (21 features)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   ML Model      │ → hairstyle_family_model.pkl
│ (Random Forest) │    Predicts hairstyle families
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Top 10 Results │ → Hairstyle recommendations
└─────────────────┘    with confidence scores
```

## Features

### 1. Automated Face Shape Detection
- **ResNet50 Model**: Automatically detects 7 face shapes (oval, round, square, heart, oblong, diamond, triangle)
- **Confidence Scoring**: Provides confidence percentage for detection accuracy
- **Auto-population**: Face shape automatically saved to user preferences

### 2. Comprehensive User Profiling (21 Features)

#### Core Features (6)
- `gender`: Male, Female, Non-binary, Other
- `hair_type`: Straight, Wavy, Curly, Coily
- `hair_length`: Pixie, Short, Medium, Long, Extra-long
- `faceshape`: Auto-detected by ResNet50
- `maintenance`: Low, Medium, High
- `lifestyle`: Active, Professional, Creative, Casual

#### Detailed Hair Characteristics (6)
- `volume`: Flat, Light, Medium, High
- `hair_thickness`: Fine, Medium, Thick, Very Thick
- `hair_texture_detail`: Smooth, Coarse, Silky, Frizzy, Normal
- `styling_preference`: Natural, Casual, Polished, Glamorous, Edgy
- `hair_condition`: Excellent, Good, Fair, Damaged
- `styling_maintenance`: Low, Medium, High

#### Binary Features (1)
- `wants_bangs`: Boolean (True/False)

#### Multi-select Features (8)
- `occasions`: Work, Casual, Formal, Date, Exercise, Travel, Party, Wedding

### 3. ML-Powered Recommendations
- **Model**: Random Forest Classifier (`hairstyle_family_model.pkl`)
- **Input**: 21 encoded features
- **Output**: Top 10 hairstyle recommendations with confidence scores
- **No Fallbacks**: Model-only approach - returns empty list if model fails

## File Structure

```
backend/
├── hairmixer_app/
│   ├── models.py                     # UserPreference model with 17 fields
│   ├── serializers.py                # Serializer with all new fields
│   ├── views.py                      # MLRecommendView endpoint
│   ├── urls.py                       # /api/recommend/ml/ route
│   └── services/
│       ├── hairstyle_recommender.py  # ML recommendation service
│       └── recommendation_service.py # Face shape integration
│   └── ml/
│       └── models/
│           └── hairstyle_family_model.pkl  # Trained RF model

frontend/
└── src/
    ├── pages/
    │   └── UserPreferences.js        # 9-step wizard component
    └── services/
        └── api.js                    # getMLRecommendations() method
```

## API Endpoints

### POST `/api/recommend/ml/`

Generate ML-based hairstyle recommendations.

**Request:**
```json
{
  "preference_id": "uuid-string"
}
```

**Response:**
```json
{
  "recommendation_count": 10,
  "recommendations": [
    {
      "id": "hairstyle-uuid",
      "name": "Layered Bob with Bangs",
      "description": "A modern bob cut with soft layers...",
      "image_url": "/media/hairstyles/...",
      "category": "Bob",
      "hairstyle_family": "Bob",
      "difficulty": "medium",
      "estimated_time": 30,
      "maintenance": "medium",
      "tags": ["modern", "versatile", "professional"],
      "match_score": 0.89,
      "confidence": 89.0
    },
    ...
  ],
  "model_used": "hairstyle_family_model",
  "faceshape": "oval",
  "faceshape_confidence": 0.92
}
```

## Usage

### Backend Usage

```python
from hairmixer_app.services.hairstyle_recommender import HairstyleRecommender

# Initialize recommender
recommender = HairstyleRecommender()

# Prepare preferences
preferences = {
    'faceshape': 'oval',
    'gender': 'female',
    'hair_type': 'wavy',
    'hair_length': 'medium',
    'volume': 'medium',
    'hair_thickness': 'medium',
    'hair_texture_detail': 'normal',
    'styling_preference': 'casual',
    'hair_condition': 'good',
    'wants_bangs': True,
    'lifestyle': 'professional',
    'maintenance': 'medium',
    'styling_maintenance': 'medium',
    'occasions': ['work', 'casual', 'date']
}

# Get top 10 recommendations
recommendations = recommender.get_top_recommendations(
    preferences, 
    top_n=10
)

for rec in recommendations:
    print(f"{rec['name']}: {rec['confidence']}% match")
```

### Frontend Usage

```javascript
import APIService from './services/api';

// After collecting preferences
const preferences = {
  hair_type: 'wavy',
  hair_length: 'medium',
  volume: 'medium',
  // ... all other preferences
  faceshape: uploadResponse.face_shape.shape  // Auto-detected
};

// Save preferences
const response = await APIService.savePreferences(preferences);

// Get ML recommendations
const recommendations = await APIService.getMLRecommendations(
  response.preference_id
);

// Navigate to results
navigate('/results', {
  state: {
    recommendations: recommendations,
    uploadResponse: uploadResponse
  }
});
```

## Model Training

The `hairstyle_family_model.pkl` should be trained using:

### Training Data Format
```python
# Features (21 columns)
X = [
    # Core features (6)
    gender_encoded,        # 0-3
    hair_type_encoded,     # 0-3
    hair_length_encoded,   # 0-4
    faceshape_encoded,     # 0-6
    maintenance_encoded,   # 0-2
    lifestyle_encoded,     # 0-3
    
    # Detailed features (6)
    volume_encoded,        # 0-3
    styling_maintenance,   # 0-2
    styling_preference,    # 0-4
    hair_condition,        # 0-3
    hair_thickness,        # 0-3
    hair_texture_detail,   # 0-4
    
    # Binary (1)
    wants_bangs,           # 0-1
    
    # Multi-select (8)
    occasion_work,         # 0-1
    occasion_casual,       # 0-1
    occasion_formal,       # 0-1
    occasion_date,         # 0-1
    occasion_exercise,     # 0-1
    occasion_travel,       # 0-1
    occasion_party,        # 0-1
    occasion_wedding       # 0-1
]

# Target
y = hairstyle_family  # "Bob", "Layers", "Pixie", "Updo", etc.
```

### Training Script
```python
from sklearn.ensemble import RandomForestClassifier
import pickle

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
with open('hairstyle_family_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

## Database Migrations

Run migrations to apply model changes:

```bash
cd backend
python manage.py makemigrations
python manage.py migrate
```

## Testing

### Test the Complete Flow

1. **Start Backend:**
```bash
cd backend
python manage.py runserver
```

2. **Start Frontend:**
```bash
cd frontend
npm start
```

3. **Test Steps:**
   - Upload an image → Verify face shape detection
   - Complete 9 preference steps
   - Submit → Verify top 10 recommendations received
   - Check confidence scores are displayed

### Unit Tests

```python
# Test preference encoding
def test_preference_encoding():
    from hairmixer_app.services.hairstyle_recommender import HairstyleRecommender
    
    recommender = HairstyleRecommender()
    prefs = {
        'gender': 'female',
        'hair_type': 'wavy',
        # ... other prefs
    }
    
    features = recommender._encode_preferences(prefs)
    assert features is not None
    assert features.shape == (1, 21)

# Test recommendation generation
def test_get_recommendations():
    recommender = HairstyleRecommender()
    prefs = {...}  # Full preferences
    
    recommendations = recommender.get_top_recommendations(prefs, top_n=10)
    assert len(recommendations) <= 10
    assert all('match_score' in rec for rec in recommendations)
```

## Troubleshooting

### Model Not Loading
```
ERROR: Failed to load hairstyle model: [Errno 2] No such file or directory
```
**Solution:** Ensure `hairstyle_family_model.pkl` exists in `backend/hairmixer_app/ml/models/`

### Feature Count Mismatch
```
ValueError: X has 20 features but RandomForestClassifier is expecting 21
```
**Solution:** Verify all 21 features are being encoded in `_encode_preferences()`

### No Recommendations Returned
**Solution:** Check that:
1. Hairstyle data exists in database
2. Model file (`hairstyle_family_model.pkl`) is present and valid
3. User preferences match available hairstyle categories
4. Database has sufficient hairstyles with correct gender tags

## Performance Considerations

- **Model Loading**: Model loaded once on initialization, cached in memory
- **Caching**: Recommendation results can be cached by preference hash
- **Database Queries**: Optimized with indexes on face_shapes, hair_lengths
- **Fallback System**: Graceful degradation when ML unavailable

## Future Enhancements

1. **Feedback Loop**: Collect user ratings to retrain model
2. **A/B Testing**: Compare ML vs rule-based recommendations
3. **Model Versioning**: Support multiple model versions
4. **Real-time Training**: Periodically retrain with new data
5. **Explainability**: Show why each hairstyle was recommended
6. **Image-based Recommendations**: Add visual similarity matching

## References

- ResNet50 Face Shape Detection: `backend/hairmixer_app/ml/face_analyzer.py`
- Random Forest Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Django Models: https://docs.djangoproject.com/en/4.2/topics/db/models/
- React Hooks: https://react.dev/reference/react

## Contributors

Implementation Date: October 3, 2025
Model: hairstyle_family_model.pkl (Random Forest Classifier)
Face Detection: ResNet50 (80 epochs)

## License

Part of the HairMixer project.
