# HairMixer Model Data Flow - Current vs Expected

## Current Flow (with Issues)

```
User Preferences
│
├─ gender: 'female'
├─ hair_type: 'wavy'
├─ hair_length: 'medium'
├─ faceshape: 'oval'
├─ maintenance: 'low'
├─ lifestyle: 'casual'
├─ volume: 'medium'
├─ styling_maintenance: 'low'
├─ styling_preference: 'natural'
├─ hair_condition: 'good'
├─ hair_thickness: 'medium'
├─ hair_texture_detail: 'normal'
├─ wants_bangs: True
└─ occasions: ['work', 'casual']
      │
      ▼
_encode_preferences() - GENERATES 21 FEATURES IN WRONG ORDER
      │
      ├─ Feature 1: gender = 1        ← Model expects faceshape here!
      ├─ Feature 2: hair_type = 1     ← Model expects gender here!
      ├─ Feature 3: hair_length = 2   ← Model expects hair_type here!
      ├─ Feature 4: faceshape = 0     ← Model expects hair_length here!
      ├─ Feature 5: maintenance = 0   ← Model expects volume here!
      ├─ Feature 6: lifestyle = 3     ✓ Correct!
      ├─ Feature 7: volume = 2        ← Model expects maintenance here!
      ├─ Feature 8: styling_maintenance = 0  ✓ Correct!
      ├─ Feature 9: styling_preference = 0   ← Model expects occasions here!
      ├─ Feature 10: hair_condition = 2      ← Model expects hair_color here!
      ├─ Feature 11: hair_thickness = 1      ← Model expects hair_texture_detail here!
      ├─ Feature 12: hair_texture_detail = 4 ← Model expects styling_preference here!
      ├─ Feature 13: wants_bangs = 1         ← Model expects hair_condition here!
      ├─ Feature 14: work = 1                ← Model expects hair_thickness here!
      ├─ Feature 15: casual = 1              ← Model expects wants_bangs here!
      ├─ Feature 16: formal = 0              ← EXTRA! Gets truncated
      ├─ Feature 17: date = 0                ← EXTRA! Gets truncated
      ├─ Feature 18: exercise = 0            ← EXTRA! Gets truncated
      ├─ Feature 19: travel = 0              ← EXTRA! Gets truncated
      ├─ Feature 20: party = 0               ← EXTRA! Gets truncated
      └─ Feature 21: wedding = 0             ← EXTRA! Gets truncated
      │
      ▼
TRUNCATE to 15 features (WARNING logged)
      │
      ▼
RandomForestClassifier.predict_proba()
      │
      ├─ Receives WRONG values for most features
      ├─ Makes prediction based on incorrect data
      └─ Returns probabilities for 144 classes
      │
      ▼
Top prediction: Class 61 (french_bob) - 1.66% confidence
      │
      ▼
_get_styles_by_family(61)
      │
      └─ Maps class 61 to category... FAILS!
          (Only classes 0-25 have mappings)
      │
      ▼
[WARNING] Unknown family 61, using as-is
      │
      └─ Tries to query database by category_id=61
          └─ No hairstyles found (invalid category)
      │
      ▼
Relaxed filter mode - searches by name patterns
      │
      └─ Returns 3 recommendations (not 10)
```

---

## Expected Flow (Fixed)

```
User Preferences (SAME)
│
├─ gender: 'female'
├─ hair_type: 'wavy'
├─ hair_length: 'medium'
├─ faceshape: 'oval'
├─ maintenance: 'low'
├─ lifestyle: 'casual'
├─ volume: 'medium'
├─ styling_maintenance: 'low'
├─ styling_preference: 'natural'
├─ hair_condition: 'good'
├─ hair_thickness: 'medium'
├─ hair_texture_detail: 'normal'
├─ hair_color: 'brown'          ← ADD THIS!
├─ wants_bangs: True
└─ occasions: ['work', 'casual']
      │
      ▼
_encode_preferences() - GENERATES 15 FEATURES IN CORRECT ORDER
      │
      ├─ Feature 1: faceshape = 0            ✓ CORRECT ORDER!
      ├─ Feature 2: gender = 1               ✓
      ├─ Feature 3: hair_type = 1            ✓
      ├─ Feature 4: hair_length = 2          ✓
      ├─ Feature 5: volume = 2               ✓
      ├─ Feature 6: lifestyle = 3            ✓
      ├─ Feature 7: maintenance = 0          ✓
      ├─ Feature 8: styling_maintenance = 0  ✓
      ├─ Feature 9: occasions = 2            ✓ (count or encoded value)
      ├─ Feature 10: hair_color = 1          ✓ (brown)
      ├─ Feature 11: hair_texture_detail = 4 ✓
      ├─ Feature 12: styling_preference = 0  ✓
      ├─ Feature 13: hair_condition = 2      ✓
      ├─ Feature 14: hair_thickness = 1      ✓
      └─ Feature 15: wants_bangs = 1         ✓
      │
      ▼
RandomForestClassifier.predict_proba()
      │
      ├─ Receives CORRECT values for all features
      ├─ Makes prediction based on proper training data
      └─ Returns probabilities for 144 classes
      │
      ▼
Top prediction: Class 41 (curly_medium) - 25% confidence
(Example - actual prediction depends on data)
      │
      ▼
_get_styles_by_class(41)
      │
      └─ Use label_encoder.json:
          Class 41 → "curly_medium"
      │
      ▼
Query database for hairstyle name "curly medium"
      │
      ├─ SELECT * FROM hairstyles
      ├─ WHERE name ILIKE '%curly medium%'
      ├─ AND suitable_gender IN ('female', 'unisex')
      └─ AND is_active = TRUE
      │
      ▼
Found 5 matching hairstyles
      │
      ▼
Repeat for top 10 predicted classes
      │
      ▼
Returns 10 diverse recommendations
      │
      ├─ High confidence (15-30%)
      ├─ All match user preferences
      └─ Diverse styles from top predictions
```

---

## Key Differences

### Feature Encoding
| Aspect | Current (Wrong) | Expected (Fixed) |
|--------|----------------|------------------|
| Features generated | 21 | 15 |
| Feature order | Wrong | Correct (matches training) |
| Occasions encoding | 8 binary features | 1 feature (count/encoded) |
| Hair color | Missing | Included |
| Truncation warning | Yes (every call) | No |

### Prediction Results
| Aspect | Current | Expected |
|--------|---------|----------|
| Model input | Wrong values | Correct values |
| Top confidence | 1-2% | 15-30% |
| Predictions | Unreliable | Reliable |
| Database lookup | Fails (invalid mapping) | Succeeds (name-based) |
| Results returned | 3 (fallback) | 10 (as requested) |

### System Behavior
| Aspect | Current | Expected |
|--------|---------|----------|
| Warnings logged | 20+ per request | 0 |
| Category mapping | Fails for 118/144 classes | Works for all 144 |
| Recommendation quality | Poor (wrong features) | Good (correct features) |
| User experience | Inconsistent results | Consistent, relevant results |

---

## Feature Importance Reminder

Only 3 features really matter (based on model training):
- **hair_length**: 41.4% importance ⭐⭐⭐
- **gender**: 37.9% importance ⭐⭐⭐
- **faceshape**: 20.7% importance ⭐⭐

All other features: <0.01% importance

**Implication:** Even with wrong feature order, if these 3 are roughly correct, model may still work somewhat (explains why it doesn't completely fail).

---

*See MODEL_INTEGRATION_ANALYSIS.md for implementation details*
