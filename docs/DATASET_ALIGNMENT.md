# Dataset Alignment Update

## Overview
Updated all UI options and backend model choices to match the exact values in the training dataset (`merged_preferences_max_voting.csv`).

## Date
2025-10-03

## Changes Made

### 1. Backend Model Choices (models.py)

#### GENDER_CHOICES
- **Before**: male, female, nb, other (4 options)
- **After**: male, female (2 options)
- **Reason**: Dataset only contains male/female

#### OCCASION_CHOICES
- **Before**: work, casual, formal, date, exercise, travel, party, wedding (8 options)
- **After**: work, casual, formal, party, wedding, birthday (6 options)
- **Reason**: Dataset contains these 6 specific occasions

#### LIFESTYLE_CHOICES
- **Before**: active, professional, creative, casual (4 options)
- **After**: active, moderate, relaxed (3 options)
- **Reason**: Dataset has these 3 lifestyle categories

#### LENGTH_CHOICES
- **Before**: pixie, short, medium, long, extra_long (5 options)
- **After**: short, medium, long (3 options)
- **Reason**: Dataset only has these 3 lengths

#### VOLUME_CHOICES
- **Before**: flat, light, medium, high (4 options)
- **After**: low, medium, high (3 options)
- **Reason**: Dataset uses low/medium/high terminology

#### HAIR_THICKNESS_CHOICES
- **Before**: fine, medium, thick, very_thick (4 options)
- **After**: thin, medium, thick (3 options)
- **Reason**: Dataset uses thin instead of fine, no very_thick

#### HAIR_TEXTURE_DETAIL_CHOICES
- **Before**: smooth, coarse, silky, frizzy, normal (5 options)
- **After**: fine, normal, thick (3 options)
- **Reason**: Dataset has these 3 texture categories

#### STYLING_PREFERENCE_CHOICES
- **Before**: natural, casual, polished, glamorous, edgy (5 options)
- **After**: natural, classic, elegant, trendy, edgy (5 options)
- **Reason**: Dataset uses classic/elegant/trendy instead of casual/polished/glamorous

#### HAIR_CONDITION_CHOICES
- **Before**: excellent, good, fair, damaged (4 options)
- **After**: none, damaged, dry_ends, oily_scalp, dandruff, frizzy, split_ends, thinning, sensitive_scalp (9 options)
- **Reason**: Dataset has detailed condition categories

#### HAIR_TYPE_CHOICES
- **Before**: straight, wavy, curly, coily/kinky (4 options)
- **After**: straight, wavy, curly, coily (4 options)
- **Reason**: Simplified label to just "Coily"

### 2. Frontend UI Updates (UserPreferences.js)

#### Validation Rules Updated
All `validationRules` object options updated to match dataset values for:
- hair_length: ['short', 'medium', 'long']
- volume: ['low', 'medium', 'high']
- hair_thickness: ['thin', 'medium', 'thick']
- hair_texture_detail: ['fine', 'normal', 'thick']
- lifestyle: ['active', 'moderate', 'relaxed']
- styling_preference: ['natural', 'classic', 'elegant', 'trendy', 'edgy']
- gender: ['male', 'female']
- hair_condition: ['none', 'thinning', 'split_ends', 'dry_ends', 'frizzy', 'dandruff', 'oily_scalp', 'sensitive_scalp', 'damaged']

#### UI Steps Updated
- **Step 2 (Hair Length)**: Removed pixie and extra-long options
- **Step 3 (Volume)**: Changed from flat/light to low/medium/high
- **Step 4 (Hair Thickness)**: Changed from fine/very_thick to thin/medium/thick
- **Step 5 (Hair Texture)**: Changed from 5 options to 3 (fine/normal/thick)
- **Step 6 (Lifestyle)**: Updated to active/moderate/relaxed
- **Step 8 (Styling Preference)**: Changed to natural/classic/elegant/trendy/edgy
- **Step 9 (Occasions)**: Now fetches from API with 6 dataset values

#### Face Shape Display
- **Removed**: Face shape detection still runs but is no longer displayed to users

### 3. ML Model Updates (hairstyle_recommender.py)

#### Occasion Types
- **Before**: 8 occasions (work, casual, formal, date, exercise, travel, party, wedding)
- **After**: 6 occasions (work, casual, formal, party, wedding, birthday)

#### Feature Vector Updates
- **Total Features**: Reduced from 21 to 19 features
- **Occasions Features**: Changed from 14-21 to 14-19 (6 binary features instead of 8)
- **Updated Feature Ranges**:
  - gender: 0-1 (was 0-3)
  - hair_length: 0-2 (was 0-4)
  - lifestyle: 0-2 (was 0-3)
  - volume: 0-2 (was 0-3)
  - hair_condition: 0-8 (was 0-3)
  - hair_thickness: 0-2 (was 0-3)
  - hair_texture_detail: 0-2 (was 0-4)

### 4. Database Migration

**Migration File**: `0004_alter_userpreference_gender_and_more.py`

**Changes Applied**:
- Altered field gender on userpreference
- Altered field hair_condition on userpreference
- Altered field hair_length on userpreference
- Altered field hair_texture_detail on userpreference
- Altered field hair_thickness on userpreference
- Altered field hair_type on userpreference
- Altered field lifestyle on userpreference
- Altered field styling_preference on userpreference
- Altered field volume on userpreference

**Migration Status**: Successfully applied ✅

### 5. Validation Alignment

All validation in the following files now references `UserPreference.OCCASION_CHOICES` and other updated choices:
- `serializers.py` - validate_occasions()
- `views.py` - Occasion validation in analyze endpoint
- `views_extended.py` - OccasionsView API endpoint

## Dataset Reference

**File**: `merged_preferences_max_voting.csv`
**Records**: 45,000 rows
**Columns**: 19 columns

### Dataset Unique Values (Key Fields)
- **hair_length**: long, medium, short (3 values)
- **lifestyle**: active, moderate, relaxed (3 values)
- **occasions**: birthday, casual, formal, party, wedding, work (6 values)
- **gender**: female, male (2 values)
- **hair_type**: coily, curly, straight, wavy (4 values)
- **volume**: high, low, medium (3 values)
- **maintenance**: high, low, medium (3 values)
- **styling_preference**: classic, edgy, elegant, natural, trendy (5 values)
- **hair_thickness**: medium, thick, thin (3 values)
- **hair_texture_detail**: fine, normal, thick (3 values)
- **hair_condition**: damaged, dandruff, dry_ends, frizzy, none, oily_scalp, sensitive_scalp, split_ends, thinning (9 values)
- **wants_bangs**: no, yes (2 values)

## Testing Checklist

- [x] Backend model choices updated
- [x] Frontend validation rules updated
- [x] Frontend UI steps updated
- [x] ML feature encoder updated
- [x] Database migration created and applied
- [x] Face shape display removed
- [ ] Test complete user flow: upload → preferences → recommendations
- [ ] Verify all dropdowns show correct options
- [ ] Verify validation accepts only dataset values
- [ ] Verify ML model receives correct feature vector

## Impact

### Critical
- **ML Model Accuracy**: Input features now match training data exactly
- **Validation**: No more invalid values can be submitted
- **User Experience**: Options now match what the model expects

### Breaking Changes
- Existing user preferences with old values (e.g., "pixie", "extra_long", "date", "exercise") will be invalid
- May need data migration for existing UserPreference records

## Next Steps

1. Test the complete flow with real user input
2. Verify recommendations are working correctly
3. Consider data migration for existing user preferences
4. Update any documentation that references old option values
5. Update any test fixtures that use old values

## Files Modified

### Backend
- `backend/hairmixer_app/models.py` - Updated all CHOICES constants
- `backend/hairmixer_app/services/hairstyle_recommender.py` - Updated occasion_types and feature ranges
- `backend/hairmixer_app/migrations/0004_alter_userpreference_gender_and_more.py` - New migration

### Frontend
- `frontend/src/pages/UserPreferences.js` - Updated validationRules and all UI steps

### Documentation
- `docs/DATASET_ALIGNMENT.md` - This file

## Notes

- The occasions API endpoint automatically pulls from `UserPreference.OCCASION_CHOICES`, so no direct API changes were needed
- Serializers and views already referenced model choices, so they automatically validate against updated values
- Face shape detection still runs in the backend but is no longer displayed to users per request
