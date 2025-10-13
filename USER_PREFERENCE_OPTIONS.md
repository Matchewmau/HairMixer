# HairMixer - User Preference Options for Hairstyle Recommendations

## 📋 Complete List of User Preference Options

This document lists all the options users can select when getting hairstyle recommendations from the HairMixer system.

---

## 1️⃣ **Face Shape** (faceshape)
**Required:** Yes (Auto-detected via AI, can be overridden)

- `oval` - Oval
- `round` - Round
- `square` - Square
- `heart` - Heart
- `oblong` - Oblong
- `diamond` - Diamond
- `triangle` - Triangle

---

## 2️⃣ **Gender** (gender)
**Required:** Yes

- `male` - Male
- `female` - Female
- `nb` - Non-Binary
- `other` - Other

**Note:** 
- Male users receive male/unisex hairstyles
- Female users receive female/unisex hairstyles
- Non-binary/Other users receive unisex hairstyles only

---

## 3️⃣ **Hair Type** (hair_type)
**Required:** Yes

- `straight` - Straight
- `wavy` - Wavy
- `curly` - Curly
- `coily` - Coily

---

## 4️⃣ **Hair Length** (hair_length)
**Required:** Yes

- `pixie` - Pixie (Very Short)
- `short` - Short
- `medium` - Medium
- `long` - Long
- `extra_long` - Extra Long

**Note:** Database uses `short`, `medium`, `long` primarily

---

## 5️⃣ **Hair Color** (hair_color)
**Required:** Yes

- `black` - Black ⬛
- `brown` - Brown 🟤
- `blonde` - Blonde 🟡
- `red` - Red 🔴
- `gray` - Gray ⚪
- `white` - White ⚪
- `other` - Other (dyed/unusual colors) 🎨

---

## 6️⃣ **Hair Volume** (volume)
**Required:** Yes

- `low` - Low (Flat)
- `light` - Light
- `medium` - Medium
- `high` - High (Full)

**Note:** `low` maps to `flat` in the model

---

## 7️⃣ **Hair Thickness** (hair_thickness)
**Required:** Yes

- `thin` - Thin/Fine
- `medium` - Medium
- `thick` - Thick
- `very_thick` - Very Thick

---

## 8️⃣ **Hair Texture Detail** (hair_texture_detail)
**Required:** Yes

- `fine` - Fine
- `normal` - Normal
- `thick` - Thick
- `smooth` - Smooth
- `coarse` - Coarse
- `silky` - Silky
- `frizzy` - Frizzy

---

## 9️⃣ **Hair Condition** (hair_condition)
**Required:** Optional

- `none` - None/Healthy
- `excellent` - Excellent
- `good` - Good
- `fair` - Fair
- `damaged` - Damaged
- `dry_ends` - Dry Ends
- `oily_scalp` - Oily Scalp
- `dandruff` - Dandruff
- `frizzy` - Frizzy
- `split_ends` - Split Ends
- `thinning` - Thinning
- `sensitive_scalp` - Sensitive Scalp

---

## 🔟 **Lifestyle** (lifestyle)
**Required:** Yes

- `active` - Active (Sports, Gym)
- `professional` - Professional (Office, Business)
- `creative` - Creative (Artist, Designer)
- `casual` - Casual
- `moderate` - Moderate (balanced lifestyle)
- `relaxed` - Relaxed (low-key)

**Note:** `moderate`, `relaxed`, and `casual` map to the same category

---

## 1️⃣1️⃣ **Maintenance Level** (maintenance)
**Required:** Yes

- `low` - Low (Wash & Go)
- `medium` - Medium (Some Styling)
- `high` - High (Daily Styling)

---

## 1️⃣2️⃣ **Styling Maintenance** (styling_maintenance)
**Required:** Yes

- `low` - Low (Minimal daily effort)
- `medium` - Medium (Moderate daily effort)
- `high` - High (Significant daily styling)

---

## 1️⃣3️⃣ **Styling Preference** (styling_preference)
**Required:** Yes

- `natural` - Natural (Low-maintenance, effortless)
- `casual` - Casual
- `classic` - Classic (Timeless)
- `polished` - Polished (Professional, refined)
- `elegant` - Elegant (Sophisticated)
- `glamorous` - Glamorous (High-fashion)
- `edgy` - Edgy (Bold, modern)
- `trendy` - Trendy (Current fashion)

**Note:** Similar values may map to the same category in the model

---

## 1️⃣4️⃣ **Occasions** (occasions)
**Required:** Yes (Multi-select, at least 1)

- `work` - Work/Office
- `casual` - Casual/Everyday
- `formal` - Formal Events
- `party` - Parties
- `wedding` - Weddings
- `birthday` - Birthdays
- `date` - Dates (model may use)
- `exercise` - Exercise/Gym (model may use)
- `travel` - Travel (model may use)

**Note:** User can select multiple occasions. Model encodes this as a count.

---

## 1️⃣5️⃣ **Wants Bangs/Fringe** (wants_bangs)
**Required:** Yes

- `true` - Yes, I want bangs
- `false` - No, I don't want bangs

---

## 📊 Summary Statistics

- **Total Fields:** 15
- **Required Fields:** 14 (hair_condition is optional)
- **Multi-select Fields:** 1 (occasions)
- **Boolean Fields:** 1 (wants_bangs)
- **Categorical Fields:** 13

---

## 🎯 Feature Order in ML Model

The model expects features in this **exact order**:

1. faceshape
2. gender
3. hair_type
4. hair_length
5. volume
6. lifestyle
7. maintenance
8. styling_maintenance
9. occasions (count)
10. hair_color
11. hair_texture_detail
12. styling_preference
13. hair_condition
14. hair_thickness
15. wants_bangs

---

## 💡 Example User Preference JSON

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
  "hair_condition": "good",
  "lifestyle": "professional",
  "maintenance": "medium",
  "styling_maintenance": "medium",
  "styling_preference": "polished",
  "occasions": ["work", "casual", "formal"],
  "wants_bangs": true
}
```

---

## 🔄 Model Processing

1. User submits preferences through UI
2. Backend encodes categorical values to integers
3. ML model predicts top hairstyle classes (144 options)
4. System queries database for matching hairstyles
5. Gender filter ensures appropriate recommendations
6. Returns top 10 recommendations with confidence scores

---

## 📝 Notes for Developers

- **Gender Filtering:** Always enforced, never relaxed
- **Default Values:** Empty strings default to middle values (e.g., medium)
- **Occasions:** Encoded as count (number of selected occasions)
- **Face Shape:** Auto-detected but can be manually overridden
- **Hair Color:** New field added in UI (step 10 of wizard)

---

**Last Updated:** October 9, 2025  
**Model Version:** hairmixer_model.pkl (144 classes, 15 features)
