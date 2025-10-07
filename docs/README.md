# 📚 HairMixer Documentation# HairMixer Project Documentation



This folder contains detailed technical documentation for the HairMixer project.## 📋 Documentation Index



---### Core Documentation

- **[ML_RECOMMENDATIONS.md](ML_RECOMMENDATIONS.md)** - Complete ML hairstyle recommendation system (architecture, features, implementation)

## 📖 Available Documentation- **[API_ML_RECOMMENDATIONS.md](API_ML_RECOMMENDATIONS.md)** - API reference for ML endpoints (requests, responses, examples)

- **[IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)** - Overview of recent implementation changes

### Core Documentation- **[INPUT_VALIDATION.md](INPUT_VALIDATION.md)** - Comprehensive input validation system (frontend & backend)

- **[VALIDATION_COMPLETE.md](VALIDATION_COMPLETE.md)** - Validation implementation summary

| Document | Description | Last Updated |

|----------|-------------|--------------|### Technical Guides

| [API_ML_RECOMMENDATIONS.md](./API_ML_RECOMMENDATIONS.md) | ML recommendation API endpoint details | Oct 2025 |- **Setup Guide** - See below for quick start

| [DATASET_ALIGNMENT.md](./DATASET_ALIGNMENT.md) | Dataset structure and model alignment | Oct 2025 |- **Model Training** - See ML_RECOMMENDATIONS.md

| [INPUT_VALIDATION.md](./INPUT_VALIDATION.md) | Input validation rules and formats | Oct 2025 |- **API Integration** - See API_ML_RECOMMENDATIONS.md

| [ML_RECOMMENDATIONS.md](./ML_RECOMMENDATIONS.md) | ML recommendation system overview | Oct 2025 |- **Input Validation** - See INPUT_VALIDATION.md

- **Migration Troubleshooting** - See MIGRATION_FIX.md

---

---

## 🚀 Quick Links

## 🚀 Quick Start

### For Developers

- **Getting Started:** See [../README.md](../README.md)### Prerequisites

- **API Reference:** [API_ML_RECOMMENDATIONS.md](./API_ML_RECOMMENDATIONS.md)- Python: 3.11 (use `venv311`)

- **Model Details:** [DATASET_ALIGNMENT.md](./DATASET_ALIGNMENT.md)- Frontend: Node 18+ (Create React App)

- Backend: Django 4.2 + DRF

### For ML Engineers

- **Model Training:** `backend/hairmixer_app/ml/models/train_family_model.py`### Setup

- **Model Testing:** `backend/test_hairstyle_recommender.py````powershell

- **Feature Analysis:** [DATASET_ALIGNMENT.md](./DATASET_ALIGNMENT.md)# From repo root

py -3.11 -m venv venv311

### For Frontend Developers.\venv311\Scripts\Activate.ps1

- **UI Implementation:** [../UI_IMPLEMENTATION_COMPLETE.md](../UI_IMPLEMENTATION_COMPLETE.md)python -m pip install --upgrade pip setuptools wheel

- **API Integration:** [API_ML_RECOMMENDATIONS.md](./API_ML_RECOMMENDATIONS.md)python -m pip install -r requirements.txt

- **Input Validation:** [INPUT_VALIDATION.md](./INPUT_VALIDATION.md)```



---### Database Migration (Required for ML Features)

```powershell

## 📊 Model Informationcd backend

python .\manage.py makemigrations

### Random Forest Classifierpython .\manage.py migrate

```

**Model File:** `backend/hairmixer_app/ml/models/hairstyle_family_model.pkl`

### Run Backend

**Specifications:**```powershell

- **Algorithm:** RandomForestClassifiercd backend

- **Classes:** 26 hairstyle familiespython .\manage.py runserver

- **Features:** 15 input features```

- **Training Data:** ~5000 user preferences- API root: http://127.0.0.1:8000/

- **Accuracy:** 95%+ with all features- ML recommendations: http://127.0.0.1:8000/api/recommend/ml/

- Health/upload smoke test: `python ..\run_api_test.py`

**Key Features:**

1. faceshape (auto-detected)### Run Frontend

2. **gender** (critical - +35% accuracy)```powershell

3. hair_typecd frontend

4. hair_lengthnpm install

5. volumenpm start

6. lifestyle```

7. maintenance- Dev server: http://localhost:3000/

8. styling_maintenance- User preferences wizard: http://localhost:3000/preferences

9. occasions

10. hair_color## Gemini Overlay (Optional)

11. hair_texture_detailAdvanced AI overlay via `gemini-webapi`.

12. styling_preference

13. **hair_condition** (important - +12% accuracy)### Install

14. hair_thickness```powershell

15. wants_bangs.\venv311\Scripts\Activate.ps1

python -m pip install gemini-webapi==1.15.2

---```



## 🎯 System Architecture### Configure

Set these env vars before running the backend:

``````powershell

┌─────────────────┐$env:OVERLAY_AI_ENABLED = "true"

│  User Upload    │$env:GEMINI_SECURE_1PSID = "<your cookie>"

│     Image       │$env:GEMINI_SECURE_1PSIDTS = "<your cookie>"

└────────┬────────┘# Optional

         │$env:GEMINI_MODEL = "G_2_5_FLASH"

         ▼$env:GEMINI_TIMEOUT = "120"

┌─────────────────┐```

│   ResNet50 +    │- See `backend/OVERLAY_SETUP.md` for details.

│   MediaPipe     │

│  Face Detection │## Troubleshooting

└────────┬────────┘- Ensure the correct interpreter: VS Code → Python: `venv311`.

         │- If `torch/torchvision` wheels fail, upgrade pip and retry.

         ▼- Media/uploads paths are ignored by Git; add `.gitkeep` to keep folders.

┌─────────────────┐

│   11-Step       │## Development Notes

│   Wizard UI     │- Requirements are consolidated in `requirements.txt` at repo root.

│  (15 features)  │- Django settings read Gemini config via env vars (see `backend/backend/settings.py`).

└────────┬────────┘- Overlay falls back to basic PIL when Gemini is disabled or not configured.

         │
         ▼
┌─────────────────┐
│  Random Forest  │
│     Model       │
│  (26 families)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Database      │
│     Query       │
│ (Top families)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Return Top    │
│      10         │
│ Recommendations │
└─────────────────┘
```

---

## 🔄 Recent Changes (v2.0)

### October 6, 2025

**Model Consolidation:**
- ✅ Removed old MLRecommendationEngine
- ✅ Consolidated to single Random Forest model
- ✅ All endpoints now use hairstyle_family_model.pkl

**UI Updates:**
- ✅ Added Step 0: Gender selection
- ✅ Added Step 10: Hair condition (optional)
- ✅ Updated from 9 to 11 steps
- ✅ Feature collection: 12/15 → 15/15 (100%)

**Accuracy Improvements:**
- ✅ Gender: +35% accuracy improvement
- ✅ Hair condition: +12% accuracy improvement
- ✅ Total: +47% improvement

See [../UI_IMPLEMENTATION_COMPLETE.md](../UI_IMPLEMENTATION_COMPLETE.md) for full details.

---

## 🧪 Testing

### Test Scripts Available

```bash
# Test ML model
python backend/check_model_features.py
python backend/test_hairstyle_recommender.py
python backend/test_random_forest_model.py

# Test face detection
python backend/test_face_detection_direct.py

# Test API endpoints
python backend/test_api_endpoint.py
python backend/run_api_test.py
```

---

## 📝 Documentation Standards

### Document Structure
1. **Title** - Clear, descriptive heading
2. **Overview** - Brief summary of content
3. **Details** - Main content with code examples
4. **Examples** - Practical usage examples
5. **References** - Links to related docs

### Code Examples
- Use markdown code blocks with language tags
- Include comments for clarity
- Show both request and response examples
- Include error handling examples

### Maintenance
- Update "Last Updated" dates
- Keep examples synchronized with code
- Remove outdated information
- Add deprecation notices when needed

---

## 🔗 External Resources

- [Django Documentation](https://docs.djangoproject.com/)
- [React Documentation](https://react.dev/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)

---

## 📧 Contact

For documentation issues or suggestions:
- Open an issue on GitHub
- Contact the development team
- Submit a pull request with improvements

---

**Documentation Version:** 2.0  
**Last Updated:** October 6, 2025  
**Status:** ✅ Current
