# HairMixer Backend

> Django + DRF API serving face analysis, ML recommendations, Gemini AI insights, overlays, and user data.

This backend powers the HairMixer API (Django + DRF + drf-spectacular).

- Base URL: `http://localhost:8000/api/`
- Docs: `/api/docs/` (Swagger), `/api/redoc/`
- Schema: `/api/schema/`

## Quick Start (Windows PowerShell)

```powershell
cd backend
python -m venv ..\venv311
..\venv311\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Migrate DB
python manage.py migrate

# Create Superuser
python manage.py createsuperuser

# Populate Database (Important)
python scripts/repopulate_hairstyles.py
python manage.py populate_discover_hairstyles

# ML model weights (Important, not committed to git)
# Download from: https://drive.google.com/drive/folders/1TFjWz9Yq-ZZBFaJNr25ZeJUTaqf1TcX8?usp=drive_link
# Place ResNet weights at: backend/hairmixer_app/ml/models/resnet50_best_model.pth
# Place Random Forest artifacts (*.joblib) at: backend/hairmixer_app/ml/models/hairstyle_model/

# Run server
python manage.py runserver
```

Open Swagger UI: http://localhost:8000/api/docs/

See the root `README.md` for full-stack setup (frontend, env vars, versions) and `docs/USER_MANUAL.md` for the end-user guide.

## AI Configuration (Gemini)

To enable AI features (styling tips, overlays), configure your `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-2.0-flash
OVERLAY_AI_ENABLED=true
# Optional: Gemini WebAPI cookies for image generation
GEMINI_SECURE_1PSID=...
GEMINI_SECURE_1PSIDTS=...
```

See `OVERLAY_SETUP.md` for basic vs advanced overlay setup, and `API_DOCS.md` for endpoint payloads.

## Tests

```powershell
# Run smoke tests
python manage.py test hairmixer_app.tests.test_smoke -v 2

# Run all app tests
python manage.py test hairmixer_app -v 2
```

## Logging

The backend minimizes noisy logs. Enable debug logs via env var:

```powershell
$env:DJANGO_LOG_LEVEL = 'DEBUG'
python manage.py runserver
```

## Face Analysis Pipeline

- **Detection**: MediaPipe (primary).
- **Face Shape**: ResNet50 (default)
- **Image Processing**: PIL + NumPy.
