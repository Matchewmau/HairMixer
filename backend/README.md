# HairMixer Backend

This backend powers the HairMixer API (Django + DRF + drf-spectacular).

- Base URL: `http://localhost:8000/api/`
- Docs: `/api/docs/` (Swagger), `/api/redoc/`
- Schema: `/api/schema/`

## Quick Start (Windows PowerShell)

```powershell
# Activate venv
D:/CODING/Python/HairMixer/venv311/Scripts/Activate.ps1

# Migrate DB
python manage.py migrate

# Create Superuser
python manage.py createsuperuser

# Populate Database (Important)
D:\CODING\Python\HairMixer\backend\scripts\repopulate_hairstyles.py
D:\CODING\Python\HairMixer\backend\hairmixer_app\management\commands\populate_discover_hairstyles.py

#Models upload (Important)
https://drive.google.com/drive/folders/1TFjWz9Yq-ZZBFaJNr25ZeJUTaqf1TcX8?usp=drive_link
#add the models here: 
D:\CODING\Python\HairMixer\backend\hairmixer_app\ml\models\resnet50_best_model.pth
D:\CODING\Python\HairMixer\backend\hairmixer_app\ml\models\hairstyle_model -> all hairstyle joblib files

# Run server
python manage.py runserver
```

Open Swagger UI: http://localhost:8000/api/docs/

## AI Configuration (Gemini)

To enable AI features (styling tips, overlays), configure your `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
OVERLAY_AI_ENABLED=true
# Optional: Gemini WebAPI cookies for image generation
GEMINI_SECURE_1PSID=...
GEMINI_SECURE_1PSIDTS=...
```

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
