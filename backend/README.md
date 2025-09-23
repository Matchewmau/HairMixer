# HairMixer Backend

This backend powers the HairMixer API (Django + DRF + drf-spectacular).

- Base URL: `http://localhost:8000/api/`
- Docs: `/api/docs/` (Swagger), `/api/redoc/`
- Schema: `/api/schema/`

## Quick Start (Windows PowerShell)

```powershell
# Activate venv
D:/CODING/Python/HairMixer/venv/Scripts/Activate.ps1

# Install deps (use consolidated root file)
D:/CODING/Python/HairMixer/venv/Scripts/python.exe -m pip install -r requirements.txt

# Migrate DB
D:/CODING/Python/HairMixer/venv/Scripts/python.exe backend/manage.py migrate --noinput

# Run server
D:/CODING/Python/HairMixer/venv/Scripts/python.exe backend/manage.py runserver
```

Open Swagger UI: http://localhost:8000/api/docs/

## Tests

```powershell
# Run smoke tests (recommended quick check)
D:/CODING/Python/HairMixer/venv/Scripts/python.exe backend/manage.py test hairmixer_app.tests.test_smoke -v 2 --noinput

# Run all app tests
D:/CODING/Python/HairMixer/venv/Scripts/python.exe backend/manage.py test hairmixer_app -v 2 --noinput
```

Note: Using PowerShell pipelines with `Tee-Object` may cause a non-zero exit code even when tests pass. Prefer running directly to rely on Django’s exit code.

## Logging

The backend now minimizes noisy logs:
- TensorFlow/absl/mediapipe verbosity suppressed where applicable
- Internal debug logs use the `logging` module (no prints)
- Default console output is concise

Enable debug logs while developing:

```powershell
$env:DJANGO_LOG_LEVEL = 'DEBUG'
D:/CODING/Python/HairMixer/venv/Scripts/python.exe backend/manage.py runserver

Stop debug logs (revert to default)
Remove-Item Env:DJANGO_LOG_LEVEL

Or explicitly set a quieter level:
$env:DJANGO_LOG_LEVEL = 'INFO'

Set to show only warnings and errors
$env:DJANGO_LOG_LEVEL = 'WARNING'

```

## Face Analysis Pipeline

- Detection: MediaPipe (primary). FaceNet (MTCNN) is an optional fallback if installed.
- Face Shape: MobileNetV3 or ResNet50 classifier.
- Image Processing: PIL + NumPy, OpenCV not required.

## File Paths

- Analyzer: `backend/hairmixer_app/ml/face_analyzer.py`
- MobileNet Loader: `backend/hairmixer_app/ml/mobilenet_classifier.py`
- ResNet Loader: `backend/hairmixer_app/ml/resnet_classifier.py`
- Face Shapes: `backend/hairmixer_app/ml/model.py`
- Tests: `backend/hairmixer_app/tests/`

## Overlay Feature

See `backend/OVERLAY_SETUP.md` for advanced overlay (Gemini) configuration. Falls back to basic PIL overlay when AI is disabled.

## Switching Classifier Models

You can now choose which face shape classifier to use:

- `FACE_CLASSIFIER_MODEL`: `mobilenet_v3` (default) or `resnet50`
- `FACE_CLASSIFIER_MOBILENET_PATH`: optional path to MobileNet weights
- `FACE_CLASSIFIER_RESNET_PATH`: optional path to ResNet weights
- `FACE_CLASSIFIER_WEIGHTS`: optional generic path used by either model if specific one not set

Example (PowerShell):

```powershell
$env:FACE_CLASSIFIER_MODEL = 'resnet50'
$env:FACE_CLASSIFIER_RESNET_PATH = 'backend/hairmixer_app/ml/models/resnet50_80epoch.pth'
D:/CODING/Python/HairMixer/venv/Scripts/python.exe backend/manage.py runserver
```

To switch back to MobileNetV3 (default path `backend/hairmixer_app/ml/models/mobilenetv3_small.pth`):

```powershell
$env:FACE_CLASSIFIER_MODEL = 'mobilenet_v3'
# optionally override path
# $env:FACE_CLASSIFIER_MOBILENET_PATH = 'backend/hairmixer_app/ml/models/mobilenetv3_small.pth'
D:/CODING/Python/HairMixer/venv/Scripts/python.exe backend/manage.py runserver
```
