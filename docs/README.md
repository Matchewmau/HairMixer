# HairMixer Project Docs

## Quick Start
- Python: 3.11 (use `venv311`)
- Frontend: Node 18+ (Create React App)
- Backend: Django 4.2 + DRF

### Setup
```powershell
# From repo root
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Run Backend
```powershell
cd backend
python .\manage.py migrate
python .\manage.py runserver
```
- API root: http://127.0.0.1:8000/
- Health/upload smoke test: `python ..\run_api_test.py`

### Run Frontend
```powershell
cd frontend
npm install
npm start
```
- Dev server: http://localhost:3000/

## Gemini Overlay (Optional)
Advanced AI overlay via `gemini-webapi`.

### Install
```powershell
.\venv311\Scripts\Activate.ps1
python -m pip install gemini-webapi==1.15.2
```

### Configure
Set these env vars before running the backend:
```powershell
$env:OVERLAY_AI_ENABLED = "true"
$env:GEMINI_SECURE_1PSID = "<your cookie>"
$env:GEMINI_SECURE_1PSIDTS = "<your cookie>"
# Optional
$env:GEMINI_MODEL = "G_2_5_FLASH"
$env:GEMINI_TIMEOUT = "120"
```
- See `backend/OVERLAY_SETUP.md` for details.

## Troubleshooting
- Ensure the correct interpreter: VS Code → Python: `venv311`.
- If `torch/torchvision` wheels fail, upgrade pip and retry.
- Media/uploads paths are ignored by Git; add `.gitkeep` to keep folders.

## Development Notes
- Requirements are consolidated in `requirements.txt` at repo root.
- Django settings read Gemini config via env vars (see `backend/backend/settings.py`).
- Overlay falls back to basic PIL when Gemini is disabled or not configured.
