# Overlay Feature Setup

> Setup guide for both overlay modes: default local PIL compositing and optional Gemini AI hair editing.

The backend supports two overlay modes:
- basic: Local PIL-based compositing (default)
- advanced: Gemini AI hair editing via gemini-webapi

## 1) Install dependencies

In your virtual environment install project dependencies (run from the `backend/` directory):

```
python -m pip install -r requirements.txt
```

Optional for advanced overlays (if you want Gemini):

```
python -m pip install gemini-webapi==1.15.2
```

## 2) Environment variables

Create a `.env` at the repository root or inside `backend/` based on `.env.example`.

Required for advanced overlay:
- OVERLAY_AI_ENABLED=true
- GEMINI_SECURE_1PSID=...
- GEMINI_SECURE_1PSIDTS=...

Optional:
- GEMINI_MODEL_NAME=gemini-2.0-flash
- GEMINI_TIMEOUT=120

If credentials are missing or AI is disabled, advanced mode will fall back to basic overlay.

## 3) API usage

POST /api/overlay/ (basic or advanced in one call)

Payload:
- image_id: UUID of an `UploadedImage`
- hairstyle_id: UUID of a `Hairstyle`
- overlay_type: "basic" | "advanced"

POST /api/overlay/auto/ (AI overlay, requires auth)

Payload:
- image_id: UUID of an `UploadedImage`
- hairstyle_id: UUID of a `Hairstyle`

The response contains `overlay_url` pointing to the saved PNG under `/media/overlays/...`.

## Notes
- The Gemini flow is asynchronous under the hood and is executed synchronously for the HTTP request. If the server runs with an active event loop (e.g., uvicorn), a nested loop strategy is used.
- If the AI returns multiple images, the first one is saved.
