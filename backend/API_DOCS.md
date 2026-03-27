# HairMixer API Documentation

This document provides a practical reference for integrating with the HairMixer backend API. It complements the interactive docs powered by drf-spectacular.

- Base URL: `http://<host>:<port>/api/`
- Interactive Docs: `GET /api/docs/` (Swagger UI)
- ReDoc: `GET /api/redoc/`
- OpenAPI Schema (JSON): `GET /api/schema/`

## Authentication

JWT-based authentication using SimpleJWT. Most core endpoints are open, but some require auth.

- `POST /auth/signup/` — Create a new account
- `POST /auth/login/` — Obtain access + refresh tokens
- `POST /auth/logout/` — Logout and blacklist refresh token (optional)
- `POST /auth/refresh/` — Refresh access token
- `GET  /auth/profile/` — Get current user profile (requires auth)

Authorization header for protected endpoints:

```
Authorization: Bearer <access_token>
```

## Core Endpoints

### Upload Image
- `POST /upload/`
- Open to all (no auth). Multipart request.

Request (multipart/form-data):
- `image`: binary file (jpg/png), max 10MB

Success response (abbreviated):

```
200 OK
{
  "success": true,
  "image_id": "<uuid>",
  "face_detected": true,
  "face_shape": { "shape": "oval", "confidence": 0.93 },
  "message": "Image uploaded and analyzed successfully"
}
```

### Set Preferences
- `POST /preferences/`
- Open to all (auth optional). Validates and stores preferences.

Body (JSON):

```
{
  "hair_type": "wavy|straight|curly|coily",
  "hair_length": "pixie|short|medium|long|extra_long",
  "maintenance": "low|medium|high",
  "lifestyle": "casual|active|professional|glam|...",
  "occasions": ["casual", "party"],
  "gender": "male|female|nb|other",
  "hair_color": "brown",
  "color_preference": "warm",
  "budget_range": "$$"
}
```

### Recommend
- `POST /recommend/`
- Open to all (throttled). Generates hairstyle recommendations given an uploaded image and preferences.

Body (JSON):

```
{
  "image_id": "<uuid>",
  "preference_id": "<uuid>"
}
```

### Hairstyle Details with AI
- `GET /hairstyles/<uuid:hairstyle_id>/details/` (open)
- Get detailed hairstyle information with AI-generated personalized content
- Query Parameters:
  - `preference_id`: UUID of user preferences (optional)
  - `image_id`: UUID of uploaded image (optional)

Response:

```
200 OK
{
  "hairstyle": { ... },
  "ai_generated": true,
  "personalized_description": "This layered bob will beautifully frame your oval face...",
  "recommended_products": [ ... ],
  "maintenance_guide": [ ... ],
  "styling_tips": [ ... ]
}
```

### Auto Overlay (AI)
- `POST /overlay/auto/` (requires auth)
- Generate an AI-powered overlay for a selected hairstyle on the uploaded image.

Body (JSON):

```
{
  "image_id": "<uuid>",
  "hairstyle_id": "<uuid>"
}
```

Response:

```
200 OK
{
  "overlay_url": "/media/overlays/ai_generated_....png",
  "status": "success"
}
```

## User Management Endpoints

### Preference Profiles
- `GET /preference-profiles/` - List all profiles
- `POST /preference-profiles/` - Create new profile
- `GET /preference-profiles/<id>/` - Get profile details
- `POST /preference-profiles/<id>/set-default/` - Set as default profile

### Saved Hairstyles
- `GET /saved-hairstyles/` - List saved favorites
- `POST /saved-hairstyles/` - Save a hairstyle
- `GET /saved-hairstyles/<id>/` - Get details of saved style
- `DELETE /saved-hairstyles/<id>/` - Remove from favorites

### Hairstyle Likes
- `POST /hairstyle-likes/` - Like or Dislike a hairstyle
- `GET /hairstyle-likes/user/` - Get all liked hairstyles

## Search & Filters

- `GET /search/` (open)
- `GET /hairstyles/` — list (open)
- `GET /hairstyles/featured/` — featured list (open)
- `GET /hairstyles/trending/` — trending list (open)
- `GET /hairstyles/categories/` — list categories (open)
- `GET /filter/face-shapes/` — available face shapes (open)
- `GET /filter/occasions/` — available occasions (open)

## Analytics & Admin

- `POST /analytics/event/` (auth)
- `GET /admin/cache/stats/` (admin)
- `POST /admin/cache/cleanup/` (admin)
- `GET /admin/analytics/` (admin)

## Health Check

- `GET /health/` — returns system status and feature flags.

## Rate Limits (Throttling)

- Uploads: 10 per hour per user/IP
- Recommendations: 20 per hour per user/IP

## Running Locally

```
# activate env (PowerShell)
D:/CODING/Python/HairMixer/venv/Scripts/Activate.ps1

# run server (quiet logs by default)
D:/CODING/Python/HairMixer/venv/Scripts/python.exe backend/manage.py runserver

# open docs
# Swagger: http://localhost:8000/api/docs/
# ReDoc:   http://localhost:8000/api/redoc/
```

