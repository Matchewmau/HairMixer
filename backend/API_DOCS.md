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

Example: Login

```
POST /api/auth/login/
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "YourPassword123"
}
```

Response:

```
200 OK
{
  "message": "Login successful",
  "user": { "id": "...", "email": "user@example.com", ... },
  "access_token": "<jwt>",
  "refresh_token": "<jwt>"
}
```

## Core Endpoints

### Upload Image
- `POST /upload/`
- Open to all (no auth). Multipart request.

Request (multipart/form-data):
- `image`: binary file (jpg/png), max 10MB

cURL example:

```
curl -X POST http://localhost:8000/api/upload/ \
  -F "image=@/path/to/photo.jpg"
```

Success response (abbreviated):

```
200 OK
{
  "success": true,
  "image_id": "<uuid>",
  "face_detected": true,
  "face_shape": { "shape": "oval", "confidence": 0.93 },
  "message": "Image uploaded and analyzed successfully",
  "face_shape_description": "Balanced proportions - most hairstyles suit you!"
}
```

Error response:

```
400 Bad Request
{ "error": "No image provided" }
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

Response:

```
200 OK
{
  "success": true,
  "preference_id": "<uuid>",
  "message": "Preferences saved successfully",
  "preferences": { ...validated fields... }
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

Example response (shape varies by engine):

```
200 OK
{
  "recommendations": [
    { "id": "<uuid>", "name": "Layered Bob", "trend_score": 0.82, ... },
    { "id": "<uuid>", "name": "Textured Pixie", "trend_score": 0.79, ... }
  ],
  "metadata": { "generated_at": "2025-09-21T22:11:00Z" }
}
```

Errors:
- 404 if `image_id` or `preference_id` not found
- 500 if generation fails

### Overlay
- `POST /overlay/` (requires auth)
- Generate an overlay for a selected hairstyle on the uploaded image.

Body (JSON):

```
{
  "image_id": "<uuid>",
  "hairstyle_id": "<uuid>",
  "overlay_type": "basic|advanced"
}
```

Response:

```
200 OK
{
  "overlay_url": "/media/overlays/<image>_<style>_basic.png",
  "overlay_type": "basic"
}
```

### Search
- `GET /search/` (open)
- Parameters:
  - `q`: string (text search)
  - `face_shape`: one of `oval, round, square, heart, diamond, oblong`
  - `occasion`: one of `casual, formal, party, business, wedding, date, work`
  - `hair_type`: one of `straight, wavy, curly, coily`
  - `maintenance`: one of `low, medium, high`
  - `page`: integer, default 1
  - `per_page`: integer, default 20 (max 50)

Example:

```
GET /api/search/?q=layered&face_shape=oval&per_page=12
```

Response (abbreviated):

```
200 OK
{
  "results": [ { "id": "<uuid>", "name": "Layered Bob", ... } ],
  "search_query": "layered",
  "filters_applied": { ... },
  "pagination": {
    "page": 1,
    "per_page": 12,
    "total_pages": 3,
    "total_count": 36,
    "has_next": true,
    "has_previous": false
  }
}
```

## Hairstyle Endpoints

- `GET /hairstyles/` — list (open; supports filters via query params similar to Search)
- `GET /hairstyles/featured/` — featured list (open)
- `GET /hairstyles/trending/` — trending list (open)
- `GET /hairstyles/<style_id>/` — detail (open)
- `GET /hairstyles/categories/` — list categories (open)

## User Endpoints

- `GET /user/recommendations/` — user recommendation history (auth)
- `GET /user/favorites/` — favorites (auth; placeholder)
- `GET /user/history/` — activity history (auth; placeholder)

## Filters

- `GET /filter/face-shapes/` — available face shapes + guidance (open)
- `GET /filter/occasions/` — available occasions (open)

## Analytics & Admin

- `POST /analytics/event/` (auth):

```
{
  "event_type": "overlay_generated",
  "event_data": { "style_id": "<uuid>" },
  "session_id": "abc-123"
}
```

- `GET /admin/cache/stats/` (admin)
- `POST /admin/cache/cleanup/` (admin)
- `GET /admin/analytics/` (admin)

Note: Admin-permission enforcement can be environment-gated; by default it requires authentication and may be tightened for production.

## Health Check

- `GET /health/` — returns system status and feature flags.

```
200 OK
{
  "status": "ok",
  "ml_available": true,
  "preprocess_available": true,
  ...
}
```

## Errors & Status Codes

- `400 Bad Request` — validation or malformed input
- `401 Unauthorized` — missing/invalid token for protected routes
- `404 Not Found` — resource not found
- `429 Too Many Requests` — throttling (see rate limits)
- `500 Internal Server Error` — server-side error

Error format typically:

```
{ "error": "message", "details": "optional context" }
```

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

## Notes

- For better request/response details, use the interactive docs. Some views use explicit schema annotations to ensure accurate examples and parameter enums.
- Overlay requires a valid authenticated user. Obtain a JWT via login and set the `Authorization` header.
- Image analysis happens at upload time; the upload response includes analysis metadata used by recommendations.
- Logging: internal debug is suppressed by default; set `DJANGO_LOG_LEVEL=DEBUG` to see detailed analyzer output.
