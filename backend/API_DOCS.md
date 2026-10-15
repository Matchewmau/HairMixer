# HairMixer API Documentation

> Practical backend API reference with request/response payloads; interactive docs at `/api/docs/` (Swagger) and `/api/redoc/`.

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
- `image`: binary file (JPEG/PNG/WebP), max 10MB, min 200x200 px

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
  "gender": "female|male",
  "hair_type": "straight|wavy|curly|coily",
  "hair_length": "short|medium|long",
  "hair_color": "black|brown|blonde|red|gray|...",
  "maintenance": "low|medium|high",
  "lifestyle": "active|professional|creative|casual|moderate|relaxed",
  "occasions": ["work", "casual", "formal", "party", "wedding", "birthday"],
  "wants_bangs": false
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
- Anonymous: 20 per hour per user/IP
- Authenticated users: 200 per hour

## Running Locally

```powershell
cd backend
..\venv311\Scripts\Activate.ps1
python manage.py runserver

# open docs
# Swagger: http://localhost:8000/api/docs/
# ReDoc:   http://localhost:8000/api/redoc/
```

---
*HairMixer API docs v2.1.0. See the root `README.md` for setup and `docs/USER_MANUAL.md` for the end-user guide.*

