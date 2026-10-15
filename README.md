# HairMixer - AI-Powered Hairstyle Recommendation System

> Intelligent hairstyle recommendations using Random Forest ML, ResNet50 face shape classification, and Google Gemini AI

![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![React](https://img.shields.io/badge/react-19.1-blue.svg)
![Django](https://img.shields.io/badge/django-4.2-green.svg)
![ML](https://img.shields.io/badge/ML-Random_Forest-orange.svg)
![AI](https://img.shields.io/badge/AI-Gemini-purple.svg)

---

## Table of Contents

1. [What is HairMixer?](#what-is-hairmixer)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [User Flow and Pages](#user-flow-and-pages)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [API Endpoints](#api-endpoints)
8. [ML and AI Model Details](#ml-and-ai-model-details)
9. [Photo Requirements and Preferences](#photo-requirements-and-preferences)
10. [Database Schema](#database-schema)
11. [Security](#security)
12. [Project Structure](#project-structure)
13. [Documentation](#documentation)
14. [Team](#team)

---

## What is HairMixer?

HairMixer is a web-based hairstyle recommendation platform that combines computer vision, machine learning, and generative AI:

- **AI Face Shape Detection** (ResNet50 + MediaPipe, 5 shapes: Oval, Round, Square, Heart, Oblong)
- **ML Recommendations** (Random Forest over 55 hairstyle classes, returns top 10)
- **Generative AI** (Google Gemini for personalized descriptions, product advice, maintenance guides, and photo overlays)
- **Personalization** (15 preference inputs, named preference profiles, saved favorites, history, likes/dislikes)
- **Modern UI** (React 19 + TailwindCSS 3.4, responsive down to 320px width)

Guest users can upload a photo and get session recommendations. Registered users can save preference profiles, bookmark favorites with notes, view history, and use virtual try-on across sessions. See `docs/USER_MANUAL.md` for the full end-user manual.

---

## Key Features

- **Automatic Face Analysis** - Upload a photo (JPEG/PNG/WebP, min 200x200, max 10 MB), get face shape plus confidence score.
- **Smart Recommendations** - ML model suggests top 10 hairstyles with match scores from face shape plus preferences.
- **AI-Powered Insights** - Per-style personalized description, preference-match breakdown, product recommendations, maintenance guide, and styling tips via Gemini.
- **Virtual Try-On (AI Overlay)** - Visualize a hairstyle on your own photo. Two modes: basic local PIL compositing (default) and advanced Gemini WebAPI hair editing (optional, 10-30 seconds per image, downloadable result).
- **Preference Profiles** - Save multiple named presets (for example "Work", "Party") to skip re-entering the form.
- **Discover Page** - Browse the full catalog by category, filter by hair type/length/style, try on or save without running the full flow.
- **Save and Organize** - Save favorites with notes, recommendation snapshot, face shape context, and overlay URL.
- **Community Feedback** - Like/dislike styles, submit 1-5 star feedback with accuracy and satisfaction ratings.
- **History and Reporting** - Registered users can revisit past sessions (date, detected shape, recommended styles) and download try-on images to show a stylist.

---

## System Architecture

```text
User Photo -> MediaPipe face detection -> ResNet50 face shape classification
     +
User Preferences (14 manual inputs + auto face shape)
     |
     v
Random Forest predict_proba -> Top 10 hairstyle families -> Database lookup
     |
     v
Google Gemini AI (optional)
  |- Personalized description, preference match, products, maintenance, tips
  |- Hairstyle overlay image generation (advanced mode)
```

### Tech Stack

**Backend:** Django 4.2.23, Django REST Framework 3.16.1, SimpleJWT 5.5.1, drf-spectacular 0.27.2, django-axes 6.2.0, django-csp 3.7
**ML:** scikit-learn 1.5.2 (runtime; model artifacts trained with 1.7.x), PyTorch 2.2.2+cpu + torchvision 0.17.2+cpu, MediaPipe 0.10.21, Pillow 10.2.0, NumPy 1.26.4, pandas 2.2.3
**AI Services:** google-generativeai 0.8.3 (text, model configurable via `GEMINI_MODEL_NAME`), gemini-webapi (optional image editing via web cookies)
**Frontend:** React 19.1.1, react-router-dom 6.21.1, react-scripts 5.0.1, TailwindCSS 3.4.17, fetch-based API service (no Axios dependency)
**Database:** SQLite (development) / PostgreSQL (production-ready)
**ML Models:** ResNet50 face shape classifier (5 classes, 224x224 ImageNet-normalized input), Random Forest hairstyle recommender (55 classes, ~290 one-hot columns)

Interactive API docs (when backend is running): `GET /api/docs/` (Swagger), `GET /api/redoc/`, `GET /api/schema/`.

---

## User Flow and Pages

1. **Upload Photo** - Camera capture or gallery upload. System validates format/size/dimensions and runs face analysis.
2. **Set Preferences** - Answer the preference form manually or load a saved Preference Profile.
3. **Get Results** - View detected face shape with confidence plus top 10 ranked hairstyle cards (name, preview image, match score, View Details / Try On / Save).
4. **Explore Details** - Open a style for AI-generated description, products, maintenance guide, and styling tips.
5. **Virtual Try-On** - Generate and download an overlay preview.
6. **Save** - Bookmark to Saved Hairstyles with optional notes.

**Pages:** LandingPage, Signup, Login, Dashboard (recent + saved + profiles shortcuts), PhotoUpload, UserPreferences, Results, Discover, UserProfile (account, profiles, saved styles, history).

---

## Quick Start

### Prerequisites

- Python 3.11 (see `.python-version`)
- Node.js 18+ and npm
- Google Gemini API key (optional, required only for AI text/overlay features)

### 1. Clone Repository

```bash
git clone <repository-url>
cd HairMixer
```

### 2. Backend Setup

```powershell
cd backend
python -m venv ..\venv311
..\venv311\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python manage.py migrate
python manage.py createsuperuser

# Seed the catalog (important):
python scripts/repopulate_hairstyles.py
python manage.py populate_discover_hairstyles

# ML model weights (important, not committed to git):
# Download from: https://drive.google.com/drive/folders/1TFjWz9Yq-ZZBFaJNr25ZeJUTaqf1TcX8?usp=drive_link
# Place ResNet weights at: backend/hairmixer_app/ml/models/resnet50_best_model.pth
# Place Random Forest artifacts (*.joblib) at: backend/hairmixer_app/ml/models/hairstyle_model/

python manage.py runserver
```

Backend runs at `http://localhost:8000` (API base `http://localhost:8000/api/`).

Useful backend commands:

```powershell
python manage.py test hairmixer_app.tests.test_smoke -v 2
python manage.py test hairmixer_app -v 2
$env:DJANGO_LOG_LEVEL = 'DEBUG'
python manage.py runserver
```

### 3. Frontend Setup

```bash
cd ../frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`. Set the backend URL in `frontend/.env`:

```env
REACT_APP_API_URL=http://127.0.0.1:8000/api
```

Supported browsers: Chrome (recommended), Firefox, Edge, Safari. Requires JavaScript enabled and a stable internet connection.

---

## Configuration

### Backend (.env)

Copy `.env.example` to `.env` at the repo root or inside `backend/`. Key variables:

```env
DJANGO_SECRET_KEY=your-secret-key-here-change-in-production
DJANGO_DEBUG=true
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOW_ALL=true
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
CSRF_TRUSTED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Gemini AI
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL_NAME=gemini-2.0-flash
OVERLAY_AI_ENABLED=true
GEMINI_TIMEOUT=120
# Optional, only for advanced Gemini WebAPI overlays:
# GEMINI_SECURE_1PSID=your-1psid-cookie
# GEMINI_SECURE_1PSIDTS=your-1psidts-cookie

# Face classifier
FACE_CLASSIFIER_MODEL=resnet50
# FACE_CLASSIFIER_RESNET_PATH=/path/to/resnet_weights.pth

DJANGO_LOG_LEVEL=INFO
RECOMMENDATION_CACHE_TIMEOUT=3600
```

Notes:

- `GEMINI_MODEL_NAME` default in `backend/backend/settings.py` is `gemini-2.5-pro` if unset; the shipped `.env.example` uses `gemini-2.0-flash`.
- Production should set `DJANGO_DEBUG=false`, restrict `DJANGO_ALLOWED_HOSTS` / `CORS_ALLOWED_ORIGINS` / `CSRF_TRUSTED_ORIGINS`, and enable `SECURE_SSL_REDIRECT`, secure cookies, and HSTS (see `.env.example` and `backend/backend/settings.py`).
- JWT lifetimes: access 60 minutes, refresh 1 day with rotation. Brute-force lockout: 5 failures, 1 hour cooloff. Throttles: anon 20/hour, user 200/hour, uploads 10/hour.

### Frontend (.env)

```env
REACT_APP_API_URL=http://127.0.0.1:8000/api
```

---

## API Endpoints

Base URL: `http://<host>:<port>/api/`. Auth uses `Authorization: Bearer <access_token>`.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/auth/signup/` | POST | No | Create account |
| `/api/auth/login/` | POST | No | Obtain access + refresh tokens |
| `/api/auth/logout/` | POST | Yes | Blacklist refresh token |
| `/api/auth/refresh/` | POST | No | Refresh access token |
| `/api/auth/profile/` | GET | Yes | Current user profile |
| `/api/upload/` | POST | No | Upload image (multipart `image`, max 10 MB), detect face shape |
| `/api/preferences/` | POST | Optional | Validate and store preference set |
| `/api/recommend/` | POST | No (throttled) | Full flow: body `{ image_id, preference_id }`, returns top 10 |
| `/api/recommend/ml/` | POST | Varies | ML-only recommendation |
| `/api/hairstyles/` | GET | No | List hairstyles |
| `/api/hairstyles/featured/` | GET | No | Featured list |
| `/api/hairstyles/trending/` | GET | No | Trending list |
| `/api/hairstyles/categories/` | GET | No | List categories |
| `/api/hairstyles/<uuid:style_id>/` | GET | No | Hairstyle detail |
| `/api/hairstyles/<uuid:hairstyle_id>/details/` | GET | No | Detail plus Gemini insights; query `preference_id`, `image_id` |
| `/api/overlay/` | POST | Varies | Basic overlay |
| `/api/overlay/auto/` | POST | Yes | AI overlay; body `{ image_id, hairstyle_id }`, returns `overlay_url` |
| `/api/preference-profiles/` | GET/POST | Yes | List/create profiles |
| `/api/preference-profiles/<id>/` | GET/PUT/DELETE | Yes | Profile detail |
| `/api/preference-profiles/<id>/set-default/` | POST | Yes | Set default profile |
| `/api/saved-hairstyles/` | GET/POST | Yes | List/save favorites |
| `/api/saved-hairstyles/<id>/` | GET/DELETE | Yes | Saved style detail/remove |
| `/api/hairstyle-likes/` | POST | Yes | Like/dislike a style |
| `/api/hairstyle-likes/user/` | GET | Yes | All liked styles |
| `/api/hairstyle-likes/stats/<uuid:hairstyle_id>/` | GET | Varies | Like stats |
| `/api/search/` | GET | No | Search catalog |
| `/api/filter/face-shapes/` | GET | No | Available face shapes |
| `/api/filter/occasions/` | GET | No | Available occasions |
| `/api/feedback/` | POST | Varies | Submit rating/feedback |
| `/api/user/recommendations/` | GET | Yes | User recommendation history |
| `/api/user/favorites/` | GET | Yes | User favorites |
| `/api/user/history/` | GET | Yes | Session history |
| `/api/analytics/event/` | POST | Yes | Track UI event |
| `/api/health/` | GET | No | System status and feature flags |

Full request/response examples: `backend/API_DOCS.md`. Overlay modes: `backend/OVERLAY_SETUP.md`.

---

## ML and AI Model Details

### Face Shape Classification

- **Models:** MediaPipe face detection (primary) plus ResNet50 classifier (default, singleton-loaded once at server start).
- **Classes (5):** Oval, Round, Square, Heart, Oblong.
- **Pipeline:** Detect bounding box, crop with vertical padding, resize to 224x224, ImageNet normalize, ResNet50 inference, softmax, argmax plus confidence.
- **Reported accuracy (model training notes; re-verify after any retraining):** ResNet50 approx 95 percent train / 91 percent test.

### Hairstyle Recommendation

- **Type:** Random Forest Classifier (`hairstyle_model.joblib` plus `hairstyle_family_label_encoder.joblib` plus `model_columns.joblib`).
- **Inputs:** 15 logical fields (face shape auto-detected plus gender, hair type, hair length, hair color, volume, thickness, texture detail, hair condition, lifestyle, maintenance, styling time, style preference, occasions, bangs preference). One-hot encoded to 290 columns at inference.
- **Output:** Probability distribution over 55 hairstyle classes (verified from the shipped label encoder), ranked to top 10 with `predict_proba` plus `argsort`, then mapped to `Hairstyle` database rows.
- **Reported accuracy (model training notes; re-verify after any retraining):** 100 percent train / 80.94 percent test (5-fold cross-validation). Treat as training-time numbers, not a live guarantee.
- **Serving:** Singleton recommender, result caching (`CachedRecommendation`, default TTL 3600 s), throttled recommend endpoint.

### Generative AI (Gemini)

- **Text:** `google-generativeai` with API key. Prompt includes face shape plus confidence, hair attributes, maintenance, lifestyle, and target style. Generates 5 sections: personalized description, preference match, recommended products, maintenance guide, styling tips.
- **Image:** Two overlay paths. Basic: local PIL compositing (always available). Advanced: `gemini-webapi` cookie auth for AI hair editing with automatic blending/lighting handling.
- **Security note:** API key lives only in backend env vars, never exposed to the frontend.

---

## Photo Requirements and Preferences

Derived from `docs/USER_MANUAL.md`. Enforced in serializers: valid image format, max 10 MB, min 200x200 px, sane aspect ratio.

For best face detection:

- Good even lighting without harsh shadows (natural daylight preferred).
- Face clearly visible, looking straight at camera, hair away from forehead.
- Clear, non-blurry image; plain background preferred; remove glasses if glare occurs.
- Supported formats: JPEG, PNG, WebP via camera capture or gallery upload.

Preference form (15 items; face shape is auto-filled):

Gender, Hair Type (Straight/Wavy/Curly/Coily), Hair Length, Hair Color, Volume, Thickness, Texture Detail, Lifestyle, Maintenance Level, Styling Time, Style Preference, Occasions (multi-select: Work/Casual/Formal/Party/Wedding/Birthday), Bangs Preference, Hair Condition. Save answers as a named Preference Profile for reuse.

Confidence guide: 90 percent or higher very confident; 70-89 percent reliable; 50-69 percent moderate (best estimate); below 50 percent low (retake photo).

Contingencies: offline means no analysis/recommendations/try-on; expired sessions require re-login; undetectable faces return an actionable error message.

---

## Database Schema

### Core Catalog and Inputs

- `Hairstyle`: 55 model classes mapped to catalog rows (48 core seeds via `backend/scripts/repopulate_hairstyles.py` plus 24 Discover entries via `populate_discover_hairstyles`; counts overlap). Fields include name, category, suitable gender, face shapes, hair types/lengths, occasions, images, difficulty, trend/popularity scores.
- `HairstyleCategory`: Catalog organization with optional parent category.
- `UploadedImage`: User photo plus file metadata, face detection status/count, processing status.
- `UserPreference`: One recommendation input set including auto face shape plus confidence and all preference fields.

### User Features

- `PreferenceProfile`: Named reusable preference sets with `is_default` and `last_used_at`.
- `SavedHairstyle`: Favorite with notes, recommendation snapshot, face shape context, overlay URL, personalized description.
- `HairstyleLike`: One reaction per user per style (like/dislike), feedback tracking only, does not reorder ML output.
- `Feedback`: Linked to a `RecommendationLog`, includes liked flag, 1-5 rating, accuracy/style satisfaction, free-text note, would-recommend flag.
- `RecommendationLog`: Per-run record (user, image, preference, face shape/confidence, candidates/scores, processing time, model version, view/share counts).
- `CustomUser` / `UserProfile`: Email-login user plus extended profile.

### System

- `AnalyticsEvent`: page view, upload, recommendation, style click, overlay, feedback, share events with session/IP/user-agent.
- `CachedRecommendation`: cache key, face shape, preference hash, recommendations JSON, hit count, expiry.

---

## Security

- **JWT (SimpleJWT):** Signed access (60 min) plus rotating refresh (1 day) tokens; stateless verification, no server session lookup.
- **Brute-force protection (django-axes):** Block after 5 failed logins, 1 hour cooloff, counter resets on success.
- **Rate limiting (DRF throttles):** anon 20/hour, authenticated user 200/hour, uploads 10/hour.
- **Input validation (serializers + Pillow):** File size, image validity (`verify()`), minimum dimensions, aspect ratio checks before ML inference.
- **HTTP hardening:** `SECURE_SSL_REDIRECT`, HSTS (31536000 s), `X_FRAME_OPTIONS: DENY`, Content Security Policy via django-csp, CORS/CSRF allowlists for production.
- See `backend/backend/settings.py` for exact security values.

---

## Project Structure

```text
HairMixer/
├── backend/
│   ├── backend/                  # Django settings, URLs, WSGI
│   ├── hairmixer_app/
│   │   ├── models.py             # All models (user, catalog, logs, likes, cache)
│   │   ├── serializers.py
│   │   ├── urls.py               # Full API routes under /api/
│   │   ├── views/                # auth, recommendation, catalog, overlay, user, system
│   │   ├── services/             # gemini_service, overlay_service, recommendation_service, pipeline, cache, analytics
│   │   ├── logic/                # Domain logic
│   │   ├── ml/                   # face_analyzer, resnet_classifier, recommenders, models/ weights + joblibs
│   │   ├── management/commands/  # populate_discover_hairstyles, sample data, smoke tests
│   │   ├── middleware.py / exceptions.py / admin.py
│   ├── scripts/                  # repopulate_hairstyles.py, list_model_hairstyles.py
│   ├── API_DOCS.md
│   ├── OVERLAY_SETUP.md
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/                # LandingPage, Login, Signup, Dashboard, PhotoUpload, UserPreferences, Results, Discover, UserProfile
│   │   ├── components/           # Navbar, HairstyleCard, DiscoverSection, ui/*
│   │   ├── services/             # api.js, AuthService.js, ProtectedRoutes
│   │   └── App.js / index.js
│   ├── package.json
│   └── .env.example
├── docs/
│   └── USER_MANUAL.md            # End-user manual (source for Section 9 above)
├── SYSTEM_DOCUMENTATION.md       # Architecture and API summary
├── thesis_preparation.md         # Defense reference (models, security, schema, Q&A)
├── .env.example
└── requirements.txt
```

---

## Documentation

- `docs/USER_MANUAL.md` - Installation-free user guide: account creation, menu tour, photo upload, 15-question form, results, details, try-on, saving, troubleshooting, reporting. Version 2.1.0, updated January 2026.
- `SYSTEM_DOCUMENTATION.md` - Architecture diagram, core features, database and API summary. Version 2.1.0.
- `backend/API_DOCS.md` - Authoritative endpoint reference with payloads plus local Swagger/ReDoc URLs.
- `backend/OVERLAY_SETUP.md` - Basic vs advanced overlay setup, including `gemini-webapi` install and cookie env vars.
- `backend/README.md` - Backend-only quick start, tests, and logging notes.

---

## Team

- John Mathew Mauricio
- Theris Eldrene Carroz
- Jazzer Ong

---

<div align="center">

**Built for better hairstyle decisions**

Version 2.1.0 | Last updated January 2026

</div>
