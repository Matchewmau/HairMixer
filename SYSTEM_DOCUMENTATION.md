# HairMixer System Documentation

> Technical system reference: architecture, features, ML/AI integration, API map, database schema, and deployment.

**Version:** 2.1.0
**Last Updated:** January 2026
**Status:** Production Ready

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Features](#core-features)
4. [AI and ML Integration](#ai-and-ml-integration)
5. [API Reference](#api-reference)
6. [Database Schema](#database-schema)
7. [Setup and Deployment](#setup-and-deployment)

---

## System Overview

HairMixer is an intelligent hairstyle recommendation platform that uses machine learning, computer vision, and generative AI to provide personalized hairstyle suggestions and visualizations.

### Key Capabilities

1. **Face Shape Detection**: MediaPipe detection plus ResNet50 classification of user photos into 5 shapes.
2. **ML Recommendations**: Random Forest model matching 15 preference inputs (14 manual plus auto-detected face shape) to 55 hairstyle classes, returning the top 10.
3. **Generative AI Insights**: Google Gemini-powered styling tips, maintenance guides, and product recommendations.
4. **AI Overlays**: Basic local compositing plus optional Gemini WebAPI visualization of hairstyles on user photos.
5. **User Personalization**: Preference profiles, saved favorites with notes, history tracking, and like/dislike feedback.

Guest users get session recommendations; registered users persist profiles, favorites, and history. End-user instructions live in `docs/USER_MANUAL.md`.

---

## Architecture

### Technology Stack

#### Backend
- **Framework:** Django 4.2.23 + Django REST Framework 3.16.1 + SimpleJWT 5.5.1
- **ML:** scikit-learn 1.5.2 (runtime), PyTorch 2.2.2+cpu, MediaPipe 0.10.21
- **AI:** Google Gemini API (text, `google-generativeai` 0.8.3), Gemini WebAPI (optional image editing)
- **Database:** SQLite (dev) / PostgreSQL (prod)
- **Docs/Security:** drf-spectacular 0.27.2, django-axes 6.2.0, django-csp 3.7

#### Frontend
- **Framework:** React 19.1.1 with react-router-dom 6.21.1
- **Styling:** TailwindCSS 3.4.17
- **API layer:** fetch-based service using `REACT_APP_API_URL`

### System Architecture Diagram

```
+-----------------+
|     Client      | (React Frontend)
+--------+--------+
         |
+--------v--------------------------------------------------+
|                Django Backend (API)                        |
|                                                            |
|  +----------------+  +----------------+  +--------------+  |
|  |  ML Services   |  |  AI Services   |  |     DB       |  |
|  | ---------------- |  | ---------------- |  | -------------- |  |
|  | - MediaPipe    |  | - Gemini API   |  | - Users      |  |
|  | - ResNet50     |  |   (Text)       |  | - Styles     |  |
|  | - Random       |  | - Gemini Web   |  | - Profiles   |  |
|  |   Forest       |  |   (Image)      |  | - Saved/Likes|  |
|  +----------------+  +----------------+  +--------------+  |
+------------------------------------------------------------+
```

---

## Core Features

### 1. Face Shape Detection
**Model:** MediaPipe + ResNet50
**Shapes:** Oval, Round, Square, Heart, Oblong
**Process:** Validates the upload (JPEG/PNG/WebP, max 10 MB, min 200x200 px), detects face landmarks, crops with padding, resizes to 224x224, and classifies shape with a confidence score.

### 2. Preference Profiles
Users can create multiple named profiles (for example "Professional", "Casual") to quickly switch between different preference sets without re-entering data. One profile can be marked default.

### 3. ML Recommendations
**Model:** Random Forest Classifier
**Input:** 15 logical fields (face shape plus gender, hair type/length/color, volume, thickness, texture, condition, lifestyle, maintenance, styling time, style preference, occasions, bangs).
**Output:** Top 10 ranked hairstyles based on probability scores over 55 classes.

### 4. Gemini AI Integration
**Text Generation:**
- Generates personalized "Why this works for you" explanations.
- Creates custom maintenance schedules.
- Suggests specific products based on hair type.

**Image Generation (Overlays):**
- Basic mode: local PIL compositing (always available).
- Advanced mode: Gemini WebAPI synthesizes the recommended hairstyle onto the uploaded photo with automatic blending and lighting handling (10-30 seconds, downloadable result).

---

## Database Schema

### Enhanced Models

#### `PreferenceProfile`
Stores named sets of user preferences.
- `user`: FK to User
- `profile_name`: String (for example "Work Mode")
- `is_default`: Boolean
- Preference fields: gender, hair type/length/color, volume, thickness, texture, lifestyle, maintenance, styling time, style preference, bangs, conditions and occasions (JSON)

#### `SavedHairstyle`
Allows users to save favorites.
- `user`: FK to User
- `hairstyle`: FK to Hairstyle
- `notes`: User's personal notes
- `recommendation_data`: Snapshot of why it was recommended
- `face_shape` plus confidence and `overlay_url` context

#### `HairstyleLike`
Tracks user feedback.
- `reaction`: Like/Dislike (one per user per style, feedback tracking only)
- Backed by `Feedback` records with 1-5 ratings for accuracy and satisfaction

#### `AnalyticsEvent`
Tracks system usage.
- `event_type`: Page view, Recommendation, Overlay, etc.
- `session_id`: For anonymous tracking

Supporting models: `CustomUser`/`UserProfile`, `UploadedImage`, `UserPreference`, `Hairstyle`/`HairstyleCategory`, `RecommendationLog`, `CachedRecommendation`. See the root `README.md` Database Schema section for the full map.

---

## API Reference

Base URL: `http://<host>:<port>/api/`. Interactive docs at `/api/docs/` (Swagger) and `/api/redoc/`. Auth: `Authorization: Bearer <access_token>`.

### Auth
- `POST /api/auth/signup/`, `POST /api/auth/login/`, `POST /api/auth/logout/`, `POST /api/auth/refresh/`, `GET /api/auth/profile/`

### Core Flow
- `POST /api/upload/` (multipart image), `POST /api/preferences/`, `POST /api/recommend/` (body `{ image_id, preference_id }`)

### AI Endpoints

#### Get Hairstyle Details with AI
`GET /api/hairstyles/<uuid>/details/`
- **Params:** `preference_id`, `image_id`
- **Returns:** Hairstyle info plus AI-generated tips, products, and personalized description.

#### Generate Overlay
`POST /api/overlay/auto/` (requires auth)
- **Body:** `{ "image_id": "...", "hairstyle_id": "..." }`
- **Returns:** URL of the generated overlay image.

### User Management

#### Preference Profiles
- `GET /api/preference-profiles/`: List all profiles
- `POST /api/preference-profiles/`: Create new profile
- `POST /api/preference-profiles/<id>/set-default/`: Set active profile

#### Saved Hairstyles
- `GET /api/saved-hairstyles/`: List favorites
- `POST /api/saved-hairstyles/`: Save a style
- `DELETE /api/saved-hairstyles/<id>/`: Remove from favorites

Full payload reference: `backend/API_DOCS.md`.

---

## Setup and Configuration

### Gemini Setup
1. Obtain `GEMINI_API_KEY` from Google AI Studio.
2. (Optional) Obtain `GEMINI_SECURE_1PSID` / `GEMINI_SECURE_1PSIDTS` cookies for advanced image generation.
3. Add to `.env` (see root `.env.example`):
   ```env
   GEMINI_API_KEY=...
   GEMINI_MODEL_NAME=gemini-2.0-flash
   OVERLAY_AI_ENABLED=true
   ```

### Running the System
1. **Backend:**
   ```powershell
   cd backend
   ..\venv311\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python manage.py migrate
   python scripts/repopulate_hairstyles.py
   python manage.py populate_discover_hairstyles
   python manage.py runserver
   ```
   ML weights are not committed to git; download link and target paths are in `backend/README.md`.
2. **Frontend:**
   ```bash
   cd frontend
   npm install
   npm start
   ```

---
*HairMixer system docs v2.1.0. End-user guide: `docs/USER_MANUAL.md`. Endpoint payloads: `backend/API_DOCS.md`. Overlay modes: `backend/OVERLAY_SETUP.md`.*
