# 💇 HairMixer System Documentation

> Comprehensive documentation for the AI-Powered Hairstyle Recommendation System

**Version:** 2.1.0  
**Last Updated:** December 2025  
**Status:** Production Ready

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Features](#core-features)
4. [AI & ML Integration](#ai--ml-integration)
5. [API Reference](#api-reference)
6. [Database Schema](#database-schema)
7. [Setup & Deployment](#setup--deployment)

---

## 🎯 System Overview

HairMixer is an intelligent hairstyle recommendation platform that uses machine learning, computer vision, and Generative AI to provide personalized hairstyle suggestions and visualizations.

### Key Capabilities

1. **Face Shape Detection**: ResNet50 analysis of user photos.
2. **ML Recommendations**: Random Forest model matching 21 user preferences to hairstyle families.
3. **Generative AI Insights**: Google Gemini-powered styling tips, maintenance guides, and product recommendations.
4. **AI Overlays**: Generative visualization of hairstyles on user photos.
5. **User Personalization**: Saved profiles, favorites, and history tracking.

---

## 🏗️ Architecture

### Technology Stack

#### Backend
- **Framework:** Django 4.2 + DRF
- **ML:** scikit-learn, PyTorch, MediaPipe
- **AI:** Google Gemini API (Text), Gemini WebAPI (Image)
- **Database:** SQLite (Dev) / PostgreSQL (Prod)

#### Frontend
- **Framework:** React 19
- **Styling:** TailwindCSS

### System Architecture Diagram

```
┌─────────────┐
│   Client    │ (React Frontend)
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────────┐
│           Django Backend (API)                   │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────┐ │
│  │  ML Services │  │  AI Services │  │   DB   │ │
│  │ ──────────── │  │ ──────────── │  │ ────── │ │
│  │ • ResNet50   │  │ • Gemini API │  │ • User │ │
│  │ • Random     │  │   (Text)     │  │ • Style│ │
│  │   Forest     │  │ • Gemini Web │  │ • Prof │ │
│  │              │  │   (Image)    │  │ • Save │ │
│  └──────────────┘  └──────────────┘  └────────┘ │
└─────────────────────────────────────────────────┘
```

---

## ⚡ Core Features

### 1. Face Shape Detection
**Model:** ResNet50 + MediaPipe  
**Shapes:** Oval, Round, Square, Heart, Oblong  
**Process:** Detects face landmarks and classifies shape with confidence score.

### 2. Preference Profiles
Users can create multiple named profiles (e.g., "Professional", "Casual") to quickly switch between different preference sets without re-entering data.

### 3. ML Recommendations
**Model:** Random Forest Classifier  
**Input:** 21 features (Hair type, texture, lifestyle, etc.)  
**Output:** Top 10 ranked hairstyles based on probability scores.

### 4. Gemini AI Integration
**Text Generation:**
- Generates personalized "Why this works for you" explanations.
- Creates custom maintenance schedules.
- Suggests specific products based on hair type.

**Image Generation (Overlays):**
- Uses Gemini WebAPI to synthesize the recommended hairstyle onto the user's uploaded photo.
- Handles blending and lighting adjustments automatically.

---

## 🗄️ Database Schema

### Enhanced Models

#### `PreferenceProfile`
Stores named sets of user preferences.
- `user`: FK to User
- `profile_name`: String (e.g., "Work Mode")
- `is_default`: Boolean
- `preferences`: JSON (All 21 preference fields)

#### `SavedHairstyle`
Allows users to save favorites.
- `user`: FK to User
- `hairstyle`: FK to Hairstyle
- `notes`: User's personal notes
- `recommendation_data`: Snapshot of why it was recommended

#### `HairstyleLike`
Tracks user feedback.
- `reaction`: Like/Dislike
- Used for future model retraining.

#### `AnalyticsEvent`
Tracks system usage.
- `event_type`: Page view, Recommendation, Overlay, etc.
- `session_id`: For anonymous tracking.

---

## 📡 API Reference

### AI Endpoints

#### Get Hairstyle Details with AI
`GET /api/hairstyles/<uuid>/details/`
- **Params:** `preference_id`, `image_id`
- **Returns:** Hairstyle info + AI-generated tips, products, and personalized description.

#### Generate Overlay
`POST /api/overlay/auto/`
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

---

## 🔧 Setup & Configuration

### Gemini Setup
1. Obtain `GEMINI_API_KEY` from Google AI Studio.
2. (Optional) Obtain `GEMINI_SECURE_1PSID` cookies for image generation.
3. Add to `.env`:
   ```env
   GEMINI_API_KEY=...
   OVERLAY_AI_ENABLED=true
   ```

### Running the System
1. **Backend:** `python manage.py runserver`
2. **Frontend:** `npm start`

