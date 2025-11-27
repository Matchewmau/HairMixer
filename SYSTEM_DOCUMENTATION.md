# 💇 HairMixer System Documentation

> Comprehensive documentation for the AI-Powered Hairstyle Recommendation System

**Version:** 2.0.0  
**Last Updated:** December 2024  
**Status:** Production Ready

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Features](#core-features)
4. [ML Model](#ml-model)
5. [API Reference](#api-reference)
6. [Database Schema](#database-schema)
7. [Setup & Deployment](#setup--deployment)
8. [User Flow](#user-flow)
9. [Development Guide](#development-guide)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 System Overview

HairMixer is an intelligent hairstyle recommendation platform that uses machine learning and computer vision to provide personalized hairstyle suggestions.

### What It Does

1. **Detects Face Shape** - Automatically analyzes uploaded photos using ResNet50
2. **Collects Preferences** - Gathers 21 user preferences through intuitive UI
3. **Recommends Styles** - Uses Random Forest ML model to suggest top 10 hairstyles
4. **Shows Previews** - (Future: AI overlay generation)

### Key Statistics

- **26 Hairstyle Families** predicted by ML model
- **21 User Features** analyzed for recommendations
- **10 Personalized Recommendations** per request
- **7 Face Shapes** detected automatically
- **9 Hairstyle Categories** in database

---

## 🏗️ Architecture

### Technology Stack

#### Backend
- **Framework:** Django 4.2 + Django REST Framework
- **Database:** SQLite (dev) / PostgreSQL (prod)
- **ML Libraries:**
  - scikit-learn 1.7.2 (Random Forest Classifier)
  - PyTorch (ResNet50 face detection)
  - MediaPipe (Facial landmarks)
  - OpenCV (Image processing)
- **Authentication:** JWT tokens
- **API Documentation:** drf-spectacular (OpenAPI/Swagger)

#### Frontend
- **Framework:** React 18
- **Routing:** React Router v6
- **Styling:** TailwindCSS
- **HTTP Client:** Axios
- **Build Tool:** Create React App

#### ML Models
1. **hairstyle_family_model.pkl** - Random Forest Classifier
   - **Input:** 21 numerical features
   - **Output:** 26 hairstyle family probabilities
   - **Size:** ~500KB
   - **Training Data:** Merged preferences CSV

2. **ResNet50** - Face Shape Detector
   - **Input:** Face image
   - **Output:** 7 face shape classes + confidence
   - **Shapes:** oval, round, square, heart, diamond, oblong, triangle

### System Architecture Diagram

```
┌─────────────┐
│   Client    │ (React Frontend)
│  (Browser)  │
└──────┬──────┘
       │ HTTP/REST
       │
┌──────▼──────────────────────────────────────────┐
│           Django Backend (API)                   │
│                                                  │
│  ┌────────────────────────────────────────┐    │
│  │     API Endpoints (views.py)           │    │
│  │  - /api/upload/                        │    │
│  │  - /api/preferences/                   │    │
│  │  - /api/recommend/                     │    │
│  │  - /api/recommend/ml/                  │    │
│  └───┬────────────────────────────────────┘    │
│      │                                          │
│  ┌───▼────────────────────┐  ┌──────────────┐ │
│  │  Services Layer        │  │   ML Models  │ │
│  │  ────────────────      │  │  ──────────  │ │
│  │  • RecommendationSvc   │◄─┤  ResNet50    │ │
│  │  • HairstyleRecommender│  │  (Face Det.) │ │
│  │  • ImageService        │  │              │ │
│  │  • OverlayService      │  │  RF Model    │ │
│  └───┬────────────────────┘  │  (Recommend) │ │
│      │                        └──────────────┘ │
│  ┌───▼────────────────────┐                    │
│  │    Database (SQLite)   │                    │
│  │  ────────────────────  │                    │
│  │  • Hairstyle (styles)  │                    │
│  │  • UserPreference      │                    │
│  │  • UploadedImage       │                    │
│  │  • RecommendationLog   │                    │
│  └────────────────────────┘                    │
└─────────────────────────────────────────────────┘
```

### Request Flow

```
User → Upload Image → Django → ResNet50 → Face Shape
                         ↓
User → Fill Preferences (10 steps) → Django → Save Preferences
                                        ↓
User → Request Recommendations → Django → HairstyleRecommender
                                            ↓
                               Random Forest Model (hairstyle_family_model.pkl)
                                            ↓
                               Predict 26 families with confidence scores
                                            ↓
                               Query Database (filter by gender + preferences)
                                            ↓
                               Return Top 10 Hairstyles → User
```

---

## ⚡ Core Features

### 1. Face Shape Detection

**Technology:** ResNet50 + MediaPipe

**Capabilities:**
- Automatic face shape classification
- 7 face shape categories
- Confidence score (0-1)
- Facial landmark detection

**Face Shapes Detected:**
1. Oval
2. Round
3. Square
4. Heart
5. Diamond
6. Oblong
7. Triangle

**API Endpoint:** `POST /api/upload/`

**Process:**
1. User uploads image
2. ResNet50 detects face region
3. MediaPipe extracts facial landmarks
4. Model predicts face shape
5. Returns shape + confidence + facial features

### 2. User Preference Collection

**Total Features:** 21 (used by ML model)

**Preference Categories:**

#### A. Core Features (6)
1. **Gender** - male, female, nb, other
2. **Hair Type** - straight, wavy, curly, coily
3. **Hair Length** - pixie, short, medium, long, extra_long
4. **Face Shape** - (auto-detected or manual)
5. **Maintenance** - low, medium, high
6. **Lifestyle** - active, professional, creative, casual

#### B. Detailed Features (6)
7. **Volume** - flat, light, medium, high
8. **Styling Maintenance** - low, medium, high
9. **Styling Preference** - natural, casual, classic, elegant, trendy, edgy
10. **Hair Condition** - excellent, good, fair, damaged
11. **Hair Thickness** - fine/thin, medium, thick, very_thick
12. **Hair Texture Detail** - smooth, coarse, silky, frizzy, normal

#### C. Binary Feature (1)
13. **Wants Bangs** - true/false

#### D. Multi-Select Features (8)
14-21. **Occasions** (can select multiple):
- Work
- Casual
- Formal
- Date
- Exercise
- Travel
- Party
- Wedding

**UI Implementation:** 10-step wizard in React

**API Endpoint:** `POST /api/preferences/`

### 3. ML-Based Recommendations

**Model:** Random Forest Classifier (`hairstyle_family_model.pkl`)

**Approach:** Hierarchical Family-Based Prediction

**How It Works:**

1. **Feature Encoding**
   - Converts 21 user preferences into numerical vector
   - Categorical features → integers (0-N)
   - Boolean features → 0 or 1
   - Multi-select → binary encoding

2. **Family Prediction**
   - Model predicts probabilities for 26 hairstyle families
   - Sorts by confidence (highest first)
   - Top families allocated more recommendation slots

3. **Database Query**
   - Maps families (0-25) to database categories (1-9)
   - Filters by user gender (male/female/unisex match)
   - Applies preference filters (length, face shape)

4. **Recommendation Assembly**
   - Selects diverse hairstyles from top families
   - Returns top 10 with match scores
   - Includes metadata (difficulty, time, tags)

**API Endpoints:**
- `POST /api/recommend/` - Image + preferences (full flow)
- `POST /api/recommend/ml/` - Preferences only (faster)

**Output Format:**
```json
{
  "recommendation_count": 10,
  "recommendations": [
    {
      "id": "uuid",
      "name": "Hairstyle Name",
      "description": "Description text",
      "image_url": "/media/path/to/image.jpg",
      "category": "Category Name",
      "hairstyle_family": "5",
      "difficulty": "medium",
      "estimated_time": 30,
      "maintenance": "medium",
      "tags": ["tag1", "tag2"],
      "match_score": 0.857,
      "confidence": 85.7
    },
    // ... 9 more
  ],
  "model_used": "hairstyle_family_model",
  "faceshape": "oval",
  "faceshape_confidence": 0.92
}
```

### 4. Gender-Based Filtering

**Critical Feature:** Ensures recommendations match user's gender

**Logic:**
- Male users → male or unisex styles only
- Female users → female or unisex styles only
- Non-binary/other → unisex styles only

**Database Field:** `Hairstyle.suitable_gender`

**Filtering Levels:**
1. **Database Query** - Primary filter (performance)
2. **Post-Query Filter** - Secondary filter (accuracy)

---

## 🤖 ML Model

### hairstyle_family_model.pkl

**Model Type:** Random Forest Classifier

**Training Details:**
- **Algorithm:** sklearn.ensemble.RandomForestClassifier
- **Features:** 21 numerical inputs
- **Classes:** 26 hairstyle families (0-25)
- **Training Data:** Merged preferences CSV file
- **scikit-learn Version:** 1.7.2

### Feature Engineering

**Input Vector (21 features):**

```python
[
    gender,              # 0-3
    hair_type,           # 0-3
    hair_length,         # 0-4
    faceshape,           # 0-6
    maintenance,         # 0-2
    lifestyle,           # 0-3
    volume,              # 0-3
    styling_maintenance, # 0-2
    styling_preference,  # 0-4
    hair_condition,      # 0-3
    hair_thickness,      # 0-3
    hair_texture_detail, # 0-4
    wants_bangs,         # 0-1
    work,                # 0-1
    casual,              # 0-1
    formal,              # 0-1
    date,                # 0-1
    exercise,            # 0-1
    travel,              # 0-1
    party,               # 0-1
    wedding              # 0-1
]
```

### Family-to-Category Mapping

The model predicts 26 families, which map to 9 database categories:

```python
MODEL_TO_DB_CATEGORY = {
    # Short styles (category 1)
    0: 1, 1: 1, 2: 1,
    # Long styles (category 2)
    3: 2, 4: 2, 5: 2,
    # Medium styles (category 3)
    6: 3, 7: 3, 8: 3,
    # Curly styles (category 4)
    9: 4, 10: 4, 11: 4,
    # Formal styles (category 5)
    12: 5, 13: 5, 14: 5,
    # Classic styles (category 6)
    15: 6, 16: 6, 17: 6,
    # Trendy styles (category 7)
    18: 7, 19: 7, 20: 7,
    # Retro styles (category 8)
    21: 8, 22: 8, 23: 8,
    # Casual styles (category 9)
    24: 9, 25: 9
}
```

### Prediction Strategy

1. **Get Probabilities:** `model.predict_proba(features)`
2. **Sort by Confidence:** Highest first
3. **Allocate Slots:**
   - Top family: 3-4 styles
   - Next 2-3 families: 2 styles each
   - Remaining: 1 style each
4. **Query Database:** For each allocated family
5. **Apply Filters:** Gender, length, face shape
6. **Return Diverse Set:** Top 10 unique styles

### Model Performance

**No Fallbacks:** Model-only approach
- If model fails → return empty list
- No rule-based fallbacks
- No static placeholder recommendations

---

## 📡 API Reference

### Base URL
- **Development:** `http://localhost:8000/api`
- **Production:** `https://yourdomain.com/api`

### Authentication

**Method:** JWT (JSON Web Tokens)

**Headers:**
```
Authorization: Bearer <token>
```

**Login:** `POST /auth/login/`
**Register:** `POST /auth/register/`

### Core Endpoints

#### 1. Upload Image

**Endpoint:** `POST /api/upload/`

**Request:**
```json
Content-Type: multipart/form-data

{
  "image": <file>
}
```

**Response:**
```json
{
  "image_id": "uuid",
  "image_url": "/media/uploads/...",
  "upload_date": "2024-12-...",
  "face_shape": "oval",
  "face_shape_confidence": 0.92,
  "facial_features": {
    "face_ratio": 1.42,
    "jawline_strength": "medium",
    "symmetry_score": 0.88
  }
}
```

#### 2. Save Preferences

**Endpoint:** `POST /api/preferences/`

**Request:**
```json
{
  "gender": "female",
  "hair_type": "wavy",
  "hair_length": "medium",
  "faceshape": "oval",
  "maintenance": "medium",
  "lifestyle": "professional",
  "volume": "medium",
  "styling_maintenance": "low",
  "styling_preference": "elegant",
  "hair_condition": "good",
  "hair_thickness": "medium",
  "hair_texture_detail": "smooth",
  "wants_bangs": false,
  "occasions": ["work", "formal", "date"]
}
```

**Response:**
```json
{
  "preference_id": "uuid",
  "message": "Preferences saved successfully",
  "preferences": { /* all fields echoed back */ }
}
```

#### 3. Get Recommendations (Full Flow)

**Endpoint:** `POST /api/recommend/`

**Request:**
```json
{
  "image_id": "uuid",
  "preference_id": "uuid"
}
```

**Response:** (See "ML-Based Recommendations" section for full format)

#### 4. Get ML Recommendations (Fast)

**Endpoint:** `POST /api/recommend/ml/`

**Request:**
```json
{
  "preference_id": "uuid"
}
```

**Response:** Same as `/api/recommend/` but faster (no image processing)

#### 5. List Hairstyles

**Endpoint:** `GET /api/hairstyles/`

**Query Parameters:**
- `category_id` - Filter by category
- `gender` - Filter by gender (male/female/unisex)
- `difficulty` - Filter by difficulty
- `search` - Search by name/description

**Response:**
```json
{
  "count": 150,
  "results": [
    {
      "id": "uuid",
      "name": "Classic Bob",
      "description": "...",
      "category": "Classic",
      "image_url": "/media/...",
      "difficulty": "easy",
      "maintenance": "medium",
      "estimated_time": 30,
      "tags": ["professional", "versatile"],
      "suitable_gender": "female"
    },
    // ...
  ]
}
```

#### 6. Get Hairstyle Detail

**Endpoint:** `GET /api/hairstyles/<uuid>/`

**Response:** Single hairstyle object with full details

### Additional Endpoints

- `GET /api/hairstyles/featured/` - Featured hairstyles
- `GET /api/hairstyles/trending/` - Trending hairstyles
- `GET /api/hairstyles/categories/` - List categories
- `GET /api/filter/face-shapes/` - Available face shapes
- `GET /api/filter/occasions/` - Available occasions
- `GET /api/health/` - API health check
- `GET /api/docs/` - OpenAPI/Swagger documentation

---

## 🗄️ Database Schema

### Tables

#### 1. Hairstyle

Primary table storing all hairstyle information.

**Fields:**
- `id` (UUID) - Primary key
- `name` (String) - Hairstyle name
- `description` (Text) - Detailed description
- `category` (FK) - HairstyleCategory foreign key
- `suitable_gender` (String) - male/female/unisex
- `image` (ImageField) - Hairstyle image
- `image_url` (String) - Alternative image URL
- `difficulty` (String) - easy/medium/hard/professional
- `maintenance` (String) - low/medium/high
- `estimated_time` (Int) - Minutes needed
- `trend_score` (Float) - 0-10 popularity
- `popularity_score` (Float) - 0-10 rating
- `tags` (JSON) - Array of tags
- `face_shapes` (JSON) - Compatible face shapes
- `hair_types` (JSON) - Compatible hair types
- `hair_lengths` (JSON) - Compatible lengths
- `occasions` (JSON) - Suitable occasions
- `is_active` (Boolean) - Active status
- `is_featured` (Boolean) - Featured status
- `created_at` (DateTime)
- `updated_at` (DateTime)

#### 2. UserPreference

Stores user's hairstyle preferences (21 features).

**Fields:**
- `id` (UUID) - Primary key
- `user` (FK) - User foreign key (nullable)
- `faceshape` (String) - Auto-detected or manual
- `faceshape_confidence` (Float) - Detection confidence
- `gender` (String) - male/female/nb/other
- `hair_type` (String) - straight/wavy/curly/coily
- `hair_length` (String) - pixie/short/medium/long/extra_long
- `maintenance` (String) - low/medium/high
- `lifestyle` (String) - active/professional/creative/casual
- `volume` (String) - flat/light/medium/high
- `styling_maintenance` (String) - low/medium/high
- `styling_preference` (String) - natural/casual/classic/elegant/trendy/edgy
- `hair_condition` (String) - excellent/good/fair/damaged
- `hair_thickness` (String) - fine/medium/thick/very_thick
- `hair_texture_detail` (String) - smooth/coarse/silky/frizzy/normal
- `wants_bangs` (Boolean)
- `occasions` (JSON) - Array of occasion strings
- `created_at` (DateTime)
- `updated_at` (DateTime)

#### 3. UploadedImage

Stores uploaded user images.

**Fields:**
- `id` (UUID) - Primary key
- `user` (FK) - User foreign key (nullable)
- `image` (ImageField) - Uploaded image
- `upload_date` (DateTime)
- `face_shape` (String) - Detected face shape
- `face_shape_confidence` (Float)
- `facial_features` (JSON) - Detected features

#### 4. RecommendationLog

Tracks all recommendations generated.

**Fields:**
- `id` (UUID) - Primary key
- `user` (FK) - User foreign key
- `uploaded` (FK) - UploadedImage foreign key
- `preference` (FK) - UserPreference foreign key
- `face_shape` (String)
- `face_shape_confidence` (Float)
- `detected_features` (JSON)
- `selected_hairstyle` (FK) - Top recommendation
- `candidates` (JSON) - Array of hairstyle IDs
- `recommendation_scores` (JSON) - ID → score mapping
- `status` (String) - pending/completed/failed
- `processing_time` (Float) - Seconds
- `model_version` (String)
- `view_count` (Int)
- `created_at` (DateTime)

#### 5. HairstyleCategory

Categories for organizing hairstyles.

**Fields:**
- `id` (Int) - Primary key
- `name` (String) - Category name
- `description` (Text)
- `parent` (FK) - Self-referential for hierarchy
- `is_active` (Boolean)
- `sort_order` (Int)
- `created_at` (DateTime)

### Database Categories

Current 9 categories:
1. Short
2. Long
3. Medium
4. Curly
5. Formal
6. Classic
7. Trendy
8. Retro
9. Casual

---

## 🚀 Setup & Deployment

### Prerequisites

- Python 3.11+
- Node.js 18+
- pip
- npm/yarn
- Git

### Backend Setup

1. **Clone Repository**
```bash
git clone <repository-url>
cd HairMixer/backend
```

2. **Create Virtual Environment**
```bash
python -m venv venv311
# Windows
.\venv311\Scripts\activate
# Linux/Mac
source venv311/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Variables**
Create `.env` file:
```env
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:///db.sqlite3
```

5. **Run Migrations**
```bash
python manage.py migrate
```

6. **Create Superuser**
```bash
python manage.py createsuperuser
```

7. **Populate Database** (optional)
```bash
python populate_all_hairstyles.py
```

8. **Run Server**
```bash
python manage.py runserver
```

Backend will run at `http://localhost:8000`

### Frontend Setup

1. **Navigate to Frontend**
```bash
cd ../frontend
```

2. **Install Dependencies**
```bash
npm install
```

3. **Environment Variables**
Create `.env.local`:
```env
REACT_APP_API_BASE_URL=http://localhost:8000/api
```

4. **Run Development Server**
```bash
npm start
```

Frontend will run at `http://localhost:3000`

### Production Deployment

#### Backend (Django)

1. **Configure Settings**
   - Set `DEBUG=False`
   - Update `ALLOWED_HOSTS`
   - Configure production database (PostgreSQL)
   - Set up static/media file serving

2. **Collect Static Files**
```bash
python manage.py collectstatic
```

3. **Deploy Options**
   - **Heroku:** Use Procfile
   - **Docker:** Use Dockerfile
   - **AWS:** EC2 + RDS
   - **DigitalOcean:** App Platform

#### Frontend (React)

1. **Build for Production**
```bash
npm run build
```

2. **Deploy Options**
   - **Netlify:** Connect GitHub repo
   - **Vercel:** Import project
   - **AWS S3 + CloudFront**
   - **Serve with Django:** Place build in Django static

---

## 👤 User Flow

### Complete User Journey

```
┌───────────────────────────────────────────────────────────┐
│  1. Landing Page                                          │
│     - View features                                       │
│     - Click "Get Started" or "Try Now"                    │
└─────────────────────┬─────────────────────────────────────┘
                      ↓
┌───────────────────────────────────────────────────────────┐
│  2. Upload Photo                                          │
│     - Drag & drop or browse for image                     │
│     - System analyzes face shape (ResNet50)               │
│     - Shows detected face shape + confidence              │
└─────────────────────┬─────────────────────────────────────┘
                      ↓
┌───────────────────────────────────────────────────────────┐
│  3. Preferences Wizard (10 Steps)                         │
│     Step 1: Gender selection                              │
│     Step 2: Hair type (straight/wavy/curly/coily)        │
│     Step 3: Current hair length                           │
│     Step 4: Hair thickness                                │
│     Step 5: Hair texture detail                           │
│     Step 6: Lifestyle (active/professional/creative)      │
│     Step 7: Maintenance preference                        │
│     Step 8: Styling preference (natural/elegant/trendy)   │
│     Step 9: Occasions (multi-select)                      │
│     Step 10: Hair condition                               │
│                                                           │
│     - Progress indicator (Step X of 10)                   │
│     - Back/Next navigation                                │
│     - Visual, emoji-enhanced selection                    │
└─────────────────────┬─────────────────────────────────────┘
                      ↓
┌───────────────────────────────────────────────────────────┐
│  4. Loading / Processing                                  │
│     - "Analyzing your preferences..."                     │
│     - ML model generates recommendations                  │
│     - Takes 1-3 seconds                                   │
└─────────────────────┬─────────────────────────────────────┘
                      ↓
┌───────────────────────────────────────────────────────────┐
│  5. Results Page                                          │
│     - Shows detected face shape                           │
│     - Displays top 10 recommended hairstyles              │
│     - Each hairstyle shows:                               │
│       • Image                                             │
│       • Name & description                                │
│       • Match score (%)                                   │
│       • Difficulty & time estimate                        │
│       • Tags                                              │
│       • "View Details" button                             │
│     - Option to try different preferences                 │
└─────────────────────┬─────────────────────────────────────┘
                      ↓
┌───────────────────────────────────────────────────────────┐
│  6. Future: Preview Generation (Overlay AI)               │
│     - Select a hairstyle                                  │
│     - Generate AI preview on user's photo                 │
│     - Save favorites                                      │
└───────────────────────────────────────────────────────────┘
```

### Alternative Flow: Browse Hairstyles

```
Landing Page → Discover Page → Filter/Search → View Details
```

---

## 💻 Development Guide

### Project Structure

```
HairMixer/
├── backend/
│   ├── backend/                 # Django project settings
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── hairmixer_app/          # Main Django app
│   │   ├── models.py           # Database models
│   │   ├── views.py            # API views
│   │   ├── serializers.py      # DRF serializers
│   │   ├── urls.py             # App URLs
│   │   ├── services/           # Business logic
│   │   │   ├── hairstyle_recommender.py
│   │   │   ├── recommendation_service.py
│   │   │   ├── image_service.py
│   │   │   └── overlay_service.py
│   │   ├── ml/                 # ML components
│   │   │   ├── face_analyzer.py
│   │   │   ├── model.py
│   │   │   └── models/
│   │   │       └── hairstyle_family_model.pkl
│   │   └── migrations/         # Database migrations
│   ├── media/                  # User uploads
│   ├── manage.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── Navbar.js
│   │   │   └── Footer.js
│   │   ├── pages/              # Page components
│   │   │   ├── LandingPage.js
│   │   │   ├── Upload.js
│   │   │   ├── UserPreferences.js
│   │   │   ├── Results.js
│   │   │   └── Discover.js
│   │   ├── services/           # API integration
│   │   │   ├── api.js
│   │   │   └── AuthService.js
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json
│   └── tailwind.config.js
│
├── docs/                       # Documentation
├── README.md
└── SYSTEM_DOCUMENTATION.md     # This file
```

### Adding a New Feature

1. **Backend:**
   - Add model in `models.py`
   - Create migration: `python manage.py makemigrations`
   - Add serializer in `serializers.py`
   - Create view in `views.py` or `views/`
   - Add URL route in `urls.py`
   - Write tests

2. **Frontend:**
   - Create component in `components/` or `pages/`
   - Add route in `App.js`
   - Integrate API call in `services/api.js`
   - Style with TailwindCSS

### Running Tests

**Backend:**
```bash
python manage.py test
python test_hairstyle_recommender.py
python test_random_forest_model.py
```

**Frontend:**
```bash
npm test
```

### Code Style

**Backend:**
- Follow PEP 8
- Use Black formatter
- Max line length: 79

**Frontend:**
- Use ESLint
- Prettier for formatting
- Functional components + hooks

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Not Loading

**Error:** `Model file not found: hairstyle_family_model.pkl`

**Solution:**
- Ensure model file exists at `backend/hairmixer_app/ml/models/hairstyle_family_model.pkl`
- Check file permissions
- Verify scikit-learn version: `pip list | grep scikit-learn`

#### 2. No Recommendations Returned

**Possible Causes:**
1. No hairstyles in database
2. No hairstyles match gender filter
3. Model file missing or corrupted

**Solutions:**
```bash
# Check hairstyle count
python manage.py shell
>>> from hairmixer_app.models import Hairstyle
>>> Hairstyle.objects.filter(is_active=True).count()

# Populate database if needed
python populate_all_hairstyles.py
```

#### 3. CORS Issues

**Error:** `Access to XMLHttpRequest has been blocked by CORS policy`

**Solution:**
Add to `backend/settings.py`:
```python
INSTALLED_APPS = [
    'corsheaders',
    # ...
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    # ...
]

CORS_ALLOWED_ORIGINS = [
    'http://localhost:3000',
]
```

#### 4. Image Upload Fails

**Error:** `413 Request Entity Too Large`

**Solution:**
Adjust max upload size in settings:
```python
DATA_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB
```

#### 5. Face Detection Fails

**Possible Causes:**
- No face in image
- Poor image quality
- Model not loaded

**Solution:**
- Ensure image has clear face
- Check ResNet50 model is loaded
- View logs for detailed error

### Debug Mode

**Enable verbose logging:**

`backend/settings.py`:
```python
LOGGING = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'hairmixer_app': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    },
}
```

### Getting Help

1. Check logs in console
2. Review API documentation at `/api/docs/`
3. Check GitHub issues
4. Contact development team

---

## 📚 Additional Resources

### Related Documentation

- `README.md` - Quick start guide
- `backend/API_DOCS.md` - Detailed API reference
- `docs/ML_RECOMMENDATIONS.md` - ML system deep dive
- `HAIRSTYLE_RECOMMENDATION_CLEANUP.md` - Recent system cleanup

### External Links

- [Django Documentation](https://docs.djangoproject.com/)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [React Documentation](https://react.dev/)
- [TailwindCSS](https://tailwindcss.com/)
- [scikit-learn](https://scikit-learn.org/)

---

## 📝 Changelog

### Version 2.0.0 (December 2024)

**Major Changes:**
- ✅ Implemented ML-only recommendation system
- ✅ Removed all fallback methods and static recommendations
- ✅ Enhanced gender filtering with `suitable_gender` field
- ✅ Upgraded to scikit-learn 1.7.2
- ✅ Cleaned up unnecessary documentation
- ✅ Output exactly 10 recommendations per request

**Improvements:**
- Better family-to-category mapping
- Improved allocation strategy
- Stricter gender filtering
- Enhanced UI/UX in preference wizard

---

**End of Documentation**

For questions or contributions, please contact the development team.
