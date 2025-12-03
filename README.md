# 💇 HairMixer - AI-Powered Hairstyle Recommendation System

> Intelligent hairstyle recommendations using Random Forest ML, ResNet50 face detection, and Google Gemini AI

![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![React](https://img.shields.io/badge/react-19.0-blue.svg)
![Django](https://img.shields.io/badge/django-4.2-green.svg)
![ML](https://img.shields.io/badge/ML-Random_Forest-orange.svg)
![AI](https://img.shields.io/badge/AI-Gemini_Pro-purple.svg)

---

## 🎯 What is HairMixer?

HairMixer is an intelligent hairstyle recommendation platform that combines:
- 🤖 **AI Face Shape Detection** (ResNet50 face shape detection of 5 shapes)
- 🧠 **ML Recommendations** (Random Forest with 55 hairstyles)
- ✨ **Generative AI** (Google Gemini for personalized styling tips and overlays)
- 🎨 **Personalization** (21 preference features)
- 📱 **Modern UI** (React 19 + TailwindCSS)

### Key Features

✅ **Automatic Face Analysis** - Upload a photo, get your face shape  
✅ **Smart Recommendations** - ML model suggests top 10 hairstyles  
✅ **AI-Powered Insights** - Personalized styling tips, product recommendations, and maintenance guides via Gemini  
✅ **AI Overlays** - Visualize hairstyles on your own photo using Generative AI  
✅ **Preference Profiles** - Save multiple style profiles (e.g., "Work", "Party") for quick access  
✅ **Save & Organize** - Save favorite hairstyles with notes and track your history  
✅ **Community Feedback** - Like/Dislike styles to improve recommendations  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- pip & npm
- Google Gemini API Key (optional, for AI features)

### 1. Clone Repository

```bash
git clone <repository-url>
cd HairMixer
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv311
# Windows
.\venv311\Scripts\activate
# Linux/Mac
source venv311/bin/activate

pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python populate_all_hairstyles.py  # Optional: seed database
python manage.py runserver
```

Backend runs at `http://localhost:8000`

### 3. Frontend Setup

```bash
cd ../frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`

---

## 📊 System Architecture

```
User Photo → ResNet50 → Face Shape Detection
     ↓
User Preferences (21 features) → Random Forest ML Model
     ↓
Predict Hairstyle Family → Database Query
     ↓
Top 10 Recommendations
     ↓
Google Gemini AI (Optional)
  ├─ Generates Personalized Tips
  ├─ Creates Maintenance Guides
  └─ Generates Hairstyle Overlays
```

### Tech Stack

**Backend:** Django 4.2, DRF, scikit-learn 1.7.2, PyTorch, MediaPipe  
**AI Services:** Google Gemini API (Text), Gemini WebAPI (Image Generation)  
**Frontend:** React 19, TailwindCSS, Axios  
**Database:** SQLite (dev) / PostgreSQL (prod)  
**ML Models:** ResNet50 (Face Shape), Random Forest (Recommendations)

---

## 🎮 User Flow

1. **Upload Photo** → AI detects face shape (5 types)
2. **Set Preferences** → Use Wizard or select a saved **Preference Profile**
3. **Get Results** → View top 10 recommended hairstyles
4. **Explore Details** → Get AI-generated styling tips and product advice
5. **Visualize** → Generate AI preview overlay on your photo
6. **Save** → Save favorites to your collection

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload/` | POST | Upload image & detect face shape |
| `/api/recommend/` | POST | Get recommendations (full flow) |
| `/api/hairstyles/<id>/details/` | GET | Get details with Gemini AI insights |
| `/api/overlay/auto/` | POST | Generate AI overlay |
| `/api/preference-profiles/` | GET/POST | Manage user preference profiles |
| `/api/saved-hairstyles/` | GET/POST | Manage saved favorites |

---

## 🤖 ML & AI Model Details

### Recommendation Engine
- **Type:** Random Forest Classifier
- **Features:** 21 user inputs (Face shape, hair texture, lifestyle, etc.)
- **Output:** Hairstyle Family Probabilities

### Computer Vision
- **Model:** ResNet50
- **Task:** Face Shape Classification (Oval, Round, Square, Heart, Oblong)

### Generative AI
- **Service:** Google Gemini
- **Tasks:** 
  - Context-aware styling advice
  - Product recommendations based on hair type
  - Image synthesis for hairstyle overlays

---

## 📁 Project Structure

```
HairMixer/
├── backend/
│   ├── hairmixer_app/
│   │   ├── models.py         # Enhanced models (Profiles, Saved Styles)
│   │   ├── services/         # Business logic
│   │   │   ├── gemini_service.py  # AI Text Generation
│   │   │   └── overlay.py         # AI Image Generation
│   │   └── ml/               # ML models
│   └── ...
├── frontend/
│   ├── src/
│   │   ├── components/       # React 19 components
│   │   └── ...
└── ...
```

---

## 🔧 Configuration

### Backend (.env)

```env
SECRET_KEY=your-secret-key
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3

# Gemini AI Configuration
GEMINI_API_KEY=your_api_key_here
OVERLAY_AI_ENABLED=true
```

---

## 📈 Database Schema

### Core Models
- `Hairstyle`: 55+ styles with metadata
- `UserPreference`: Detailed user inputs
- `UploadedImage`: User photos + analysis

### User Features
- `PreferenceProfile`: Named presets (e.g., "Summer Look")
- `SavedHairstyle`: Favorites with notes
- `HairstyleLike`: User feedback tracking
- `Feedback`: Detailed review system

---

## 👥 Team

John Mathew Mauricio
Theris Eldrene Carroz
Jazzer Ong

---

<div align="center">

**Made with ❤️ for better hairstyle decisions**

</div>
