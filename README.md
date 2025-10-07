# 💇 HairMixer - AI-Powered Hairstyle Recommendation System

> Intelligent hairstyle recommendations using Random Forest ML and ResNet50 face detection

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![React](https://img.shields.io/badge/react-18.0-blue.svg)
![Django](https://img.shields.io/badge/django-4.2-green.svg)
![ML](https://img.shields.io/badge/ML-Random_Forest-orange.svg)

---

## 🎯 What is HairMixer?

HairMixer is an intelligent hairstyle recommendation platform that combines:
- 🤖 **AI Face Shape Detection** (ResNet50)
- 🧠 **ML Recommendations** (Random Forest with 26 families)
- 🎨 **Personalization** (21 preference features)
- 📱 **Modern UI** (React + TailwindCSS)

### Key Features

✅ **Automatic Face Analysis** - Upload a photo, get your face shape  
✅ **Smart Recommendations** - ML model suggests top 10 hairstyles  
✅ **Detailed Preferences** - 10-step wizard collects 21 features  
✅ **Gender-Specific** - Respects gender preferences strictly  
✅ **No Fallbacks** - Pure ML approach, no static recommendations  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- pip & npm

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
Predict 26 Hairstyle Families → Database Query
     ↓
Top 10 Personalized Recommendations
```

### Tech Stack

**Backend:** Django 4.2, DRF, scikit-learn 1.7.2, PyTorch, MediaPipe  
**Frontend:** React 18, TailwindCSS, Axios  
**Database:** SQLite (dev) / PostgreSQL (prod)  
**ML Models:** hairstyle_family_model.pkl (Random Forest), ResNet50

---

## 🎮 User Flow

1. **Upload Photo** → AI detects face shape (7 types)
2. **Set Preferences** → 10-step wizard (gender, hair type, lifestyle, etc.)
3. **Get Results** → View top 10 recommended hairstyles with match scores
4. *(Future)* Generate AI preview overlay on your photo

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload/` | POST | Upload image & detect face shape |
| `/api/preferences/` | POST | Save user preferences |
| `/api/recommend/` | POST | Get recommendations (full flow) |
| `/api/recommend/ml/` | POST | Get recommendations (fast) |
| `/api/hairstyles/` | GET | Browse hairstyles |
| `/api/docs/` | GET | OpenAPI/Swagger documentation |

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/recommend/ml/ \
  -H "Content-Type: application/json" \
  -d '{"preference_id": "uuid-here"}'
```

---

## 🤖 ML Model Details

### hairstyle_family_model.pkl

- **Type:** Random Forest Classifier
- **Features:** 21 numerical inputs
- **Output:** 26 hairstyle family probabilities
- **Training:** Merged preferences CSV dataset
- **Performance:** Pure ML, no fallbacks

### Feature Categories

**Core (6):** gender, hair_type, hair_length, faceshape, maintenance, lifestyle  
**Detailed (6):** volume, styling_maintenance, styling_preference, hair_condition, hair_thickness, hair_texture_detail  
**Binary (1):** wants_bangs  
**Multi-select (8):** occasions (work, casual, formal, date, exercise, travel, party, wedding)

---

## 📁 Project Structure

```
HairMixer/
├── backend/
│   ├── backend/              # Django settings
│   ├── hairmixer_app/        # Main app
│   │   ├── models.py         # Database models
│   │   ├── views.py          # API endpoints
│   │   ├── services/         # Business logic
│   │   └── ml/               # ML models
│   │       └── models/
│   │           └── hairstyle_family_model.pkl
│   ├── media/                # User uploads
│   └── manage.py
│
├── frontend/
│   ├── src/
│   │   ├── pages/            # React pages
│   │   ├── components/       # React components
│   │   └── services/         # API integration
│   └── package.json
│
├── docs/                     # Additional documentation
├── README.md                 # This file
└── SYSTEM_DOCUMENTATION.md   # Comprehensive docs
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[SYSTEM_DOCUMENTATION.md](./SYSTEM_DOCUMENTATION.md)** | 📘 Complete system documentation |
| **[backend/API_DOCS.md](./backend/API_DOCS.md)** | 📡 API reference guide |
| **[docs/ML_RECOMMENDATIONS.md](./docs/ML_RECOMMENDATIONS.md)** | 🤖 ML system deep dive |
| **[HAIRSTYLE_RECOMMENDATION_CLEANUP.md](./HAIRSTYLE_RECOMMENDATION_CLEANUP.md)** | 🧹 Recent cleanup summary |

---

## 🔧 Configuration

### Backend (.env)

```env
SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:///db.sqlite3
```

### Frontend (.env.local)

```env
REACT_APP_API_BASE_URL=http://localhost:8000/api
```

---

## 🧪 Testing

### Backend Tests

```bash
cd backend
python manage.py test
python test_hairstyle_recommender.py
python test_random_forest_model.py
python verify_cleanup.py
```

### Frontend Tests

```bash
cd frontend
npm test
```

---

## 🚀 Deployment

### Backend (Django)

**Options:**
- Heroku (easiest)
- Docker + AWS/DigitalOcean
- VPS with Gunicorn + Nginx

**Steps:**
1. Set `DEBUG=False`
2. Configure production database
3. Run `python manage.py collectstatic`
4. Set up WSGI server (Gunicorn)
5. Configure reverse proxy (Nginx)

### Frontend (React)

**Options:**
- Netlify (recommended)
- Vercel
- AWS S3 + CloudFront
- Serve with Django static files

**Steps:**
1. Run `npm run build`
2. Deploy `build/` folder

---

## 📈 Database Schema

### Core Models

| Model | Description |
|-------|-------------|
| `Hairstyle` | Hairstyle information (150+ styles) |
| `UserPreference` | User's 21 preference features |
| `UploadedImage` | User photos + face analysis |
| `RecommendationLog` | Recommendation history |
| `HairstyleCategory` | 9 style categories |

**See [SYSTEM_DOCUMENTATION.md](./SYSTEM_DOCUMENTATION.md) for full schema details**

---

## 🛠️ Development

### Adding New Features

1. **Backend:** Model → Migration → Serializer → View → URL → Test
2. **Frontend:** Component → API Call → Route → Style

### Code Style

**Backend:** PEP 8, Black formatter, 79 char line limit  
**Frontend:** ESLint, Prettier, functional components

### Git Workflow

```bash
git checkout -b feature/your-feature
# Make changes
git commit -m "feat: description"
git push origin feature/your-feature
# Create pull request
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Model not loading | Check `hairstyle_family_model.pkl` exists in `ml/models/` |
| No recommendations | Run `python populate_all_hairstyles.py` to seed database |
| CORS errors | Add `corsheaders` to Django and configure origins |
| Face detection fails | Ensure clear face in image, check ResNet50 model |

**For detailed troubleshooting, see [SYSTEM_DOCUMENTATION.md](./SYSTEM_DOCUMENTATION.md#troubleshooting)**

---

## 📊 System Statistics

- **26** Hairstyle Families predicted
- **21** User Features analyzed
- **10** Recommendations per request
- **7** Face Shapes detected
- **9** Database Categories
- **150+** Hairstyles in database

---

## 🔄 Recent Changes (v2.0.0)

✅ ML-only recommendation system (no fallbacks)  
✅ Enhanced gender filtering (`suitable_gender` field)  
✅ Upgraded scikit-learn to 1.7.2  
✅ Output exactly 10 recommendations  
✅ Cleaned up documentation  
✅ Removed legacy recommendation engines  

**See [HAIRSTYLE_RECOMMENDATION_CLEANUP.md](./HAIRSTYLE_RECOMMENDATION_CLEANUP.md) for details**

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

### Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation
- Write clear commit messages

---

## 📄 License

[Your License Here]

---

## 👥 Team

[Your Team Information]

---

## 📞 Support

- **Documentation:** [SYSTEM_DOCUMENTATION.md](./SYSTEM_DOCUMENTATION.md)
- **API Docs:** `http://localhost:8000/api/docs/`
- **Issues:** GitHub Issues
- **Email:** [your-email@domain.com]

---

## 🎯 Roadmap

### Current Version (2.0.0)
- ✅ Face shape detection
- ✅ ML recommendations
- ✅ User preferences wizard
- ✅ Gender-aware filtering

### Future Features
- 🔜 AI overlay generation (preview hairstyles on user photo)
- 🔜 Save favorites
- 🔜 User profiles & history
- 🔜 Social sharing
- 🔜 Mobile app

---

## ⭐ Acknowledgments

- **scikit-learn** for Random Forest implementation
- **PyTorch** for ResNet50 model
- **MediaPipe** for facial landmarks
- **Django** & **React** communities

---

<div align="center">

**Made with ❤️ for better hairstyle decisions**

[⬆ Back to Top](#-hairmixer---ai-powered-hairstyle-recommendation-system)

</div>
