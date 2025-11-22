# 🧪 Test Files Quick Reference

**Last Updated:** October 8, 2025  
**Status:** Clean and Organized

---

## 📋 Available Test Files (6 Root + 8 Django)

### Root Test Files

| File | Size | Purpose | Run Command |
|------|------|---------|-------------|
| **check_model_features.py** ⚡ | 0.7 KB | Quick model check | `python check_model_features.py` |
| **check_stats.py** ⚡ | 1.2 KB | Database statistics | `python check_stats.py` |
| **test_hairstyle_recommender.py** ⭐ | 7.4 KB | Comprehensive model test | `python test_hairstyle_recommender.py` |
| **test_family_system.py** | 2.4 KB | Family system test | `python test_family_system.py` |
| **test_face_detection_direct.py** | 4.4 KB | Face detection test | `python test_face_detection_direct.py` |
| **demo_ml_recommendations.py** 🎬 | 9.9 KB | Complete API demo | `python demo_ml_recommendations.py` |

### Django Test Suite

| File | Purpose |
|------|---------|
| `test_smoke.py` | Smoke tests |
| `test_overlay_docs.py` | Overlay feature tests |
| `test_more_user_flows.py` | User flow tests |
| `test_more_endpoints.py` | API endpoint tests |
| `test_flow.py` | Main flow tests |
| `test_data_driven.py` | Data-driven tests |
| `test_auth_endpoints.py` | Authentication tests |
| `test_api_docs.py` | API documentation tests |

**Run Django tests:** `python manage.py test hairmixer_app.tests`

---

## ⚡ Quick Commands

### Daily Development
```bash
cd backend
python check_model_features.py  # Check model status
python check_stats.py            # Check database stats
```

### Before Commits
```bash
cd backend
python test_hairstyle_recommender.py  # Full model test
python manage.py test                  # Django tests
```

### Before Releases
```bash
cd backend
python test_hairstyle_recommender.py
python test_face_detection_direct.py
python test_family_system.py
python manage.py test hairmixer_app.tests
```

### For Demos
```bash
cd backend
python demo_ml_recommendations.py  # Full API workflow demo
```

---

## 🗑️ Removed Files (14)

- compare_models.py
- check_model_database_match.py
- inspect_hairmixer_model.py
- analyze_dataset.py
- analyze_training_data.py
- check_csv_features.py
- fix_gender_distribution.py
- show_gender_breakdown.py
- check_database_gender.py
- check_categories.py
- verify_cleanup.py
- test_random_forest_model.py
- test_api_endpoint.py
- run_api_test.py

**Reason:** One-time analysis, redundant, or completed tasks

---

## 📊 Test Coverage

✅ ML Model (hairstyle_family_model.pkl)  
✅ Face Detection (ResNet50)  
✅ API Endpoints (all core endpoints)  
✅ Database (hairstyles, categories, gender)  
✅ User Flows (complete workflows)  
✅ Error Handling  
✅ Gender Filtering  

---

## 📚 Documentation

- **Full Details:** `TEST_FILES_CLEANUP_SUMMARY.md`
- **System Docs:** `SYSTEM_DOCUMENTATION.md`
- **API Docs:** `backend/API_DOCS.md`
- **ML Docs:** `docs/ML_RECOMMENDATIONS.md`

---

**Legend:**  
⭐ = Comprehensive Test  
⚡ = Quick Diagnostic  
🎬 = Demo Script
