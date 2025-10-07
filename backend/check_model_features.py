"""Check model feature details"""
import os
import sys
from pathlib import Path

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
import django
django.setup()

from hairmixer_app.services.hairstyle_recommender import HairstyleRecommender

r = HairstyleRecommender()
print(f"Model loaded: {r.model_loaded}")
print(f"Model type: {type(r.model).__name__}")
print(f"Model expects {r.model.n_features_in_} features")

if hasattr(r.model, 'feature_names_in_'):
    print(f"\nFeature names:")
    for i, name in enumerate(r.model.feature_names_in_):
        print(f"  {i+1}. {name}")
else:
    print("\nNo feature names stored in model")
