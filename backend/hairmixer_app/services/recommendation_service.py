import logging
from typing import Dict, Any
from django.utils import timezone
from ..models import UploadedImage, UserPreference, RecommendationLog
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self):
        self.cache = CacheManager()

    def generate(self, uploaded: UploadedImage, prefs: UserPreference, user=None) -> Dict[str, Any]:
        start_time = timezone.now()

        # Try cache first
        cache_key = self.cache.get_recommendation_cache_key(uploaded, prefs)
        cached = self.cache.get_cached_recommendation(cache_key)
        if cached:
            cached['from_cache'] = True
            return cached

        # Placeholder logic while ML is being integrated
        sample_recommendations = [
            {
                'id': '1', 'name': 'Classic Bob', 'description': 'A timeless bob cut that suits most face shapes',
                'image_url': None, 'category': 'Classic', 'difficulty': 'Easy', 'estimated_time': 30,
                'maintenance': 'Medium', 'tags': ['classic', 'versatile'], 'match_score': 0.85
            },
            {
                'id': '2', 'name': 'Beach Waves', 'description': 'Relaxed, casual waves with natural texture',
                'image_url': None, 'category': 'Casual', 'difficulty': 'Easy', 'estimated_time': 15,
                'maintenance': 'Low', 'tags': ['casual', 'natural'], 'match_score': 0.78
            },
            {
                'id': '3', 'name': 'Layered Cut', 'description': 'Versatile layered cut for medium-length hair',
                'image_url': None, 'category': 'Versatile', 'difficulty': 'Medium', 'estimated_time': 35,
                'maintenance': 'Medium', 'tags': ['layered', 'versatile'], 'match_score': 0.72
            }
        ]

        processing_time = (timezone.now() - start_time).total_seconds()

        log = RecommendationLog.objects.create(
            user=user,
            uploaded=uploaded,
            preference=prefs,
            face_shape='oval',
            face_shape_confidence=0.8,
            detected_features={},
            candidates=[],
            recommendation_scores={},
            status='completed',
            processing_time=processing_time,
            model_version='v1.0'
        )

        response_data = {
            "recommendation_id": str(log.id),
            "face_shape": "oval",
            "face_shape_confidence": 0.8,
            "detected_features": {},
            "recommended_styles": sample_recommendations,
            "candidates": sample_recommendations,
            "processing_time": f"{processing_time:.2f}s",
            "total_styles_analyzed": len(sample_recommendations)
        }

        # Cache results
        try:
            self.cache.cache_recommendation(cache_key, response_data, prefs)
        except Exception:
            logger.warning("Failed to cache recommendation", exc_info=True)

        return response_data
