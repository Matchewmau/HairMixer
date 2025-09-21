import logging
from typing import Dict, Any
from django.utils import timezone
from ..models import UploadedImage, UserPreference, RecommendationLog, Hairstyle
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

        # Select actual hairstyles from DB for valid UUIDs; fallback to placeholders if empty
        styles = list(Hairstyle.objects.filter(is_active=True).order_by('-trend_score', '-popularity_score', 'name')[:9])
        if styles:
            sample_recommendations = []
            for hs in styles:
                # Prefer stored image; else external image_url
                image_url = None
                try:
                    if hs.image:
                        image_url = hs.image.url  # relative is fine; frontend resolves
                    elif hs.image_url:
                        image_url = hs.image_url
                except Exception:
                    image_url = hs.image_url or None

                sample_recommendations.append({
                    'id': str(hs.id),
                    'name': hs.name,
                    'description': hs.description or '',
                    'image_url': image_url,
                    'category': hs.category.name if hs.category else '',
                    'difficulty': hs.difficulty or 'Medium',
                    'estimated_time': hs.estimated_time or 30,
                    'maintenance': hs.maintenance or 'Medium',
                    'tags': hs.tags or [],
                    'match_score': 0.8,  # placeholder scoring until ML is integrated
                })
        else:
            sample_recommendations = [
                {
                    'id': None, 'name': 'Classic Bob', 'description': 'A timeless bob cut that suits most face shapes',
                    'image_url': None, 'category': 'Classic', 'difficulty': 'Easy', 'estimated_time': 30,
                    'maintenance': 'Medium', 'tags': ['classic', 'versatile'], 'match_score': 0.85
                },
                {
                    'id': None, 'name': 'Beach Waves', 'description': 'Relaxed, casual waves with natural texture',
                    'image_url': None, 'category': 'Casual', 'difficulty': 'Easy', 'estimated_time': 15,
                    'maintenance': 'Low', 'tags': ['casual', 'natural'], 'match_score': 0.78
                },
                {
                    'id': None, 'name': 'Layered Cut', 'description': 'Versatile layered cut for medium-length hair',
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
