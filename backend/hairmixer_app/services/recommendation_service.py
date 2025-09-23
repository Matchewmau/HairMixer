import logging
from typing import Dict, Any
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from ..models import (
    UploadedImage, UserPreference, RecommendationLog, Hairstyle
)
from .cache_manager import CacheManager
from ..ml.face_analyzer import analyze_face_comprehensive
try:
    from ..ml.recommendation_model import MLRecommendationEngine
    _ML_ENGINE_OK = True
except Exception:
    _ML_ENGINE_OK = False

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self):
        self.cache = CacheManager()
        self.ml_engine = MLRecommendationEngine() if _ML_ENGINE_OK else None

    def generate(
        self, uploaded: UploadedImage, prefs: UserPreference, user=None
    ) -> Dict[str, Any]:
        start_time = timezone.now()

        # Try cache first
        cache_key = self.cache.get_recommendation_cache_key(uploaded, prefs)
        cached = self.cache.get_cached_recommendation(cache_key)
        if cached:
            cached['from_cache'] = True
            return cached

        # Analyze face to incorporate face shape into scoring
        face_analysis = analyze_face_comprehensive(
            Path(settings.MEDIA_ROOT) / uploaded.image.name
        )
        face_shape = (
            (face_analysis.get('face_shape') or {}).get('shape', 'oval')
        )
        facial_features = face_analysis.get('facial_features') or {}

        # Select candidate hairstyles then score
        styles = list(
            Hairstyle.objects.filter(is_active=True)
            .order_by('-trend_score', '-popularity_score', 'name')[:50]
        )
        recommendations = []
        if styles:
            for hs in styles:
                try:
                    if self.ml_engine:
                        score_obj = self.ml_engine.predict_user_preference(
                            face_shape, facial_features, hs, prefs
                        )
                        match_score = float(score_obj.get('score') or 0.0)
                    else:
                        # simple heuristic if ML engine missing
                        match_score = 0.5
                    # Prefer stored image; else external image_url
                    try:
                        if hs.image:
                            image_url = hs.image.url
                        else:
                            image_url = hs.image_url or None
                    except Exception:
                        image_url = hs.image_url or None
                    recommendations.append({
                        'id': str(hs.id),
                        'name': hs.name,
                        'description': hs.description or '',
                        'image_url': image_url,
                        'category': hs.category.name if hs.category else '',
                        'difficulty': hs.difficulty or 'Medium',
                        'estimated_time': hs.estimated_time or 30,
                        'maintenance': hs.maintenance or 'Medium',
                        'tags': hs.tags or [],
                        'match_score': round(match_score, 3),
                    })
                except Exception:
                    logger.warning(
                        "Scoring failed for style %s", hs.id, exc_info=True
                    )
        else:
            recommendations = []

        # Sort and pick top N
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        top_recs = recommendations[:9]
        processing_time = (timezone.now() - start_time).total_seconds()

        selected_style_obj = None
        if top_recs:
            try:
                selected_style_obj = Hairstyle.objects.get(
                    id=top_recs[0]['id']
                )
            except Exception:
                selected_style_obj = None

        log = RecommendationLog.objects.create(
            user=user,
            uploaded=uploaded,
            preference=prefs,
            face_shape=face_shape,
            face_shape_confidence=float(
                face_analysis.get('confidence') or 0.0
            ),
            detected_features=facial_features,
            selected_hairstyle=selected_style_obj,
            candidates=[r['id'] for r in top_recs],
            recommendation_scores={
                r['id']: r['match_score'] for r in top_recs
            },
            status='completed',
            processing_time=processing_time,
            model_version='v1.0'
        )

        response_data = {
            "recommendation_id": str(log.id),
            "face_shape": face_shape,
            "face_shape_confidence": float(
                face_analysis.get('confidence') or 0.0
            ),
            "detected_features": facial_features,
            "recommended_styles": top_recs,
            "candidates": top_recs,
            "processing_time": f"{processing_time:.2f}s",
            "total_styles_analyzed": len(recommendations)
        }

        # Cache results
        try:
            self.cache.cache_recommendation(cache_key, response_data, prefs)
        except Exception:
            logger.warning("Failed to cache recommendation", exc_info=True)

        return response_data
