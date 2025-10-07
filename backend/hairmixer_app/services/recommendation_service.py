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
from .hairstyle_recommender import HairstyleRecommender

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Recommendation service using hairstyle_family_model (Random Forest).
    
    This service uses the trained Random Forest classifier to generate
    hairstyle recommendations based on user preferences and face analysis.
    """
    def __init__(self):
        self.cache = CacheManager()
        self.recommender = HairstyleRecommender()

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
        face_shape_confidence = float(
            (face_analysis.get('face_shape') or {}).get('confidence', 0.0)
        )
        facial_features = face_analysis.get('facial_features') or {}
        
        # Update preferences with detected face shape if not already set
        if not prefs.faceshape and face_shape:
            prefs.faceshape = face_shape
            prefs.faceshape_confidence = face_shape_confidence
            prefs.save(update_fields=['faceshape', 'faceshape_confidence'])

        # Convert preferences to dict for Random Forest recommender
        preferences_dict = {
            'gender': prefs.gender or '',
            'hair_type': prefs.hair_type or '',
            'hair_length': prefs.hair_length or '',
            'faceshape': prefs.faceshape or face_shape or '',
            'maintenance': prefs.maintenance or '',
            'lifestyle': prefs.lifestyle or '',
            'volume': prefs.volume or '',
            'styling_maintenance': prefs.styling_maintenance or '',
            'styling_preference': prefs.styling_preference or '',
            'hair_condition': prefs.hair_condition or '',
            'hair_thickness': prefs.hair_thickness or '',
            'hair_texture_detail': prefs.hair_texture_detail or '',
            'wants_bangs': prefs.wants_bangs or False,
            'occasions': prefs.occasions or [],
        }
        
        # Use Random Forest model (hairstyle_family_model) for recommendations
        # Get top 10 recommendations only - no fallbacks
        recommendations = self.recommender.get_top_recommendations(
            preferences_dict,
            top_n=10
        )
        
        top_recs = recommendations[:10]
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
