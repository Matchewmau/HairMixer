import logging
from typing import Dict, Any
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from django.db.models import Q
from ..models import (
    UploadedImage, UserPreference, RecommendationLog, Hairstyle
)
from .cache_manager import CacheManager
from ..ml.face_analyzer import analyze_face_comprehensive
from ..ml.hairstyle_model_recommender import get_hairstyle_model_recommender

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Recommendation service using Random Forest Model with One-Hot Encoding.
    
    This service uses the trained Random Forest classifier (hairstyle_model.joblib)
    to generate hairstyle recommendations based on user preferences and face analysis.
    """
    def __init__(self):
        self.cache = CacheManager()
        self.ml_recommender = get_hairstyle_model_recommender()

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

        # Convert preferences to dict for Random Forest No-Family recommender
        user_pref_dict = {
            'gender': prefs.gender or '',
            'hair_type': prefs.hair_type or '',
            'hair_length': prefs.hair_length or '',
            'hair_color': prefs.hair_color or '',
            'maintenance': prefs.maintenance or '',
            'lifestyle': prefs.lifestyle or '',
            'volume': prefs.volume or '',
            'styling_maintenance': prefs.styling_maintenance or '',
            'styling_preference': prefs.styling_preference or '',
            'hair_condition': prefs.hair_condition or '',
            'hair_thickness': prefs.hair_thickness or '',
            'hair_texture_detail': prefs.hair_texture_detail or '',
            'wants_bangs': prefs.wants_bangs if prefs.wants_bangs is not None else False,
            'occasions': prefs.occasions or [],
        }
        
        # Use Random Forest No-Family model for ML predictions
        recommendations = []
        if self.ml_recommender.is_available():
            ml_predictions = self.ml_recommender.predict_top_k(
                user_pref_dict,
                face_shape,
                k=10
            )
            
            # Look up hairstyles in database
            for pred in ml_predictions:
                hairstyle_name = pred['hairstyle_name']
                
                # Try to find matching hairstyle in DB
                # Handle both underscore and space variations
                try:
                    hairstyle = Hairstyle.objects.filter(
                        Q(name__iexact=hairstyle_name) |
                        Q(name__iexact=hairstyle_name.replace('_', ' ')) |
                        Q(name__iexact=hairstyle_name.replace(' ', '_')),
                        is_active=True
                    ).first()
                    
                    if hairstyle:
                        recommendations.append({
                            'id': str(hairstyle.id),
                            'name': hairstyle.name,
                            'description': hairstyle.description or '',
                            'image_url': (
                                hairstyle.image.url if hairstyle.image 
                                else hairstyle.image_url
                            ),
                            'category': (
                                hairstyle.category.name 
                                if hairstyle.category else ''
                            ),
                            'difficulty': hairstyle.difficulty or 'Medium',
                            'estimated_time': hairstyle.estimated_time or 30,
                            'maintenance': hairstyle.maintenance or 'Medium',
                            'tags': hairstyle.tags or [],
                            'match_score': pred['confidence'],
                            'rank': pred['rank']
                        })
                    else:
                        logger.debug(
                            f"Hairstyle '{hairstyle_name}' not found in database"
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not find hairstyle {hairstyle_name}: {e}"
                    )
        else:
            logger.warning("ML recommender not available")
        
        # If ML model fails or returns no results, log warning
        if not recommendations:
            logger.warning(
                "No ML recommendations generated. "
                "Check model loading and database hairstyle entries."
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
            face_shape_confidence=face_shape_confidence,
            detected_features=facial_features,
            selected_hairstyle=selected_style_obj,
            candidates=[r['id'] for r in top_recs],
            recommendation_scores={
                r['id']: r['match_score'] for r in top_recs
            },
            status='completed',
            processing_time=processing_time,
            model_version='RF_No_Family_v1.0'
        )

        response_data = {
            "recommendation_id": str(log.id),
            "face_shape": face_shape,
            "face_shape_confidence": face_shape_confidence,
            "detected_features": facial_features,
            "recommended_styles": top_recs,
            "recommendations": top_recs,  # Frontend expects this field
            "candidates": top_recs,
            "processing_time": f"{processing_time:.2f}s",
            "total_styles_analyzed": len(recommendations),
            "model_version": "RF_No_Family_v1.0"
        }

        # Cache results
        try:
            self.cache.cache_recommendation(cache_key, response_data, prefs)
        except Exception:
            logger.warning("Failed to cache recommendation", exc_info=True)

        return response_data
