"""
API view for retrieving detailed hairstyle information with AI content
"""
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiResponse
from django.core.cache import cache

from ..models import Hairstyle, UserPreference, UploadedImage
from ..serializers import HairstyleSerializer
from ..services.gemini_service import get_gemini_service

logger = logging.getLogger(__name__)


class HairstyleDetailWithAIView(APIView):
    """
    Get detailed hairstyle information with AI-generated personalized content
    """
    
    @extend_schema(
        summary="Get detailed hairstyle information with AI content",
        description=(
            "Retrieve detailed information about a specific hairstyle "
            "including AI-generated personalized descriptions, product "
            "recommendations, maintenance guide, and styling tips based on "
            "user preferences and face shape analysis."
        ),
        parameters=[
            {
                'name': 'hairstyle_id',
                'in': 'path',
                'required': True,
                'schema': {'type': 'string', 'format': 'uuid'},
                'description': 'UUID of the hairstyle'
            },
            {
                'name': 'preference_id',
                'in': 'query',
                'required': False,
                'schema': {'type': 'string', 'format': 'uuid'},
                'description': 'UUID of user preferences (optional)'
            },
            {
                'name': 'image_id',
                'in': 'query',
                'required': False,
                'schema': {'type': 'string', 'format': 'uuid'},
                'description': 'UUID of uploaded image (optional)'
            }
        ],
        responses={
            200: OpenApiResponse(
                description='Detailed hairstyle information retrieved'
            ),
            404: OpenApiResponse(
                description='Hairstyle not found'
            ),
            500: OpenApiResponse(
                description='Error generating AI content'
            )
        }
    )
    def get(self, request, hairstyle_id):
        """Get detailed hairstyle information with AI-generated content"""
        
        try:
            # Get hairstyle
            try:
                hairstyle = Hairstyle.objects.get(
                    id=hairstyle_id,
                    is_active=True
                )
            except Hairstyle.DoesNotExist:
                return Response(
                    {'error': 'Hairstyle not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get preference and image if provided
            preference_id = request.query_params.get('preference_id')
            image_id = request.query_params.get('image_id')
            
            user_preferences = {}
            face_shape = 'oval'
            face_shape_confidence = 0.0
            
            logger.info(
                f"Hairstyle detail request: hairstyle_id={hairstyle_id}, "
                f"preference_id={preference_id}, image_id={image_id}"
            )
            
            # Try to get user preferences
            if preference_id:
                try:
                    preference = UserPreference.objects.get(id=preference_id)
                    user_preferences = {
                        'hair_type': preference.hair_type,
                        'hair_length': preference.hair_length,
                        'maintenance': preference.maintenance,
                        'lifestyle': preference.lifestyle,
                        'gender': preference.gender,
                        'occasions': preference.occasions or [],
                        'hair_thickness': preference.hair_thickness,
                        'hair_texture_detail': preference.hair_texture_detail,
                        'wants_bangs': preference.wants_bangs,
                        'volume': preference.volume,
                        'styling_preference': preference.styling_preference,
                        'styling_maintenance': preference.styling_maintenance,
                        'hair_color': preference.hair_color,
                        'hair_condition': preference.hair_condition or [],
                    }
                    if preference.faceshape:
                        face_shape = preference.faceshape
                        face_shape_confidence = (
                            preference.faceshape_confidence or 0.0
                        )
                        logger.info(
                            f"Face shape from preference: {face_shape} "
                            f"(confidence: {face_shape_confidence:.2%})"
                        )
                    else:
                        logger.warning(
                            f"Preference {preference_id} found but no "
                            f"face shape stored"
                        )
                except UserPreference.DoesNotExist:
                    logger.warning(f"Preference {preference_id} not found")
            
            # Try to get face analysis from image via RecommendationLog
            if image_id and not preference_id:
                try:
                    from ..models import RecommendationLog
                    
                    # Get the most recent recommendation log for this image
                    # This contains the exact preference and face shape used
                    rec_log = RecommendationLog.objects.filter(
                        uploaded_id=image_id,
                        status='completed'
                    ).order_by('-created_at').first()
                    
                    if rec_log:
                        # Use face shape from recommendation log (most accurate)
                        face_shape = rec_log.face_shape
                        face_shape_confidence = rec_log.face_shape_confidence
                        
                        # Get user preferences from the recommendation
                        if rec_log.preference:
                            pref = rec_log.preference
                            user_preferences = {
                                'hair_type': pref.hair_type,
                                'hair_length': pref.hair_length,
                                'maintenance': pref.maintenance,
                                'lifestyle': pref.lifestyle,
                                'gender': pref.gender,
                                'occasions': pref.occasions or [],
                                'hair_thickness': pref.hair_thickness,
                                'hair_texture_detail': pref.hair_texture_detail,
                                'wants_bangs': pref.wants_bangs,
                                'volume': pref.volume,
                                'styling_preference': pref.styling_preference,
                                'styling_maintenance': pref.styling_maintenance,
                                'hair_color': pref.hair_color,
                                'hair_condition': pref.hair_condition or [],
                            }
                        
                        logger.info(
                            f"Using data from RecommendationLog "
                            f"(ID: {rec_log.id}): face_shape={face_shape} "
                            f"(confidence: {face_shape_confidence:.2%})"
                        )
                    else:
                        logger.warning(
                            f"No recommendation log found for image {image_id}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error retrieving recommendation log: {str(e)}"
                    )
            
            # Create cache key
            cache_key = f"hairstyle_detail_{hairstyle_id}"
            if preference_id:
                cache_key += f"_pref_{preference_id}"
            if image_id:
                cache_key += f"_img_{image_id}"
            
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result:
                logger.info(f"Returning cached details for {hairstyle_id}")
                return Response(cached_result, status=status.HTTP_200_OK)
            
            # Get Gemini service
            gemini_service = get_gemini_service()
            
            # Generate AI content
            ai_details = gemini_service.generate_hairstyle_details(
                hairstyle_name=hairstyle.name,
                hairstyle_description=hairstyle.description or '',
                user_preferences=user_preferences,
                face_shape=face_shape,
                face_shape_confidence=face_shape_confidence,
                hairstyle_tags=hairstyle.tags or [],
                hairstyle_occasions=hairstyle.occasions or []
            )
            
            # Serialize hairstyle
            hairstyle_data = HairstyleSerializer(hairstyle).data
            
            # Combine data (user_preferences removed from response)
            result = {
                'hairstyle': hairstyle_data,
                'face_shape': face_shape,
                'face_shape_confidence': face_shape_confidence,
                'ai_generated': ai_details.get('success', False),
                'personalized_description': ai_details.get(
                    'personalized_description',
                    hairstyle.description or ''
                ),
                'preference_match': ai_details.get('preference_match', []),
                'recommended_products': ai_details.get('products', []),
                'maintenance_guide': ai_details.get('maintenance_guide', []),
                'styling_tips': ai_details.get('styling_tips', [])
            }
            
            # Cache result for 1 hour
            cache.set(cache_key, result, 3600)
            
            return Response(result, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting hairstyle details: {str(e)}")
            return Response(
                {
                    'error': 'Failed to retrieve hairstyle details',
                    'details': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
