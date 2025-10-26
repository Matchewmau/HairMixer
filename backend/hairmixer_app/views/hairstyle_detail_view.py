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
                        'occasions': preference.occasions,
                        'hair_thickness': preference.hair_thickness,
                        'hair_texture_detail': preference.hair_texture_detail,
                        'wants_bangs': preference.wants_bangs,
                    }
                    if preference.faceshape:
                        face_shape = preference.faceshape
                        face_shape_confidence = (
                            preference.faceshape_confidence or 0.0
                        )
                except UserPreference.DoesNotExist:
                    logger.warning(f"Preference {preference_id} not found")
            
            # Try to get face analysis from image
            if image_id and not preference_id:
                try:
                    image = UploadedImage.objects.get(id=image_id)
                    # Try to get face shape from image analysis
                    if hasattr(image, 'faceanalysis'):
                        analysis = image.faceanalysis
                        if analysis.face_shape:
                            face_shape = analysis.face_shape
                            face_shape_confidence = (
                                analysis.face_shape_confidence or 0.0
                            )
                except UploadedImage.DoesNotExist:
                    logger.warning(f"Image {image_id} not found")
            
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
            
            # Combine data
            result = {
                'hairstyle': hairstyle_data,
                'face_shape': face_shape,
                'face_shape_confidence': face_shape_confidence,
                'user_preferences': user_preferences,
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
