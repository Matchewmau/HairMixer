from django.shortcuts import get_object_or_404
from django.db.models import Q, Avg, Count
from django.utils import timezone
from django.db import connection
from django.core.paginator import Paginator
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.parsers import JSONParser
from rest_framework.throttling import UserRateThrottle
from drf_spectacular.utils import (
    extend_schema,
    OpenApiResponse,
    OpenApiExample,
    OpenApiParameter,
)
import logging
import traceback
from datetime import timedelta

from ..models import (
    UploadedImage, UserPreference, Hairstyle, HairstyleCategory,
    RecommendationLog, Feedback
)
from ..serializers import (
    RecommendRequestSerializer, HairstyleSerializer, FeedbackSerializer,
    RecommendationLogSerializer, HairstyleCategorySerializer
)
from ..services.recommendation_service import RecommendationService
from ..services.analytics_utils import track_event_safe
from ..services.analytics import AnalyticsService

# Initialize services
try:
    analytics_service = AnalyticsService()
except ImportError:
    analytics_service = None

recommendation_service = RecommendationService()

logger = logging.getLogger(__name__)

class RecommendationThrottle(UserRateThrottle):
    rate = '20/hour'

class RecommendView(APIView):
    """
    Hairstyle recommendations using Random Forest model.
    """
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [RecommendationThrottle]

    @extend_schema(
        request=RecommendRequestSerializer,
        responses={
            200: OpenApiResponse(description='Recommendations generated'),
            400: OpenApiResponse(description='Bad request'),
            404: OpenApiResponse(description='Image or preferences not found'),
            500: OpenApiResponse(description='Server error generating recommendations'),
        },
        examples=[
            OpenApiExample(
                'Generate recommendations',
                value={
                    'image_id': '11111111-1111-1111-1111-111111111111',
                    'preference_id': '33333333-3333-3333-3333-333333333333',
                },
                request_only=True,
            )
        ],
    )
    def post(self, request):
        try:
            req_ser = RecommendRequestSerializer(data=request.data)
            req_ser.is_valid(raise_exception=True)
            image_id = req_ser.validated_data["image_id"]
            pref_id = req_ser.validated_data["preference_id"]

            logger.info(
                "Recommendation request - image_id: %s, pref_id: %s",
                image_id,
                pref_id,
            )

            if not image_id or not pref_id:
                return Response(
                    {"error": "Both image_id and preference_id are required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            
            try:
                uploaded = UploadedImage.objects.get(id=image_id)
                prefs = UserPreference.objects.get(id=pref_id)
            except UploadedImage.DoesNotExist:
                return Response(
                    {"error": "Uploaded image not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )
            except UserPreference.DoesNotExist:
                return Response(
                    {"error": "User preferences not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )
            
            rec_user = (
                request.user
                if getattr(request.user, 'is_authenticated', False)
                else None
            )
            response_data = recommendation_service.generate(
                uploaded, prefs, user=rec_user
            )
            return Response(response_data)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            traceback.print_exc()
            return Response(
                {
                    "error": "Failed to generate recommendations",
                    "details": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class MLRecommendView(APIView):
    """
    ML-based hairstyle recommendations using trained Random Forest model.
    """
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [RecommendationThrottle]
    
    @extend_schema(
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'preference_id': {
                        'type': 'string',
                        'format': 'uuid',
                        'description': 'User preference UUID'
                    },
                },
                'required': ['preference_id']
            }
        },
        responses={
            200: OpenApiResponse(description='ML-based recommendations generated'),
            400: OpenApiResponse(description='Bad request'),
            404: OpenApiResponse(description='Preferences not found'),
            500: OpenApiResponse(description='Server error generating recommendations'),
        },
    )
    def post(self, request):
        try:
            from ..ml.hairstyle_recommender import get_hairstyle_recommender
            
            pref_id = request.data.get('preference_id')
            
            if not pref_id:
                return Response(
                    {"error": "preference_id is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            
            try:
                prefs = UserPreference.objects.get(id=pref_id)
            except UserPreference.DoesNotExist:
                return Response(
                    {"error": "User preferences not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )
            
            pref_dict = {
                'gender': prefs.gender or '',
                'hair_type': prefs.hair_type or '',
                'hair_length': prefs.hair_length or '',
                'hair_color': prefs.hair_color or '',
                'lifestyle': prefs.lifestyle or '',
                'maintenance': prefs.maintenance or '',
                'volume': prefs.volume or '',
                'styling_maintenance': prefs.styling_maintenance or '',
                'hair_texture_detail': prefs.hair_texture_detail or '',
                'styling_preference': prefs.styling_preference or '',
                'hair_condition': prefs.hair_condition or '',
                'hair_thickness': prefs.hair_thickness or '',
                'wants_bangs': prefs.wants_bangs if prefs.wants_bangs is not None else False,
                'occasions': prefs.occasions or [],
            }
            
            face_shape = prefs.faceshape or 'oval'
            
            ml_recommender = get_hairstyle_recommender()
            if not ml_recommender.is_available():
                return Response(
                    {"error": "ML model not available"},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            
            ml_predictions = ml_recommender.predict_top_k(
                pref_dict, face_shape, k=10
            )
            
            recommendations = []
            for pred in ml_predictions:
                hairstyle_name = pred['hairstyle_name']
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
                            'confidence': pred['confidence'] * 100,
                            'rank': pred['rank']
                        })
                except Exception as e:
                    logger.warning(
                        f"Could not find hairstyle {hairstyle_name}: {e}"
                    )
            
            logger.info(
                f"ML recommendations generated: {len(recommendations)} styles"
            )
            
            response_data = {
                "recommendation_count": len(recommendations),
                "recommendations": recommendations,
                "model_used": "RF_No_Family_v1.0",
                "faceshape": face_shape,
                "faceshape_confidence": prefs.faceshape_confidence or 0.0,
            }
            
            track_event_safe(
                analytics_service,
                user=(
                    request.user
                    if getattr(request.user, 'is_authenticated', False)
                    else None
                ),
                event_type='ml_recommendation_generated',
                event_data={
                    'preference_id': str(pref_id),
                    'recommendation_count': len(recommendations),
                    'faceshape': prefs.faceshape or 'not_detected',
                },
                request=request,
            )
            
            return Response(response_data)
            
        except Exception as e:
            logger.error(
                f"Error generating ML recommendations: {str(e)}",
                exc_info=True
            )
            return Response(
                {
                    "error": "Failed to generate ML recommendations",
                    "details": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class FeaturedHairstylesView(APIView):
    """Get featured hairstyles"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            limit = min(int(request.query_params.get('limit', 20)), 50)
            
            featured_styles = Hairstyle.objects.filter(
                is_active=True,
                is_featured=True
            ).select_related('category').order_by('-trend_score')[:limit]
            
            serializer = HairstyleSerializer(
                featured_styles,
                many=True,
                context={'request': request},
            )
            
            return Response({
                'featured_hairstyles': serializer.data,
                'count': len(serializer.data)
            })
            
        except Exception as e:
            logger.error(f"Error fetching featured hairstyles: {str(e)}")
            return Response(
                {"error": "Failed to fetch featured hairstyles"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class TrendingHairstylesView(APIView):
    """Get trending hairstyles based on recent activity"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            limit = min(int(request.query_params.get('limit', 20)), 50)
            
            trending_styles = Hairstyle.objects.filter(
                is_active=True
            ).annotate(
                recent_recommendations=Count(
                    'recommendationlog',
                    filter=Q(
                        recommendationlog__created_at__gte=(
                            timezone.now() - timedelta(days=7)
                        )
                    ),
                ),
                avg_rating=Avg('feedback__rating')
            ).order_by(
                '-recent_recommendations',
                '-avg_rating',
                '-popularity_score',
            )[:limit]
            
            serializer = HairstyleSerializer(
                trending_styles, many=True, context={'request': request}
            )

            track_event_safe(
                analytics_service,
                user=(request.user if request.user.is_authenticated else None),
                event_type='trending_viewed',
                event_data={'count': len(serializer.data)},
                request=request,
            )
            
            return Response({
                'trending_hairstyles': serializer.data,
                'count': len(serializer.data)
            })
            
        except Exception as e:
            logger.error(f"Error fetching trending hairstyles: {str(e)}")
            return Response(
                {"error": "Failed to fetch trending hairstyles"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class HairstyleDetailView(APIView):
    """Get detailed information about a specific hairstyle"""
    permission_classes = [AllowAny]
    
    def get(self, request, style_id):
        try:
            style = get_object_or_404(
                Hairstyle.objects.select_related('category'),
                id=style_id,
                is_active=True,
            )
            
            recent_feedback = Feedback.objects.filter(
                hairstyle=style,
                is_public=True
            ).order_by('-created_at')[:5]
            
            feedback_stats = Feedback.objects.filter(
                hairstyle=style
            ).aggregate(
                avg_rating=Avg('rating'),
                total_feedback=Count('id'),
                positive_feedback=Count('id', filter=Q(liked=True))
            )

            serializer = HairstyleSerializer(
                style, context={'request': request}
            )
            feedback_serializer = FeedbackSerializer(
                recent_feedback, many=True
            )
            
            track_event_safe(
                analytics_service,
                user=(request.user if request.user.is_authenticated else None),
                event_type='hairstyle_viewed',
                event_data={
                    'style_id': str(style_id),
                    'style_name': style.name,
                },
                request=request,
            )
            
            return Response({
                'hairstyle': serializer.data,
                'feedback_stats': feedback_stats,
                'recent_feedback': feedback_serializer.data,
                'related_styles': []
            })
            
        except Exception as e:
            logger.error(f"Error fetching hairstyle detail: {str(e)}")
            return Response(
                {"error": "Failed to fetch hairstyle details"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class HairstyleDetailWithAIView(APIView):
    """
    Get detailed information about a specific hairstyle with AI compatibility analysis.
    Checks if the hairstyle suits the user's face shape and preferences.
    """
    permission_classes = [AllowAny]

    def get(self, request, hairstyle_id):
        try:
            from ..services.gemini_service import get_gemini_service
            
            style = get_object_or_404(
                Hairstyle.objects.select_related('category'),
                id=hairstyle_id,
                is_active=True,
            )
            
            # Get image_id from query params to find uploaded image and face shape
            image_id = request.query_params.get('image_id')
            face_shape = None
            
            if image_id:
                try:
                    uploaded = UploadedImage.objects.get(id=image_id)
                    # Try to find recent recommendation log for this image to get face shape
                    rec_log = RecommendationLog.objects.filter(
                        uploaded=uploaded
                    ).order_by('-created_at').first()
                    if rec_log:
                        face_shape = rec_log.face_shape
                except Exception:
                    pass
            
            # Get user preferences
            user_prefs_dict = {}
            face_shape_confidence = 0.0
            
            if request.user.is_authenticated:
                pref = UserPreference.objects.filter(
                    user=request.user
                ).order_by('-updated_at').first()
                
                if pref:
                    user_prefs_dict = {
                        'hair_type': pref.hair_type,
                        'hair_length': pref.hair_length,
                        'maintenance': pref.maintenance,
                        'lifestyle': pref.lifestyle,
                        'gender': pref.gender,
                        'occasions': pref.occasions,
                        'hair_thickness': pref.hair_thickness,
                        'hair_texture_detail': pref.hair_texture_detail,
                        'wants_bangs': pref.wants_bangs,
                        'volume': pref.volume,
                        'styling_preference': pref.styling_preference,
                        'hair_color': pref.hair_color,
                        'hair_condition': pref.hair_condition,
                    }
                    if not face_shape:
                        face_shape = pref.faceshape
                        face_shape_confidence = pref.faceshape_confidence

            # Generate AI details
            gemini_service = get_gemini_service()
            ai_details = gemini_service.generate_hairstyle_details(
                hairstyle_name=style.name,
                hairstyle_description=style.description,
                user_preferences=user_prefs_dict,
                face_shape=face_shape or 'oval',
                face_shape_confidence=face_shape_confidence,
                hairstyle_tags=style.tags,
                hairstyle_occasions=style.occasions
            )

            # Compatibility analysis (keep existing logic as fallback/supplement)
            compatibility = {
                'is_suitable': True,
                'reason': 'Suitable for all face shapes',
                'match_score': 85
            }
            
            if face_shape:
                face_shape_lower = face_shape.lower()
                suitable_shapes = [s.lower() for s in style.face_shapes]
                
                if suitable_shapes:
                    if face_shape_lower in suitable_shapes:
                        compatibility = {
                            'is_suitable': True,
                            'reason': f"Great match for your {face_shape_lower} face shape!",
                            'match_score': 95
                        }
                    else:
                        compatibility = {
                            'is_suitable': False,
                            'reason': f"This style is typically best for {', '.join(suitable_shapes)} face shapes, but you can still rock it!",
                            'match_score': 60
                        }
            
            serializer = HairstyleSerializer(
                style, context={'request': request}
            )
            
            # Merge AI details into response
            response_data = {
                'hairstyle': serializer.data,
                'compatibility': compatibility,
                'user_face_shape': face_shape,
                'personalized_description': ai_details.get('personalized_description'),
                'preference_match': ai_details.get('preference_match'),
                'recommended_products': ai_details.get('products'),
                'maintenance_guide': ai_details.get('maintenance_guide'),
                'styling_tips': ai_details.get('styling_tips'),
                'user_preferences': user_prefs_dict
            }
            
            return Response(response_data)

        except Exception as e:
            logger.error(f"Error fetching AI hairstyle detail: {str(e)}")
            return Response(
                {"error": "Failed to fetch hairstyle details"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ListHairstylesView(APIView):
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            category = request.query_params.get('category')
            face_shape = request.query_params.get('face_shape')
            occasion = request.query_params.get('occasion')
            maintenance = request.query_params.get('maintenance')
            gender = request.query_params.get('gender', '').lower()
            featured_only = (
                request.query_params.get('featured', '').lower() == 'true'
            )
            limit = min(int(request.query_params.get('limit', 50)), 100)
            
            queryset = Hairstyle.objects.filter(is_active=True).select_related(
                'category'
            )
            
            if category:
                queryset = queryset.filter(category__name__icontains=category)
            
            if face_shape:
                queryset = queryset.filter(face_shapes__contains=[face_shape])
            
            if occasion:
                queryset = queryset.filter(occasions__contains=[occasion])
                
            if maintenance:
                queryset = queryset.filter(maintenance=maintenance)
            
            if gender in ['male', 'female']:
                gender_filter = Q(suitable_gender='unisex')
                gender_filter |= Q(suitable_gender=gender)
                queryset = queryset.filter(gender_filter)
                
            if featured_only:
                queryset = queryset.filter(is_featured=True)
            
            queryset = queryset.order_by(
                '-trend_score', '-popularity_score', 'name'
            )[:limit]
            
            serializer = HairstyleSerializer(
                queryset, many=True, context={'request': request}
            )
            
            track_event_safe(
                analytics_service,
                user=(request.user if request.user.is_authenticated else None),
                event_type='hairstyles_browsed',
                event_data={
                    'filters': {
                        'category': category,
                        'face_shape': face_shape,
                        'occasion': occasion,
                        'maintenance': maintenance,
                        'featured_only': featured_only,
                    },
                    'results_count': len(serializer.data),
                },
                request=request,
            )
            
            return Response({
                'hairstyles': serializer.data,
                'total_count': len(serializer.data),
                'filters_applied': {
                    'category': category,
                    'face_shape': face_shape,
                    'occasion': occasion,
                    'maintenance': maintenance,
                    'featured_only': featured_only
                }
            })
            
        except Exception as e:
            logger.error(f"Error fetching hairstyles: {str(e)}")
            return Response(
                {"error": "Failed to fetch hairstyles", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class HairstyleCategoriesView(APIView):
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            categories = HairstyleCategory.objects.filter(
                is_active=True
            ).order_by('sort_order', 'name')
            serializer = HairstyleCategorySerializer(categories, many=True)
            return Response({'categories': serializer.data})
        except Exception as e:
            logger.error(f"Error fetching categories: {str(e)}")
            return Response(
                {"error": "Failed to fetch categories"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SearchView(APIView):
    """Search hairstyles with advanced filtering"""
    permission_classes = [AllowAny]

    @extend_schema(
        parameters=[
            OpenApiParameter('q', str, OpenApiParameter.QUERY, description='Search text'),
            OpenApiParameter('face_shape', str, OpenApiParameter.QUERY),
            OpenApiParameter('occasion', str, OpenApiParameter.QUERY),
            OpenApiParameter('hair_type', str, OpenApiParameter.QUERY),
            OpenApiParameter('maintenance', str, OpenApiParameter.QUERY),
            OpenApiParameter('page', int, OpenApiParameter.QUERY, default=1),
            OpenApiParameter('per_page', int, OpenApiParameter.QUERY, default=20),
        ],
        responses={200: OpenApiResponse(description='Search results returned')},
    )
    def get(self, request):
        try:
            query = request.query_params.get('q', '').strip()
            face_shape = request.query_params.get('face_shape', '')
            occasion = request.query_params.get('occasion', '')
            hair_type = request.query_params.get('hair_type', '')
            maintenance = request.query_params.get('maintenance', '')
            page = int(request.query_params.get('page', 1))
            per_page = min(int(request.query_params.get('per_page', 20)), 50)
            
            queryset = Hairstyle.objects.filter(is_active=True)
            engine = connection.settings_dict.get('ENGINE', '')
            supports_json = 'postgresql' in engine
            
            if query:
                if supports_json:
                    queryset = queryset.filter(
                        Q(name__icontains=query)
                        | Q(description__icontains=query)
                        | Q(tags__contains=[query])
                    )
                else:
                    queryset = queryset.filter(
                        Q(name__icontains=query)
                        | Q(description__icontains=query)
                    )
            
            if face_shape:
                if supports_json:
                    queryset = queryset.filter(face_shapes__contains=[face_shape])
            
            if occasion:
                if supports_json:
                    queryset = queryset.filter(occasions__contains=[occasion])
            
            if hair_type:
                if supports_json:
                    queryset = queryset.filter(hair_types__contains=[hair_type])
            
            if maintenance:
                queryset = queryset.filter(maintenance=maintenance)
            
            queryset = queryset.order_by(
                '-trend_score', '-popularity_score', 'name'
            )
            
            paginator = Paginator(queryset, per_page)
            page_obj = paginator.get_page(page)
            
            serializer = HairstyleSerializer(
                page_obj.object_list,
                many=True,
                context={'request': request},
            )
            
            track_event_safe(
                analytics_service,
                user=(
                    request.user if request.user.is_authenticated else None
                ),
                event_type='search_performed',
                event_data={
                    'query': query,
                    'filters': {
                        'face_shape': face_shape,
                        'occasion': occasion,
                        'hair_type': hair_type,
                        'maintenance': maintenance,
                    },
                    'results_count': paginator.count,
                },
                request=request,
            )
            
            return Response({
                'results': serializer.data,
                'search_query': query,
                'filters_applied': {
                    'face_shape': face_shape,
                    'occasion': occasion,
                    'hair_type': hair_type,
                    'maintenance': maintenance
                },
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total_pages': paginator.num_pages,
                    'total_count': paginator.count,
                    'has_next': page_obj.has_next(),
                    'has_previous': page_obj.has_previous()
                }
            })
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            return Response(
                {"error": "Search failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
