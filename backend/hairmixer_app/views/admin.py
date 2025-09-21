from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes
from drf_spectacular.utils import extend_schema, OpenApiResponse

from ..models import Hairstyle, HairstyleCategory
from .base import (
    ADMIN_PERMISSION_CLASS,
    analytics_service,
    cache_manager,
    logger,
    ML_MODEL_AVAILABLE,
    ML_PREPROCESS_AVAILABLE,
    RECOMMENDATION_ENGINE_AVAILABLE,
    OVERLAY_PROCESSOR_AVAILABLE,
    ANALYTICS_AVAILABLE,
    CACHE_MANAGER_AVAILABLE,
)


class CacheStatsView(APIView):
    permission_classes = [ADMIN_PERMISSION_CLASS]
    serializer_class = None

    @extend_schema(
        responses={
            200: OpenApiResponse(description='Cache statistics returned')
        }
    )
    def get(self, request):
        try:
            if cache_manager:
                stats = cache_manager.get_cache_stats()
                return Response({'cache_stats': stats})
            return Response(
                {
                    'cache_stats': {
                        'message': 'Cache manager not available'
                    }
                }
            )
        except Exception as e:
            logger.error(f"Error fetching cache stats: {str(e)}")
            return Response(
                {"error": "Failed to fetch cache stats"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class CacheCleanupView(APIView):
    permission_classes = [ADMIN_PERMISSION_CLASS]
    serializer_class = None

    @extend_schema(
        responses={
            200: OpenApiResponse(description='Cache cleanup executed')
        }
    )
    def post(self, request):
        try:
            if cache_manager:
                cleaned_count = cache_manager.cleanup_expired_cache()
                return Response(
                    {
                        'message': (
                            f'Cleaned up {cleaned_count} expired cache entries'
                        ),
                        'cleaned_count': cleaned_count,
                    }
                )
            return Response({'message': 'Cache manager not available'})
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")
            return Response(
                {"error": "Cache cleanup failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SystemAnalyticsView(APIView):
    permission_classes = [ADMIN_PERMISSION_CLASS]
    serializer_class = None

    @extend_schema(
        responses={
            200: OpenApiResponse(description='System analytics returned')
        }
    )
    def get(self, request):
        try:
            days = int(request.query_params.get('days', 7))
            if analytics_service:
                analytics_data = analytics_service.get_system_analytics(days)
                return Response({'analytics': analytics_data})
            return Response(
                {'analytics': {'message': 'Analytics service not available'}}
            )
        except Exception as e:
            logger.error(f"Error fetching system analytics: {str(e)}")
            return Response(
                {"error": "Failed to fetch analytics"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@extend_schema(responses={200: OpenApiResponse(description='Health status')})
@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    try:
        styles = Hairstyle.objects.filter(is_active=True).count()
        cats = HairstyleCategory.objects.filter(is_active=True).count()
    except Exception:
        styles = None
        cats = None
    return Response(
        {
            'status': 'ok',
            'message': 'HairMixer backend is running',
            'ml_available': ML_MODEL_AVAILABLE,
            'preprocess_available': ML_PREPROCESS_AVAILABLE,
            'recommendation_available': RECOMMENDATION_ENGINE_AVAILABLE,
            'overlay_available': OVERLAY_PROCESSOR_AVAILABLE,
            'analytics_available': ANALYTICS_AVAILABLE,
            'cache_available': CACHE_MANAGER_AVAILABLE,
            'metrics': {'active_styles': styles, 'active_categories': cats},
        }
    )


@extend_schema(responses={200: OpenApiResponse(description='API root index')})
@api_view(['GET'])
@permission_classes([AllowAny])
def api_root(request):
    base_url = request.build_absolute_uri('/api/')

    endpoints = {
        "message": "Welcome to HairMixer API",
        "version": "1.0",
        "status": "online",
        "endpoints": {
            "Authentication": {
                "signup": f"{base_url}auth/signup/",
                "login": f"{base_url}auth/login/",
                "logout": f"{base_url}auth/logout/",
                "refresh_token": f"{base_url}auth/refresh/",
                "user_profile": f"{base_url}auth/profile/",
            },
            "Core Features": {
                "upload_image": f"{base_url}upload/",
                "set_preferences": f"{base_url}preferences/",
                "get_recommendations": f"{base_url}recommend/",
                "create_overlay": f"{base_url}overlay/",
                "submit_feedback": f"{base_url}feedback/",
            },
            "Hairstyles": {
                "list_all": f"{base_url}hairstyles/",
                "featured": f"{base_url}hairstyles/featured/",
                "trending": f"{base_url}hairstyles/trending/",
                "categories": f"{base_url}hairstyles/categories/",
                "detail": f"{base_url}hairstyles/<style_id>/",
            },
            "Search & Filter": {
                "search": f"{base_url}search/",
                "face_shapes": f"{base_url}filter/face-shapes/",
                "occasions": f"{base_url}filter/occasions/",
            },
            "User Features": {
                "recommendations_history": f"{base_url}user/recommendations/",
                "favorites": f"{base_url}user/favorites/",
                "history": f"{base_url}user/history/",
            },
            "System": {
                "health_check": f"{base_url}health/",
                "analytics": f"{base_url}analytics/event/",
            },
        },
        "documentation": "Visit /api/ for interactive API documentation",
        "system_status": {
            'ml_available': ML_MODEL_AVAILABLE,
            'preprocess_available': ML_PREPROCESS_AVAILABLE,
            'recommendation_available': RECOMMENDATION_ENGINE_AVAILABLE,
            'overlay_available': OVERLAY_PROCESSOR_AVAILABLE,
            'analytics_available': ANALYTICS_AVAILABLE,
        },
    }

    return Response(endpoints)
