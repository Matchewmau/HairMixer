from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from rest_framework.parsers import JSONParser
import logging
import traceback
import uuid

from ..models import Hairstyle, HairstyleCategory
from ..serializers import AnalyticsEventSerializer, HairstyleCategorySerializer
from ..services.analytics import AnalyticsService
from ..services.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# Conditional imports and availability flags
try:
    from ..ml.preprocess import (
        read_image,
        detect_face,
        validate_image_quality,
    )
    ML_PREPROCESS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML preprocessing not available: {e}")
    ML_PREPROCESS_AVAILABLE = False

try:
    from ..ml.model import load_model
    ML_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML model not available: {e}")
    ML_MODEL_AVAILABLE = False

try:
    from ..overlay import AdvancedOverlayProcessor
    OVERLAY_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Overlay processor not available: {e}")
    OVERLAY_PROCESSOR_AVAILABLE = False

try:
    analytics_service = AnalyticsService()
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Analytics service not available: {e}")
    analytics_service = None
    ANALYTICS_AVAILABLE = False

try:
    cache_manager = CacheManager()
    CACHE_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cache manager not available: {e}")
    cache_manager = None
    CACHE_MANAGER_AVAILABLE = False

# Admin permission helper
try:
    ADMIN_PERMISSION_CLASS = (
        IsAuthenticated
        if getattr(settings, 'RELAX_ADMIN_PERMS', True)
        else IsAdminUser
    )
except Exception:
    ADMIN_PERMISSION_CLASS = IsAuthenticated


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Simple health check endpoint"""
    try:
        styles = Hairstyle.objects.filter(is_active=True).count()
        cats = HairstyleCategory.objects.filter(is_active=True).count()
    except Exception:
        styles = None
        cats = None
    return Response({
        'status': 'ok',
        'message': 'HairMixer backend is running',
        'ml_available': ML_MODEL_AVAILABLE,
        'preprocess_available': ML_PREPROCESS_AVAILABLE,
        'recommendation_available': True,
        'overlay_available': OVERLAY_PROCESSOR_AVAILABLE,
        'analytics_available': ANALYTICS_AVAILABLE,
        'cache_available': CACHE_MANAGER_AVAILABLE,
        'metrics': {'active_styles': styles, 'active_categories': cats},
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def api_root(request):
    """
    API Root - Shows all available endpoints
    """
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
                "user_profile": f"{base_url}auth/profile/"
            },
            "Core Features": {
                "upload_image": f"{base_url}upload/",
                "set_preferences": f"{base_url}preferences/",
                "get_recommendations": f"{base_url}recommend/",
                "create_overlay": f"{base_url}overlay/",
                "submit_feedback": f"{base_url}feedback/"
            },
            "Hairstyles": {
                "list_all": f"{base_url}hairstyles/",
                "featured": f"{base_url}hairstyles/featured/",
                "trending": f"{base_url}hairstyles/trending/",
                "categories": f"{base_url}hairstyles/categories/",
                "detail": f"{base_url}hairstyles/<style_id>/"
            },
            "Search & Filter": {
                "search": f"{base_url}search/",
                "face_shapes": f"{base_url}filter/face-shapes/",
                "occasions": f"{base_url}filter/occasions/"
            },
            "User Features": {
                "recommendations_history": f"{base_url}user/recommendations/",
                "favorites": f"{base_url}user/favorites/",
                "history": f"{base_url}user/history/"
            },
            "System": {
                "health_check": f"{base_url}health/",
                "analytics": f"{base_url}analytics/event/"
            }
        },
        "documentation": "Visit /api/ for interactive API documentation",
        "system_status": {
            "ml_available": ML_MODEL_AVAILABLE,
            "preprocess_available": ML_PREPROCESS_AVAILABLE,
            "recommendation_available": True,
            "overlay_available": OVERLAY_PROCESSOR_AVAILABLE,
            "analytics_available": ANALYTICS_AVAILABLE
        }
    }
    
    return Response(endpoints)


class AnalyticsEventView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Track custom analytics events from frontend"""
        try:
            serializer = AnalyticsEventSerializer(data=request.data)
            if serializer.is_valid():
                if analytics_service:
                    analytics_service.track_event(
                        user=(
                            request.user if request.user.is_authenticated else None
                        ),
                        event_type=serializer.validated_data['event_type'],
                        event_data=serializer.validated_data.get('event_data', {}),
                        session_id=serializer.validated_data.get('session_id', ''),
                        request=request
                    )
                    return Response({"status": "event_tracked"})
                return Response(
                    {"error": "Analytics service unavailable"},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Error tracking analytics event: {str(e)}")
            return Response(
                {"error": "Failed to track event"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CacheStatsView(APIView):
    """Get cache statistics (admin only)"""
    permission_classes = [ADMIN_PERMISSION_CLASS]
    
    def get(self, request):
        try:
            if cache_manager:
                stats = cache_manager.get_cache_stats()
                return Response({'cache_stats': stats})
            else:
                return Response(
                    {'cache_stats': {'message': 'Cache manager not available'}}
                )
        except Exception as e:
            logger.error(f"Error fetching cache stats: {str(e)}")
            return Response(
                {"error": "Failed to fetch cache stats"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class CacheCleanupView(APIView):
    """Clean up expired cache entries (admin only)"""
    permission_classes = [ADMIN_PERMISSION_CLASS]
    
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
            else:
                return Response({'message': 'Cache manager not available'})
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")
            return Response(
                {"error": "Cache cleanup failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SystemAnalyticsView(APIView):
    """Get system analytics (admin only)"""
    permission_classes = [ADMIN_PERMISSION_CLASS]
    
    def get(self, request):
        try:
            days = int(request.query_params.get('days', 7))
            if analytics_service:
                analytics_data = analytics_service.get_system_analytics(days)
                return Response({'analytics': analytics_data})
            else:
                return Response(
                    {
                        'analytics': {
                            'message': 'Analytics service not available'
                        }
                    }
                )
        except Exception as e:
            logger.error(f"Error fetching system analytics: {str(e)}")
            return Response(
                {"error": "Failed to fetch analytics"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class FaceShapesView(APIView):
    """Get available face shapes with descriptions"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            from ..ml.model import FACE_SHAPE_CHARACTERISTICS
            
            return Response({
                'face_shapes': [
                    {
                        'value': shape,
                        'label': shape.title(),
                        'description': data['description'],
                        'suitable_styles': data['suitable_styles'],
                        'avoid': data['avoid']
                    }
                    for shape, data in FACE_SHAPE_CHARACTERISTICS.items()
                ]
            })
        except Exception as e:
            logger.error(f"Error fetching face shapes: {str(e)}")
            return Response(
                {"error": "Failed to fetch face shapes"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class OccasionsView(APIView):
    """Get available occasions"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            occasions = [
                {'value': 'work', 'label': 'Work'},
                {'value': 'casual', 'label': 'Casual'},
                {'value': 'formal', 'label': 'Formal'},
                {'value': 'party', 'label': 'Party'},
                {'value': 'wedding', 'label': 'Wedding'},
                {'value': 'birthday', 'label': 'Birthday'},
            ]
            
            return Response({'occasions': occasions})
        except Exception as e:
            logger.error(f"Error fetching occasions: {str(e)}")
            return Response(
                {"error": "Failed to fetch occasions"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@api_view(['POST'])
@permission_classes([AllowAny])
def debug_face_detection(request):
    """Debug endpoint to test face detection components"""
    try:
        from ..ml.face_analyzer import FacialFeatureAnalyzer
        
        analyzer = FacialFeatureAnalyzer()
        
        debug_info = {
            'detector_type': analyzer.detector_type,
            'mediapipe_available': analyzer.detector_type == 'mediapipe',
            'facenet_available': analyzer.detector_type == 'facenet',
            'face_detector_initialized': analyzer.face_detector is not None,
            'face_mesh_initialized': analyzer.face_mesh is not None,
            'device': str(analyzer.device),
        }
        
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            
            temp_path = f"/tmp/debug_face_{uuid.uuid4()}.jpg"
            with open(temp_path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)
            
            result, error = analyzer.detect_and_analyze_face(temp_path)
            
            debug_info['detection_result'] = {
                'success': result is not None,
                'error': error,
                'face_detected': (
                    result.get('face_detected', False) if result else False
                ),
                'detection_method': (
                    result.get('detection_method', 'none') if result else 'none'
                ),
                'confidence': result.get('confidence', 0) if result else 0,
            }
            
            import os
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        
        return Response({'debug_info': debug_info})
        
    except Exception as e:
        return Response({'error': str(e), 'traceback': traceback.format_exc()})


@api_view(['POST'])
@permission_classes([AllowAny])
def debug_resnet_features(request):
    """Debug endpoint to verify ResNet50 feature extraction"""
    try:
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'})
        
        from ..ml.face_analyzer import FacialFeatureAnalyzer
        
        image_file = request.FILES['image']
        temp_path = f"/tmp/debug_resnet_{uuid.uuid4()}.jpg"
        with open(temp_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)
        
        analyzer = FacialFeatureAnalyzer()
        
        debug_info = {
            'resnet_available': (
                hasattr(analyzer, 'feature_extractor') and (
                    analyzer.feature_extractor is not None
                )
            ),
            'classifier_type': getattr(analyzer, 'shape_classifier', 'None'),
            'device': str(analyzer.device),
        }
        
        result, error = analyzer.detect_and_analyze_face(temp_path)
        
        if result:
            debug_info['face_detected'] = True
            debug_info['face_shape_result'] = result.get('face_shape', {})
            debug_info['detection_method'] = result.get(
                'detection_method', 'unknown'
            )
            face_shape_info = result.get('face_shape', {})
            debug_info['used_resnet'] = (
                face_shape_info.get('method') == 'resnet50_enhanced_geometric'
            )
            debug_info['feature_quality'] = face_shape_info.get(
                'feature_quality', 'N/A'
            )
        else:
            debug_info['face_detected'] = False
            debug_info['error'] = error
        
        import os
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        
        return Response({'debug_info': debug_info})
        
    except Exception as e:
        import traceback
        return Response(
            {
                'error': str(e),
                'traceback': traceback.format_exc(),
            }
        )
