from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.db.models import Q, Count, Avg
from django.core.paginator import Paginator
import logging
from django.utils import timezone
from datetime import timedelta

from .models import (
    Hairstyle,
    Feedback,
    RecommendationLog,
)
from .serializers import (
    HairstyleSerializer,
    FeedbackSerializer,
    RecommendationLogSerializer,
)
from django.shortcuts import get_object_or_404
from .services.analytics_utils import track_event_safe
from .services.cache_manager import CacheManager

cache_manager = CacheManager()
analytics_service = None  # optional: wired by main views module when used

logger = logging.getLogger(__name__)


class FeaturedHairstylesView(APIView):
    """Get featured hairstyles"""
    
    def get(self, request):
        try:
            limit = min(int(request.query_params.get('limit', 20)), 50)
            
            featured_styles = Hairstyle.objects.filter(
                is_active=True,
                is_featured=True
            ).select_related('category').order_by('-trend_score')[:limit]
            
            serializer = HairstyleSerializer(
                featured_styles, many=True, context={'request': request}
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
    
    def get(self, request):
        try:
            limit = min(int(request.query_params.get('limit', 20)), 50)
            
            # Get trending styles based on recent recommendations and feedback
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
                '-recent_recommendations', '-avg_rating', '-popularity_score'
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
    
    def get(self, request, style_id):
        try:
            style = get_object_or_404(Hairstyle, id=style_id, is_active=True)
            
            # Get related data
            recent_feedback = Feedback.objects.filter(
                hairstyle=style,
                is_public=True
            ).order_by('-created_at')[:5]
            
            # Calculate statistics
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
                'related_styles': []  # TODO: Implement related styles logic
            })
            
        except Exception as e:
            logger.error(f"Error fetching hairstyle detail: {str(e)}")
            return Response(
                {"error": "Failed to fetch hairstyle details"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class UserRecommendationsView(APIView):
    """Get user's recommendation history"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        try:
            page = int(request.query_params.get('page', 1))
            per_page = min(int(request.query_params.get('per_page', 10)), 50)
            
            recommendations = RecommendationLog.objects.filter(
                user=request.user,
                status='completed'
            ).order_by('-created_at')
            
            paginator = Paginator(recommendations, per_page)
            page_obj = paginator.get_page(page)
            
            serializer = RecommendationLogSerializer(
                page_obj.object_list, many=True
            )
            
            return Response({
                'recommendations': serializer.data,
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
            logger.error(f"Error fetching user recommendations: {str(e)}")
            return Response(
                {"error": "Failed to fetch recommendations"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

class SearchView(APIView):
    """Search hairstyles with advanced filtering"""
    
    def get(self, request):
        try:
            query = request.query_params.get('q', '').strip()
            face_shape = request.query_params.get('face_shape', '')
            occasion = request.query_params.get('occasion', '')
            hair_type = request.query_params.get('hair_type', '')
            maintenance = request.query_params.get('maintenance', '')
            page = int(request.query_params.get('page', 1))
            per_page = min(int(request.query_params.get('per_page', 20)), 50)
            
            # Build query
            queryset = Hairstyle.objects.filter(is_active=True)
            
            # Text search
            if query:
                queryset = queryset.filter(
                    Q(name__icontains=query) |
                    Q(description__icontains=query) |
                    Q(tags__contains=[query])
                )
            
            # Filters
            if face_shape:
                queryset = queryset.filter(face_shapes__contains=[face_shape])
            
            if occasion:
                queryset = queryset.filter(occasions__contains=[occasion])
            
            if hair_type:
                queryset = queryset.filter(hair_types__contains=[hair_type])
            
            if maintenance:
                queryset = queryset.filter(maintenance=maintenance)
            
            # Order by relevance
            queryset = queryset.order_by(
                '-trend_score', '-popularity_score', 'name'
            )
            
            # Paginate
            paginator = Paginator(queryset, per_page)
            page_obj = paginator.get_page(page)
            
            serializer = HairstyleSerializer(
                page_obj.object_list, many=True, context={'request': request}
            )
            
            # Track search
            track_event_safe(
                analytics_service,
                user=(request.user if request.user.is_authenticated else None),
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

class CacheStatsView(APIView):
    """Get cache statistics (admin only)"""
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        try:
            stats = cache_manager.get_cache_stats()
            return Response({'cache_stats': stats})
        except Exception as e:
            logger.error(f"Error fetching cache stats: {str(e)}")
            return Response(
                {"error": "Failed to fetch cache stats"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

class CacheCleanupView(APIView):
    """Clean up expired cache entries (admin only)"""
    permission_classes = [IsAdminUser]
    
    def post(self, request):
        try:
            cleaned_count = cache_manager.cleanup_expired_cache()
            return Response({
                'message': f'Cleaned up {cleaned_count} expired cache entries',
                'cleaned_count': cleaned_count
            })
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")
            return Response(
                {"error": "Cache cleanup failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

class SystemAnalyticsView(APIView):
    """Get system analytics (admin only)"""
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        try:
            days = int(request.query_params.get('days', 7))
            analytics_data = analytics_service.get_system_analytics(days)
            return Response({'analytics': analytics_data})
        except Exception as e:
            logger.error(f"Error fetching system analytics: {str(e)}")
            return Response(
                {"error": "Failed to fetch analytics"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

class FaceShapesView(APIView):
    """Get available face shapes with descriptions"""
    
    def get(self, request):
        from .ml.model import FACE_SHAPE_CHARACTERISTICS
        
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

class OccasionsView(APIView):
    """Get available occasions"""
    
    def get(self, request):
        from .models import UserPreference
        
        occasions = [
            {'value': choice[0], 'label': choice[1]}
            for choice in UserPreference.OCCASION_CHOICES
        ]
        
        return Response({'occasions': occasions})