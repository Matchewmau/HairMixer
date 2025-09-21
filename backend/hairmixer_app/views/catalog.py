from datetime import timedelta

from django.db.models import Q, Avg, Count
from django.utils import timezone
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from drf_spectacular.utils import (
    extend_schema,
    OpenApiResponse,
    OpenApiParameter,
)

from ..models import Hairstyle, HairstyleCategory, Feedback
from ..serializers import (
    HairstyleSerializer,
    FeedbackSerializer,
    HairstyleCategorySerializer,
)
from ..services.analytics_utils import track_event_safe
from .base import logger, analytics_service


class FeaturedHairstylesView(APIView):
    permission_classes = [AllowAny]
    serializer_class = None

    @extend_schema(
        responses={
            200: OpenApiResponse(description='Featured hairstyles list')
        }
    )
    def get(self, request):
        try:
            limit = min(int(request.query_params.get('limit', 20)), 50)

            featured_styles = (
                Hairstyle.objects.filter(is_active=True, is_featured=True)
                .select_related('category')
                .order_by('-trend_score')[:limit]
            )

            serializer = HairstyleSerializer(
                featured_styles, many=True, context={'request': request}
            )

            return Response(
                {
                    'featured_hairstyles': serializer.data,
                    'count': len(serializer.data),
                }
            )
        except Exception as e:
            logger.error(f"Error fetching featured hairstyles: {str(e)}")
            return Response(
                {"error": "Failed to fetch featured hairstyles"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class TrendingHairstylesView(APIView):
    permission_classes = [AllowAny]
    serializer_class = None

    @extend_schema(
        responses={
            200: OpenApiResponse(description='Trending hairstyles list')
        }
    )
    def get(self, request):
        try:
            limit = min(int(request.query_params.get('limit', 20)), 50)

            trending_styles = (
                Hairstyle.objects.filter(is_active=True)
                .annotate(
                    recent_recommendations=Count(
                        'recommendationlog',
                        filter=Q(
                            recommendationlog__created_at__gte=(
                                timezone.now() - timedelta(days=7)
                            )
                        ),
                    ),
                    avg_rating=Avg('feedback__rating'),
                )
                .order_by(
                    '-recent_recommendations',
                    '-avg_rating',
                    '-popularity_score',
                )
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

            return Response(
                {
                    'trending_hairstyles': serializer.data,
                    'count': len(serializer.data),
                }
            )
        except Exception as e:
            logger.error(f"Error fetching trending hairstyles: {str(e)}")
            return Response(
                {"error": "Failed to fetch trending hairstyles"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class HairstyleDetailView(APIView):
    permission_classes = [AllowAny]
    serializer_class = None

    @extend_schema(
        parameters=[
            OpenApiParameter('style_id', str, OpenApiParameter.PATH),
        ],
        responses={200: OpenApiResponse(description='Hairstyle detail')}
    )
    def get(self, request, style_id):
        try:
            style = get_object_or_404(
                Hairstyle.objects.select_related('category'),
                id=style_id,
                is_active=True,
            )

            recent_feedback = Feedback.objects.filter(
                hairstyle=style, is_public=True
            ).order_by('-created_at')[:5]

            feedback_stats = Feedback.objects.filter(
                hairstyle=style
            ).aggregate(
                avg_rating=Avg('rating'),
                total_feedback=Count('id'),
                positive_feedback=Count('id', filter=Q(liked=True)),
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

            return Response(
                {
                    'hairstyle': serializer.data,
                    'feedback_stats': feedback_stats,
                    'recent_feedback': feedback_serializer.data,
                    'related_styles': [],
                }
            )
        except Exception as e:
            logger.error(f"Error fetching hairstyle detail: {str(e)}")
            return Response(
                {"error": "Failed to fetch hairstyle details"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ListHairstylesView(APIView):
    permission_classes = [AllowAny]
    serializer_class = None

    @extend_schema(
        parameters=[
            OpenApiParameter('category', str, OpenApiParameter.QUERY),
            OpenApiParameter('face_shape', str, OpenApiParameter.QUERY),
            OpenApiParameter('occasion', str, OpenApiParameter.QUERY),
            OpenApiParameter('maintenance', str, OpenApiParameter.QUERY),
            OpenApiParameter('featured', bool, OpenApiParameter.QUERY),
            OpenApiParameter('limit', int, OpenApiParameter.QUERY),
        ],
        responses={200: OpenApiResponse(description='Hairstyle list')}
    )
    def get(self, request):
        try:
            category = request.query_params.get('category')
            face_shape = request.query_params.get('face_shape')
            occasion = request.query_params.get('occasion')
            maintenance = request.query_params.get('maintenance')
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

            return Response(
                {
                    'hairstyles': serializer.data,
                    'total_count': len(serializer.data),
                    'filters_applied': {
                        'category': category,
                        'face_shape': face_shape,
                        'occasion': occasion,
                        'maintenance': maintenance,
                        'featured_only': featured_only,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error fetching hairstyles: {str(e)}")
            return Response(
                {"error": "Failed to fetch hairstyles", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class HairstyleCategoriesView(APIView):
    permission_classes = [AllowAny]
    serializer_class = None

    @extend_schema(
        responses={
            200: OpenApiResponse(description='Hairstyle categories list')
        }
    )
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
    permission_classes = [AllowAny]

    from drf_spectacular.utils import (
        extend_schema,
        OpenApiParameter,
        OpenApiResponse,
    )

    @extend_schema(
        parameters=[
            OpenApiParameter(
                'q', str, OpenApiParameter.QUERY, description='Search text'
            ),
            OpenApiParameter(
                'face_shape',
                str,
                OpenApiParameter.QUERY,
                enum=['oval', 'round', 'square', 'heart', 'diamond', 'oblong'],
            ),
            OpenApiParameter(
                'occasion',
                str,
                OpenApiParameter.QUERY,
                enum=[
                    'casual',
                    'formal',
                    'party',
                    'business',
                    'wedding',
                    'date',
                    'work',
                ],
            ),
            OpenApiParameter(
                'hair_type',
                str,
                OpenApiParameter.QUERY,
                enum=['straight', 'wavy', 'curly', 'coily'],
            ),
            OpenApiParameter(
                'maintenance',
                str,
                OpenApiParameter.QUERY,
                enum=['low', 'medium', 'high'],
            ),
            OpenApiParameter('page', int, OpenApiParameter.QUERY, default=1),
            OpenApiParameter(
                'per_page', int, OpenApiParameter.QUERY, default=20
            ),
        ],
        responses={
            200: OpenApiResponse(description='Search results returned')
        },
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
            engine = ''
            try:
                from django.db import connection as dj_conn

                engine = dj_conn.settings_dict.get('ENGINE', '')
            except Exception:
                pass
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

            if face_shape and supports_json:
                queryset = queryset.filter(face_shapes__contains=[face_shape])
            if occasion and supports_json:
                queryset = queryset.filter(occasions__contains=[occasion])
            if hair_type and supports_json:
                queryset = queryset.filter(hair_types__contains=[hair_type])
            if maintenance:
                queryset = queryset.filter(maintenance=maintenance)

            queryset = queryset.order_by(
                '-trend_score', '-popularity_score', 'name'
            )

            paginator = Paginator(queryset, per_page)
            page_obj = paginator.get_page(page)

            serializer = HairstyleSerializer(
                page_obj.object_list, many=True, context={'request': request}
            )

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

            return Response(
                {
                    'results': serializer.data,
                    'search_query': query,
                    'filters_applied': {
                        'face_shape': face_shape,
                        'occasion': occasion,
                        'hair_type': hair_type,
                        'maintenance': maintenance,
                    },
                    'pagination': {
                        'page': page,
                        'per_page': per_page,
                        'total_pages': paginator.num_pages,
                        'total_count': paginator.count,
                        'has_next': page_obj.has_next(),
                        'has_previous': page_obj.has_previous(),
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            return Response(
                {"error": "Search failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class FaceShapesView(APIView):
    permission_classes = [AllowAny]
    serializer_class = None

    @extend_schema(
        responses={200: OpenApiResponse(description='Face shape definitions')}
    )
    def get(self, request):
        try:
            from ..ml.model import FACE_SHAPE_CHARACTERISTICS

            return Response(
                {
                    'face_shapes': [
                        {
                            'value': shape,
                            'label': shape.title(),
                            'description': data['description'],
                            'suitable_styles': data['suitable_styles'],
                            'avoid': data['avoid'],
                        }
                        for shape, data in FACE_SHAPE_CHARACTERISTICS.items()
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Error fetching face shapes: {str(e)}")
            return Response(
                {"error": "Failed to fetch face shapes"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class OccasionsView(APIView):
    permission_classes = [AllowAny]
    serializer_class = None

    @extend_schema(
        responses={200: OpenApiResponse(description='Occasions list')}
    )
    def get(self, request):
        try:
            occasions = [
                {'value': 'casual', 'label': 'Casual'},
                {'value': 'formal', 'label': 'Formal'},
                {'value': 'party', 'label': 'Party'},
                {'value': 'business', 'label': 'Business'},
                {'value': 'wedding', 'label': 'Wedding'},
                {'value': 'date', 'label': 'Date Night'},
                {'value': 'work', 'label': 'Work'},
            ]

            return Response({'occasions': occasions})
        except Exception as e:
            logger.error(f"Error fetching occasions: {str(e)}")
            return Response(
                {"error": "Failed to fetch occasions"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
