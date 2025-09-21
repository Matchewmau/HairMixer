from django.core.paginator import Paginator
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema, OpenApiResponse

from ..models import RecommendationLog, Hairstyle
from ..serializers import RecommendationLogSerializer
from .base import logger


class UserRecommendationsView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = None

    @extend_schema(
        responses={
            200: OpenApiResponse(description='User recommendations list')
        }
    )
    def get(self, request):
        try:
            page = int(request.query_params.get('page', 1))
            per_page = min(int(request.query_params.get('per_page', 10)), 50)

            recommendations = RecommendationLog.objects.filter(
                user=request.user, status='completed'
            ).order_by('-created_at')

            paginator = Paginator(recommendations, per_page)
            page_obj = paginator.get_page(page)

            all_ids = []
            for rec in page_obj.object_list:
                if rec.candidates:
                    all_ids.extend(rec.candidates)
            unique_ids = list({str(i) for i in all_ids}) if all_ids else []
            hairstyle_cache = {}
            if unique_ids:
                qs = Hairstyle.objects.filter(
                    id__in=unique_ids, is_active=True
                )
                for h in qs:
                    hairstyle_cache[str(h.id)] = h

            serializer = RecommendationLogSerializer(
                page_obj.object_list,
                many=True,
                context={
                    'request': request,
                    'hairstyle_cache': hairstyle_cache,
                },
            )

            return Response(
                {
                    'recommendations': serializer.data,
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
            logger.error(f"Error fetching user recommendations: {str(e)}")
            return Response(
                {"error": "Failed to fetch recommendations"}, status=500
            )


class UserFavoritesView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = None

    @extend_schema(
        responses={200: OpenApiResponse(description='Favorites placeholder')}
    )
    def get(self, request):
        return Response(
            {'favorites': [], 'message': 'Favorites feature coming soon'}
        )


class UserHistoryView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = None

    @extend_schema(
        responses={200: OpenApiResponse(description='History placeholder')}
    )
    def get(self, request):
        return Response(
            {'history': [], 'message': 'History feature coming soon'}
        )
