from django.core.paginator import Paginator
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema, OpenApiResponse

from ..models import RecommendationLog, Hairstyle, SavedHairstyle
from ..serializers import (
    RecommendationLogSerializer,
    SavedHairstyleSerializer
)
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


class SavedHairstyleListCreateView(APIView):
    """
    GET: List all saved hairstyles for the authenticated user
    POST: Save a new hairstyle with user preferences
    """
    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="List saved hairstyles",
        description=(
            "Retrieve all hairstyles saved by the authenticated user. "
            "Each saved hairstyle includes the user preferences at the "
            "time of saving, allowing tracking of user preference patterns."
        ),
        responses={200: SavedHairstyleSerializer(many=True)}
    )
    def get(self, request):
        """Get all saved hairstyles for the current user"""
        saved_hairstyles = SavedHairstyle.objects.filter(
            user=request.user
        ).select_related('hairstyle', 'hairstyle__category')

        serializer = SavedHairstyleSerializer(
            saved_hairstyles,
            many=True,
            context={'request': request}
        )
        return Response(serializer.data)

    @extend_schema(
        summary="Save a hairstyle",
        description=(
            "Save a hairstyle recommendation with user preferences. "
            "The same hairstyle can be saved multiple times with "
            "different preference contexts."
        ),
        request=SavedHairstyleSerializer,
        responses={
            201: SavedHairstyleSerializer,
            400: OpenApiResponse(description='Invalid data')
        }
    )
    def post(self, request):
        """Save a new hairstyle with user preferences"""
        serializer = SavedHairstyleSerializer(
            data=request.data,
            context={'request': request}
        )
        if serializer.is_valid():
            saved = serializer.save()
            return Response(
                SavedHairstyleSerializer(
                    saved,
                    context={'request': request}
                ).data,
                status=status.HTTP_201_CREATED
            )
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


class SavedHairstyleDetailView(APIView):
    """
    GET: Retrieve a specific saved hairstyle
    PUT: Update a saved hairstyle (e.g., add notes)
    DELETE: Remove a saved hairstyle
    """
    permission_classes = [IsAuthenticated]

    def get_object(self, saved_id, user):
        """Get saved hairstyle for authenticated user"""
        try:
            return SavedHairstyle.objects.select_related(
                'hairstyle',
                'hairstyle__category'
            ).get(id=saved_id, user=user)
        except SavedHairstyle.DoesNotExist:
            return None

    @extend_schema(
        summary="Get saved hairstyle details",
        responses={
            200: SavedHairstyleSerializer,
            404: OpenApiResponse(description='Not found')
        }
    )
    def get(self, request, saved_id):
        """Get a specific saved hairstyle"""
        saved = self.get_object(saved_id, request.user)
        if not saved:
            return Response(
                {'error': 'Saved hairstyle not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = SavedHairstyleSerializer(
            saved,
            context={'request': request}
        )
        return Response(serializer.data)

    @extend_schema(
        summary="Update saved hairstyle",
        request=SavedHairstyleSerializer,
        responses={
            200: SavedHairstyleSerializer,
            404: OpenApiResponse(description='Not found')
        }
    )
    def put(self, request, saved_id):
        """Update a saved hairstyle (e.g., add notes)"""
        saved = self.get_object(saved_id, request.user)
        if not saved:
            return Response(
                {'error': 'Saved hairstyle not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = SavedHairstyleSerializer(
            saved,
            data=request.data,
            partial=True,
            context={'request': request}
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )

    @extend_schema(
        summary="Delete saved hairstyle",
        responses={
            204: OpenApiResponse(description='Deleted successfully'),
            404: OpenApiResponse(description='Not found')
        }
    )
    def delete(self, request, saved_id):
        """Delete a saved hairstyle"""
        logger.info(
            f"Delete request for saved_id: {saved_id}, "
            f"user: {request.user.email}"
        )
        saved = self.get_object(saved_id, request.user)
        if not saved:
            logger.warning(
                f"Saved hairstyle {saved_id} not found "
                f"for user {request.user.email}"
            )
            return Response(
                {'error': 'Saved hairstyle not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        saved.delete()
        logger.info(
            f"Deleted saved hairstyle {saved_id} "
            f"for user {request.user.email}"
        )
        return Response(status=status.HTTP_204_NO_CONTENT)
