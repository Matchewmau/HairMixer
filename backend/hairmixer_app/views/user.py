from django.core.paginator import Paginator
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.parsers import JSONParser
import logging
import traceback

from ..models import (
    UserPreference, PreferenceProfile, RecommendationLog, Hairstyle, Feedback,
    SavedHairstyle, HairstyleLike
)
from ..serializers import (
    UserPreferenceSerializer, FeedbackSerializer, RecommendationLogSerializer,
    PreferenceProfileSerializer, SavedHairstyleSerializer, HairstyleLikeSerializer
)
from ..services.analytics_utils import track_event_safe
from ..services.analytics import AnalyticsService

# Initialize services
try:
    analytics_service = AnalyticsService()
except ImportError:
    analytics_service = None

logger = logging.getLogger(__name__)

class SetPreferencesView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            data = request.data
            logger.info(f"Received preferences data: {data}")
            
            preference_data = {
                'faceshape': data.get('faceshape', ''),
                'hair_type': data.get('hair_type', ''),
                'hair_length': data.get('hair_length', ''),
                'lifestyle': data.get('lifestyle', ''),
                'maintenance': data.get('maintenance', ''),
                'occasions': data.get('occasions', []),
                'gender': data.get('gender', ''),
                'hair_color': data.get('hair_color', ''),
                'color_preference': data.get('color_preference', ''),
                'budget_range': data.get('budget_range', ''),
                'volume': data.get('volume', ''),
                'styling_maintenance': data.get('styling_maintenance', ''),
                'hair_texture_detail': data.get('hair_texture_detail', ''),
                'styling_preference': data.get('styling_preference', ''),
                'hair_condition': data.get('hair_condition', []),
                'hair_thickness': data.get('hair_thickness', ''),
                'wants_bangs': data.get('wants_bangs', False),
                'hairstyle_family': data.get('hairstyle_family', ''),
                'hairstyle_name': data.get('hairstyle_name', ''),
                'avoid_styles': data.get('avoid_styles', []),
            }
            
            filtered_data = {}
            for k, v in preference_data.items():
                if isinstance(v, list) or (v != '' and v is not None):
                    filtered_data[k] = v
            preference_data = filtered_data
            
            if not isinstance(preference_data.get('occasions', []), list):
                preference_data['occasions'] = []
            
            required_fields = ['hair_type', 'hair_length', 'maintenance']
            missing_fields = []
            for field in required_fields:
                if not preference_data.get(field):
                    missing_fields.append(field)
            
            if missing_fields:
                error_msg = f"Required fields missing: {', '.join(missing_fields)}"
                logger.error(f"Validation error: {error_msg}")
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            valid_hair_types = ['straight', 'wavy', 'curly', 'coily']
            if preference_data['hair_type'] not in valid_hair_types:
                return Response(
                    {"error": f"Invalid hair_type. Must be one of: {valid_hair_types}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            serializer = UserPreferenceSerializer(data=preference_data)
            
            if not serializer.is_valid():
                logger.error("Preference validation errors: %s", serializer.errors)
                return Response(
                    {
                        "error": "Invalid preferences",
                        "details": serializer.errors,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            
            preference = serializer.save(
                user=(
                    request.user
                    if (
                        hasattr(request, 'user') and getattr(
                            request.user, 'is_authenticated', False
                        )
                    )
                    else None
                )
            )
            
            return Response({
                'success': True,
                'preference_id': str(preference.id),
                'message': 'Preferences saved successfully',
                'preferences': serializer.data
            })
            
        except Exception as e:
            logger.error(f"Error saving preferences: {str(e)}")
            traceback.print_exc()
            return Response(
                {"error": "Failed to save preferences", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class FeedbackView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            serializer = FeedbackSerializer(data=request.data)
            if serializer.is_valid():
                fb = serializer.save(
                    user=(
                        request.user if request.user.is_authenticated else None
                    )
                )
                
                if fb.hairstyle:
                    fb.hairstyle.update_popularity()
                
                track_event_safe(
                    analytics_service,
                    user=(
                        request.user if request.user.is_authenticated else None
                    ),
                    event_type='feedback_submitted',
                    event_data={
                        'feedback_id': str(fb.id),
                        'liked': fb.liked,
                        'rating': fb.rating,
                        'has_note': bool(fb.note)
                    },
                    request=request
                )
                
                return Response({
                    "feedback_id": fb.id,
                    "message": "Feedback submitted successfully"
                })
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )
            
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            return Response(
                {"error": "Failed to save feedback", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
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
            
            all_ids = []
            for rec in page_obj.object_list:
                if rec.candidates:
                    all_ids.extend(rec.candidates)
            unique_ids = list({str(i) for i in all_ids}) if all_ids else []
            hairstyle_cache = {}
            if unique_ids:
                qs = Hairstyle.objects.filter(
                    id__in=unique_ids,
                    is_active=True,
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


class UserFavoritesView(APIView):
    """
    Get user's favorite hairstyles (SavedHairstyles).
    Alias for SavedHairstyleListCreateView for frontend compatibility.
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        saved_hairstyles = SavedHairstyle.objects.filter(
            user=request.user
        ).select_related('hairstyle', 'hairstyle__category')
        
        serializer = SavedHairstyleSerializer(saved_hairstyles, many=True)
        serializer = SavedHairstyleSerializer(saved_hairstyles, many=True)
        # Frontend expects a list, not an object with 'favorites' key
        return Response(serializer.data)

    def post(self, request):
        """Allow saving favorites via this endpoint too"""
        serializer = SavedHairstyleSerializer(
            data=request.data,
            context={'request': request}
        )
        if serializer.is_valid():
            saved = serializer.save()
            
            track_event_safe(
                analytics_service,
                user=request.user,
                event_type='hairstyle_favorited',
                event_data={
                    'saved_id': str(saved.id),
                    'hairstyle_id': str(saved.hairstyle.id),
                    'hairstyle_name': saved.hairstyle_name
                },
                request=request
            )
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserHistoryView(APIView):
    """Get user's activity history (placeholder)"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        # TODO: Implement user history
        return Response({
            'history': [],
            'message': 'History feature coming soon'
        })


class PreferenceProfileListCreateView(APIView):
    """
    GET: List all preference profiles for the authenticated user
    POST: Create a new preference profile
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        profiles = PreferenceProfile.objects.filter(user=request.user)
        serializer = PreferenceProfileSerializer(profiles, many=True)
        return Response({
            'profiles': serializer.data,
            'count': profiles.count()
        })
    
    def post(self, request):
        serializer = PreferenceProfileSerializer(
            data=request.data,
            context={'request': request}
        )
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PreferenceProfileDetailView(APIView):
    """
    GET: Retrieve a specific preference profile
    PUT: Update a preference profile
    PATCH: Partially update a preference profile
    DELETE: Delete a preference profile
    """
    permission_classes = [IsAuthenticated]
    
    def get_object(self, profile_id, user):
        try:
            return PreferenceProfile.objects.get(id=profile_id, user=user)
        except PreferenceProfile.DoesNotExist:
            return None
    
    def get(self, request, profile_id):
        profile = self.get_object(profile_id, request.user)
        if not profile:
            return Response(
                {'error': 'Profile not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        serializer = PreferenceProfileSerializer(profile)
        return Response(serializer.data)
    
    def put(self, request, profile_id):
        profile = self.get_object(profile_id, request.user)
        if not profile:
            return Response(
                {'error': 'Profile not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        serializer = PreferenceProfileSerializer(
            profile,
            data=request.data,
            context={'request': request}
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def patch(self, request, profile_id):
        profile = self.get_object(profile_id, request.user)
        if not profile:
            return Response(
                {'error': 'Profile not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        serializer = PreferenceProfileSerializer(
            profile,
            data=request.data,
            partial=True,
            context={'request': request}
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request, profile_id):
        profile = self.get_object(profile_id, request.user)
        if not profile:
            return Response(
                {'error': 'Profile not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        profile.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class PreferenceProfileSetDefaultView(APIView):
    """
    POST: Set a preference profile as the default
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, profile_id):
        try:
            profile = PreferenceProfile.objects.get(
                id=profile_id,
                user=request.user
            )
            profile.is_default = True
            profile.save()

            serializer = PreferenceProfileSerializer(profile)
            return Response(serializer.data)
        except PreferenceProfile.DoesNotExist:
            return Response(
                {'error': 'Profile not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class SavedHairstyleListCreateView(APIView):
    """
    GET: List all saved hairstyles for the authenticated user
    POST: Save a hairstyle
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        saved_hairstyles = SavedHairstyle.objects.filter(
            user=request.user
        ).select_related('hairstyle', 'hairstyle__category')
        
        serializer = SavedHairstyleSerializer(saved_hairstyles, many=True)
        serializer = SavedHairstyleSerializer(saved_hairstyles, many=True)
        # Frontend expects a list, not an object with 'saved_hairstyles' key
        return Response(serializer.data)

    def post(self, request):
        serializer = SavedHairstyleSerializer(
            data=request.data,
            context={'request': request}
        )
        if serializer.is_valid():
            saved = serializer.save()
            
            track_event_safe(
                analytics_service,
                user=request.user,
                event_type='hairstyle_saved',
                event_data={
                    'saved_id': str(saved.id),
                    'hairstyle_id': str(saved.hairstyle.id),
                    'hairstyle_name': saved.hairstyle_name
                },
                request=request
            )
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SavedHairstyleDetailView(APIView):
    """
    GET: Retrieve a specific saved hairstyle
    DELETE: Remove a saved hairstyle
    """
    permission_classes = [IsAuthenticated]

    def get_object(self, saved_id, user):
        try:
            return SavedHairstyle.objects.select_related(
                'hairstyle'
            ).get(id=saved_id, user=user)
        except SavedHairstyle.DoesNotExist:
            return None

    def get(self, request, saved_id):
        saved = self.get_object(saved_id, request.user)
        if not saved:
            return Response(
                {'error': 'Saved hairstyle not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        serializer = SavedHairstyleSerializer(saved)
        return Response(serializer.data)

    def delete(self, request, saved_id):
        saved = self.get_object(saved_id, request.user)
        if not saved:
            return Response(
                {'error': 'Saved hairstyle not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        saved.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class HairstyleLikeView(APIView):
    """
    POST: Like or dislike a hairstyle
    GET: Get user's reaction for a specific hairstyle
    DELETE: Remove reaction from a hairstyle
    
    Note: Likes/dislikes do NOT affect the recommendation order.
    They are for user feedback tracking only.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """Like or dislike a hairstyle"""
        serializer = HairstyleLikeSerializer(
            data=request.data,
            context={'request': request}
        )
        if serializer.is_valid():
            like = serializer.save()
            
            track_event_safe(
                analytics_service,
                user=request.user,
                event_type='hairstyle_reaction',
                event_data={
                    'like_id': str(like.id),
                    'hairstyle_id': str(like.hairstyle.id),
                    'reaction': like.reaction
                },
                request=request
            )
            
            return Response({
                'id': str(like.id),
                'hairstyle_id': str(like.hairstyle.id),
                'hairstyle_name': like.hairstyle.name,
                'reaction': like.reaction,
                'message': f'Successfully {like.reaction}d hairstyle'
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        """Remove reaction from a hairstyle"""
        hairstyle_id = request.data.get('hairstyle_id')
        if not hairstyle_id:
            return Response(
                {'error': 'hairstyle_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            like = HairstyleLike.objects.get(
                user=request.user,
                hairstyle_id=hairstyle_id
            )
            like.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except HairstyleLike.DoesNotExist:
            return Response(
                {'error': 'Reaction not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class HairstyleLikeStatsView(APIView):
    """
    GET: Get like/dislike stats for a hairstyle
    """
    permission_classes = [AllowAny]

    def get(self, request, hairstyle_id):
        """Get like/dislike counts for a specific hairstyle"""
        try:
            hairstyle = Hairstyle.objects.get(id=hairstyle_id)
        except Hairstyle.DoesNotExist:
            return Response(
                {'error': 'Hairstyle not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        likes_count = HairstyleLike.objects.filter(
            hairstyle=hairstyle,
            reaction='like'
        ).count()
        
        dislikes_count = HairstyleLike.objects.filter(
            hairstyle=hairstyle,
            reaction='dislike'
        ).count()
        
        # Get user's reaction if authenticated
        user_reaction = None
        if request.user.is_authenticated:
            user_like = HairstyleLike.objects.filter(
                user=request.user,
                hairstyle=hairstyle
            ).first()
            if user_like:
                user_reaction = user_like.reaction
        
        return Response({
            'hairstyle_id': str(hairstyle.id),
            'hairstyle_name': hairstyle.name,
            'likes_count': likes_count,
            'dislikes_count': dislikes_count,
            'user_reaction': user_reaction
        })


class HairstyleLikeBulkStatsView(APIView):
    """
    POST: Get like/dislike stats for multiple hairstyles at once
    """
    permission_classes = [AllowAny]

    def post(self, request):
        """Get like/dislike counts for multiple hairstyles"""
        hairstyle_ids = request.data.get('hairstyle_ids', [])
        
        if not hairstyle_ids:
            return Response(
                {'error': 'hairstyle_ids is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        stats = []
        hairstyles = Hairstyle.objects.filter(id__in=hairstyle_ids)
        
        for hairstyle in hairstyles:
            likes_count = HairstyleLike.objects.filter(
                hairstyle=hairstyle,
                reaction='like'
            ).count()
            
            dislikes_count = HairstyleLike.objects.filter(
                hairstyle=hairstyle,
                reaction='dislike'
            ).count()
            
            user_reaction = None
            if request.user.is_authenticated:
                user_like = HairstyleLike.objects.filter(
                    user=request.user,
                    hairstyle=hairstyle
                ).first()
                if user_like:
                    user_reaction = user_like.reaction
            
            stats.append({
                'hairstyle_id': str(hairstyle.id),
                'hairstyle_name': hairstyle.name,
                'likes_count': likes_count,
                'dislikes_count': dislikes_count,
                'user_reaction': user_reaction
            })
        
        return Response({'stats': stats})


class UserLikedHairstylesView(APIView):
    """
    GET: Get all hairstyles liked by the authenticated user
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Get list of user's liked/disliked hairstyles"""
        reaction_filter = request.query_params.get('reaction', None)
        
        likes = HairstyleLike.objects.filter(
            user=request.user
        ).select_related('hairstyle')
        
        if reaction_filter in ['like', 'dislike']:
            likes = likes.filter(reaction=reaction_filter)
        
        serializer = HairstyleLikeSerializer(likes, many=True)
        return Response({
            'likes': serializer.data,
            'count': likes.count()
        })
