from django.shortcuts import render, get_object_or_404
from django.db import transaction
from django.db.models import Q, Avg, Count
from django.core.cache import cache
from django.utils import timezone
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
from pathlib import Path
from datetime import timedelta
import logging
import traceback  # Add this
import hashlib
import json

# Initialize logger FIRST before any imports that might use it
logger = logging.getLogger(__name__)

from .models import (
    CustomUser, UserProfile, UploadedImage, UserPreference, 
    Hairstyle, HairstyleCategory, RecommendationLog, Feedback,
    AnalyticsEvent, CachedRecommendation
)
from .serializers import (
    UserSerializer, UserRegistrationSerializer, UploadedImageSerializer, 
    UserPreferenceSerializer, HairstyleSerializer, FeedbackSerializer,
    RecommendationLogSerializer, AnalyticsEventSerializer,
    HairstyleCategorySerializer
)

# Import ML components conditionally to prevent startup issues
try:
    from .ml.preprocess import read_image, to_model_input, detect_face, validate_image_quality
    ML_PREPROCESS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML preprocessing not available: {e}")
    ML_PREPROCESS_AVAILABLE = False

try:
    from .ml.model import load_model, predict_face_shape, analyze_facial_features
    ML_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML model not available: {e}")
    ML_MODEL_AVAILABLE = False

try:
    from .logic.recommendation_engine import EnhancedRecommendationEngine
    RECOMMENDATION_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Recommendation engine not available: {e}")
    RECOMMENDATION_ENGINE_AVAILABLE = False

try:
    from .overlay import AdvancedOverlayProcessor
    OVERLAY_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Overlay processor not available: {e}")
    OVERLAY_PROCESSOR_AVAILABLE = False

try:
    from .services.analytics import AnalyticsService
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Analytics service not available: {e}")
    ANALYTICS_AVAILABLE = False

try:
    from .services.cache_manager import CacheManager
    CACHE_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cache manager not available: {e}")
    CACHE_MANAGER_AVAILABLE = False

# Custom throttle classes
class ImageUploadThrottle(UserRateThrottle):
    rate = '10/hour'

class RecommendationThrottle(UserRateThrottle):
    rate = '20/hour'

# Global instances - Initialize conditionally
MODEL = None
recommendation_engine = EnhancedRecommendationEngine() if RECOMMENDATION_ENGINE_AVAILABLE else None
overlay_processor = AdvancedOverlayProcessor() if OVERLAY_PROCESSOR_AVAILABLE else None
analytics_service = AnalyticsService() if ANALYTICS_AVAILABLE else None
cache_manager = CacheManager() if CACHE_MANAGER_AVAILABLE else None

def get_model():
    """Lazy loading of ML model"""
    global MODEL
    if MODEL is None and ML_MODEL_AVAILABLE:
        try:
            MODEL = load_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    return MODEL

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

# Authentication views (enhanced)
@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([AnonRateThrottle])
def signup(request):
    try:
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            with transaction.atomic():
                # Create user
                user = CustomUser.objects.create(
                    username=serializer.validated_data['email'],
                    email=serializer.validated_data['email'],
                    first_name=serializer.validated_data['firstName'],
                    last_name=serializer.validated_data['lastName'],
                    password=make_password(serializer.validated_data['password'])
                )
                
                # Create user profile
                UserProfile.objects.create(user=user)
                
                # Log analytics event
                analytics_service.track_event(
                    user=user,
                    event_type='user_registered',
                    event_data={'source': 'web'},
                    request=request
                )
                
                # Generate tokens
                refresh = RefreshToken.for_user(user)
                access_token = refresh.access_token
                
                user_data = UserSerializer(user).data
                
                return Response({
                    'message': 'User created successfully',
                    'user': user_data,
                    'access_token': str(access_token),
                    'refresh_token': str(refresh),
                }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return Response({
            'message': 'Registration failed',
            'error': 'An unexpected error occurred'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([AnonRateThrottle])
def login(request):
    try:
        email = request.data.get('email')
        password = request.data.get('password')
        
        if not email or not password:
            return Response({
                'message': 'Email and password are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Authenticate user
        user = authenticate(username=email, password=password)
        
        if user:
            # Log analytics event
            analytics_service.track_event(
                user=user,
                event_type='user_login',
                event_data={'source': 'web'},
                request=request
            )
            
            # Generate tokens
            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token
            
            user_data = UserSerializer(user).data
            
            return Response({
                'message': 'Login successful',
                'user': user_data,
                'access_token': str(access_token),
                'refresh_token': str(refresh),
            }, status=status.HTTP_200_OK)
        
        return Response({
            'message': 'Invalid email or password'
        }, status=status.HTTP_401_UNAUTHORIZED)
    
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return Response({
            'message': 'Login failed',
            'error': 'An unexpected error occurred'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def logout(request):
    try:
        refresh_token = request.data.get('refresh_token')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        
        # Log analytics event
        if request.user.is_authenticated:
            analytics_service.track_event(
                user=request.user,
                event_type='user_logout',
                request=request
            )
        
        return Response({
            'message': 'Logout successful'
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return Response({
            'message': 'Logout failed',
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def user_profile(request):
    try:
        user_data = UserSerializer(request.user).data
        return Response({
            'user': user_data
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"User profile error: {str(e)}")
        return Response({
            'message': 'Failed to fetch user profile',
            'error': 'An unexpected error occurred'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Enhanced Hair Analysis Views
class UploadImageView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]  # Add this line
    throttle_classes = [ImageUploadThrottle]
    
    def post(self, request):
        try:
            serializer = UploadedImageSerializer(data=request.data)
            if serializer.is_valid():
                with transaction.atomic():
                    # Save the image
                    instance = serializer.save(
                        user=request.user if request.user.is_authenticated else None
                    )
                    
                    # Process image asynchronously (or immediately for demo)
                    self.process_image(instance)
                    
                    # Log analytics event
                    analytics_service.track_event(
                        user=request.user if request.user.is_authenticated else None,
                        event_type='image_uploaded',
                        event_data={
                            'image_id': str(instance.id),
                            'file_size': instance.file_size,
                            'dimensions': f"{instance.image_width}x{instance.image_height}"
                        },
                        request=request
                    )
                    
                    return Response({
                        "image_id": instance.id,
                        "image_url": instance.image.url,
                        "face_detected": instance.face_detected,
                        "face_count": instance.face_count,
                        "processing_status": instance.processing_status
                    })
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            return Response(
                {"error": "Failed to upload image", "details": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def process_image(self, image_instance):
        """Process uploaded image for face detection and quality validation"""
        try:
            image_instance.processing_status = 'processing'
            image_instance.save()
            
            img_path = Path(settings.MEDIA_ROOT) / image_instance.image.name
            img = read_image(img_path)
            
            # Validate image quality
            quality_check = validate_image_quality(img)
            if not quality_check['is_valid']:
                image_instance.processing_status = 'failed'
                image_instance.error_message = quality_check['error']
                image_instance.save()
                return
            
            # Detect faces
            face_detected, faces = detect_face(img)
            image_instance.face_detected = face_detected
            image_instance.face_count = len(faces) if faces is not None else 0
            
            if face_detected:
                image_instance.processing_status = 'completed'
            else:
                image_instance.processing_status = 'failed'
                image_instance.error_message = 'No face detected in the image'
            
            image_instance.save()
            
        except Exception as e:
            logger.error(f"Error processing image {image_instance.id}: {str(e)}")
            image_instance.processing_status = 'failed'
            image_instance.error_message = str(e)
            image_instance.save()

class SetPreferencesView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            data = request.data
            logger.info(f"Received preferences data: {data}")
            
            # Extract only valid fields that exist in the UserPreference model
            preference_data = {
                'hair_type': data.get('hair_type', ''),
                'hair_length': data.get('hair_length', ''),
                'lifestyle': data.get('lifestyle', ''),
                'maintenance': data.get('maintenance', ''),
                'occasions': data.get('occasions', []),
                # Add other fields based on your model
                'gender': data.get('gender', ''),
                'hair_color': data.get('hair_color', ''),
                'color_preference': data.get('color_preference', ''),
                'budget_range': data.get('budget_range', ''),
            }
            
            # Remove empty strings to avoid validation errors
            preference_data = {k: v for k, v in preference_data.items() if v != ''}
            
            # Ensure occasions is a list
            if not isinstance(preference_data.get('occasions', []), list):
                preference_data['occasions'] = []
            
            # Validate required fields
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
            
            # Validate field values against choices
            valid_hair_types = ['straight', 'wavy', 'curly', 'coily']
            if preference_data['hair_type'] not in valid_hair_types:
                error_msg = f"Invalid hair_type '{preference_data['hair_type']}'. Must be one of: {valid_hair_types}"
                logger.error(f"Validation error: {error_msg}")
                return Response(
                    {"error": error_msg}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            valid_hair_lengths = ['pixie', 'short', 'medium', 'long', 'extra_long']
            if preference_data['hair_length'] not in valid_hair_lengths:
                error_msg = f"Invalid hair_length '{preference_data['hair_length']}'. Must be one of: {valid_hair_lengths}"
                logger.error(f"Validation error: {error_msg}")
                return Response(
                    {"error": error_msg}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            valid_maintenance = ['low', 'medium', 'high']
            if preference_data['maintenance'] not in valid_maintenance:
                error_msg = f"Invalid maintenance '{preference_data['maintenance']}'. Must be one of: {valid_maintenance}"
                logger.error(f"Validation error: {error_msg}")
                return Response(
                    {"error": error_msg}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validate gender if provided
            if preference_data.get('gender'):
                valid_genders = ['male', 'female', 'nb', 'other']
                if preference_data['gender'] not in valid_genders:
                    error_msg = f"Invalid gender '{preference_data['gender']}'. Must be one of: {valid_genders}"
                    logger.error(f"Validation error: {error_msg}")
                    return Response(
                        {"error": error_msg}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            # Validate lifestyle if provided
            if preference_data.get('lifestyle'):
                valid_lifestyles = ['active', 'professional', 'creative', 'casual', 'moderate', 'relaxed']
                if preference_data['lifestyle'] not in valid_lifestyles:
                    error_msg = f"Invalid lifestyle '{preference_data['lifestyle']}'. Must be one of: {valid_lifestyles}"
                    logger.error(f"Validation error: {error_msg}")
                    return Response(
                        {"error": error_msg}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            # Validate occasions if provided
            if preference_data.get('occasions'):
                valid_occasions = ['casual', 'formal', 'party', 'business', 'wedding', 'date', 'work']
                invalid_occasions = [occ for occ in preference_data['occasions'] if occ not in valid_occasions]
                if invalid_occasions:
                    error_msg = f"Invalid occasions {invalid_occasions}. Must be from: {valid_occasions}"
                    logger.error(f"Validation error: {error_msg}")
                    return Response(
                        {"error": error_msg}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            # Create preference record
            preference = UserPreference.objects.create(
                user=request.user if request.user.is_authenticated else None,
                **preference_data
            )
            
            logger.info(f"Created user preference {preference.id} with data: {preference_data}")
            
            return Response({
                'success': True,
                'preference_id': str(preference.id),
                'message': 'Preferences saved successfully',
                'preferences': preference_data
            })
            
        except Exception as e:
            logger.error(f"Error saving preferences: {str(e)}")
            traceback.print_exc()
            return Response(
                {"error": "Failed to save preferences", "details": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class RecommendView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]
    throttle_classes = [RecommendationThrottle]
    
    def post(self, request):
        start_time = timezone.now()
        
        try:
            image_id = request.data.get("image_id")
            pref_id = request.data.get("preference_id")
            
            logger.info(f"Recommendation request - image_id: {image_id}, pref_id: {pref_id}")
            
            if not image_id or not pref_id:
                return Response(
                    {"error": "Both image_id and preference_id are required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get objects with proper error handling
            try:
                uploaded = UploadedImage.objects.get(id=image_id)
                prefs = UserPreference.objects.get(id=pref_id)
            except UploadedImage.DoesNotExist:
                return Response(
                    {"error": "Uploaded image not found"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            except UserPreference.DoesNotExist:
                return Response(
                    {"error": "User preferences not found"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # For now, return sample recommendations since ML components might not be fully loaded
            processing_time = (timezone.now() - start_time).total_seconds()
            
            # Create a simple recommendation log
            log = RecommendationLog.objects.create(
                user=request.user if request.user.is_authenticated else None,
                uploaded=uploaded,
                preference=prefs,
                face_shape='oval',  # Default for now
                face_shape_confidence=0.8,
                detected_features={},
                candidates=[],
                recommendation_scores={},
                status='completed',
                processing_time=processing_time,
                model_version='v1.0'
            )
            
            # Return sample recommendations
            sample_recommendations = [
                {
                    'id': '1',
                    'name': 'Classic Bob',
                    'description': 'A timeless bob cut that suits most face shapes',
                    'image_url': None,
                    'category': 'Classic',
                    'difficulty': 'Easy',
                    'estimated_time': 30,
                    'maintenance': 'Medium',
                    'tags': ['classic', 'versatile'],
                    'match_score': 0.85
                },
                {
                    'id': '2', 
                    'name': 'Beach Waves',
                    'description': 'Relaxed, casual waves with natural texture',
                    'image_url': None,
                    'category': 'Casual',
                    'difficulty': 'Easy',
                    'estimated_time': 15,
                    'maintenance': 'Low',
                    'tags': ['casual', 'natural'],
                    'match_score': 0.78
                },
                {
                    'id': '3',
                    'name': 'Layered Cut',
                    'description': 'Versatile layered cut for medium-length hair',
                    'image_url': None,
                    'category': 'Versatile',
                    'difficulty': 'Medium',
                    'estimated_time': 35,
                    'maintenance': 'Medium',
                    'tags': ['layered', 'versatile'],
                    'match_score': 0.72
                }
            ]
            
            response_data = {
                "recommendation_id": str(log.id),
                "face_shape": "oval",
                "face_shape_confidence": 0.8,
                "detected_features": {},
                "recommended_styles": sample_recommendations,
                "candidates": sample_recommendations,
                "processing_time": f"{processing_time:.2f}s",
                "total_styles_analyzed": len(sample_recommendations)
            }
            
            return Response(response_data)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            traceback.print_exc()
            return Response(
                {"error": "Failed to generate recommendations", "details": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class OverlayView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]  # Add this line
    
    def post(self, request):
        try:
            image_id = request.data.get("image_id")
            style_id = request.data.get("hairstyle_id")
            overlay_type = request.data.get("overlay_type", "basic")  # basic, advanced
            
            if not image_id or not style_id:
                return Response(
                    {"error": "Both image_id and hairstyle_id are required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            uploaded = get_object_or_404(UploadedImage, id=image_id)
            style = get_object_or_404(Hairstyle, id=style_id)

            user_img_path = Path(settings.MEDIA_ROOT) / uploaded.image.name
            
            # Check if hairstyle has an image
            if style.image:
                style_img_path = Path(settings.MEDIA_ROOT) / style.image.name
            elif style.image_url:
                # Download image from URL (implement this)
                style_img_path = overlay_processor.download_style_image(style.image_url, style.id)
            else:
                return Response(
                    {"error": "Hairstyle image not available"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create output path
            out_rel = f"overlays/{image_id}_{style_id}_{overlay_type}.png"
            out_abs = Path(settings.MEDIA_ROOT) / out_rel
            out_abs.parent.mkdir(parents=True, exist_ok=True)

            # Generate overlay
            if overlay_type == "advanced":
                result_path = overlay_processor.create_advanced_overlay(
                    user_img_path, style_img_path, out_abs
                )
            else:
                result_path = overlay_processor.create_basic_overlay(
                    user_img_path, style_img_path, out_abs
                )
            
            # Log analytics event
            analytics_service.track_event(
                user=request.user if request.user.is_authenticated else None,
                event_type='overlay_generated',
                event_data={
                    'image_id': str(image_id),
                    'style_id': str(style_id),
                    'overlay_type': overlay_type
                },
                request=request
            )
            
            return Response({
                "overlay_url": f"{settings.MEDIA_URL}{out_rel}",
                "overlay_type": overlay_type
            })
            
        except Exception as e:
            logger.error(f"Error creating overlay: {str(e)}")
            return Response(
                {"error": "Failed to create overlay", "details": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class FeedbackView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]  # Add this line
    
    def post(self, request):
        try:
            serializer = FeedbackSerializer(data=request.data)
            if serializer.is_valid():
                fb = serializer.save(
                    user=request.user if request.user.is_authenticated else None
                )
                
                # Update hairstyle popularity
                if fb.hairstyle:
                    fb.hairstyle.update_popularity()
                
                # Log analytics event
                analytics_service.track_event(
                    user=request.user if request.user.is_authenticated else None,
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
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            return Response(
                {"error": "Failed to save feedback", "details": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AnalyticsEventView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]  # Add this line
    
    def post(self, request):
        """Track custom analytics events from frontend"""
        try:
            serializer = AnalyticsEventSerializer(data=request.data)
            if serializer.is_valid():
                analytics_service.track_event(
                    user=request.user if request.user.is_authenticated else None,
                    event_type=serializer.validated_data['event_type'],
                    event_data=serializer.validated_data.get('event_data', {}),
                    session_id=serializer.validated_data.get('session_id', ''),
                    request=request
                )
                return Response({"status": "event_tracked"})
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error tracking analytics event: {str(e)}")
            return Response(
                {"error": "Failed to track event"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Simple health check endpoint"""
    return Response({
        'status': 'ok',
        'message': 'HairMixer backend is running',
        'ml_available': ML_MODEL_AVAILABLE,
        'preprocess_available': ML_PREPROCESS_AVAILABLE,
        'recommendation_available': RECOMMENDATION_ENGINE_AVAILABLE,
        'overlay_available': OVERLAY_PROCESSOR_AVAILABLE,
        'analytics_available': ANALYTICS_AVAILABLE,
        'cache_available': CACHE_MANAGER_AVAILABLE,
    })

# Add these missing views at the end of your views.py file:

from django.core.paginator import Paginator

class FeaturedHairstylesView(APIView):
    """Get featured hairstyles"""
    permission_classes = [AllowAny]  # Add this line
    
    def get(self, request):
        try:
            limit = min(int(request.query_params.get('limit', 20)), 50)
            
            featured_styles = Hairstyle.objects.filter(
                is_active=True,
                is_featured=True
            ).select_related('category').order_by('-trend_score')[:limit]
            
            serializer = HairstyleSerializer(featured_styles, many=True, context={'request': request})
            
            return Response({
                'featured_hairstyles': serializer.data,
                'count': len(serializer.data)
            })
            
        except Exception as e:
            logger.error(f"Error fetching featured hairstyles: {str(e)}")
            return Response(
                {"error": "Failed to fetch featured hairstyles"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class TrendingHairstylesView(APIView):
    """Get trending hairstyles based on recent activity"""
    permission_classes = [AllowAny]  # Add this line
    
    def get(self, request):
        try:
            limit = min(int(request.query_params.get('limit', 20)), 50)
            
            # Get trending styles based on recent recommendations and feedback
            trending_styles = Hairstyle.objects.filter(
                is_active=True
            ).annotate(
                recent_recommendations=Count('recommendationlog', filter=Q(
                    recommendationlog__created_at__gte=timezone.now() - timedelta(days=7)
                )),
                avg_rating=Avg('feedback__rating')
            ).order_by('-recent_recommendations', '-avg_rating', '-popularity_score')[:limit]
            
            serializer = HairstyleSerializer(trending_styles, many=True, context={'request': request})
            
            if analytics_service:
                analytics_service.track_event(
                    user=request.user if request.user.is_authenticated else None,
                    event_type='trending_viewed',
                    event_data={'count': len(serializer.data)},
                    request=request
                )
            
            return Response({
                'trending_hairstyles': serializer.data,
                'count': len(serializer.data)
            })
            
        except Exception as e:
            logger.error(f"Error fetching trending hairstyles: {str(e)}")
            return Response(
                {"error": "Failed to fetch trending hairstyles"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class HairstyleDetailView(APIView):
    """Get detailed information about a specific hairstyle"""
    permission_classes = [AllowAny]  # Add this line
    
    def get(self, request, style_id):
        try:
            style = get_object_or_404(Hairstyle, id=style_id, is_active=True)
            
            # Get related data
            recent_feedback = Feedback.objects.filter(
                hairstyle=style,
                is_public=True
            ).order_by('-created_at')[:5]
            
            # Calculate statistics
            feedback_stats = Feedback.objects.filter(hairstyle=style).aggregate(
                avg_rating=Avg('rating'),
                total_feedback=Count('id'),
                positive_feedback=Count('id', filter=Q(liked=True))
            )
            
            serializer = HairstyleSerializer(style, context={'request': request})
            feedback_serializer = FeedbackSerializer(recent_feedback, many=True)
            
            if analytics_service:
                analytics_service.track_event(
                    user=request.user if request.user.is_authenticated else None,
                    event_type='hairstyle_viewed',
                    event_data={'style_id': str(style_id), 'style_name': style.name},
                    request=request
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
            
            serializer = RecommendationLogSerializer(page_obj.object_list, many=True)
            
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
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UserFavoritesView(APIView):
    """Get user's favorite hairstyles (placeholder)"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        # TODO: Implement favorites functionality
        return Response({
            'favorites': [],
            'message': 'Favorites feature coming soon'
        })

class UserHistoryView(APIView):
    """Get user's activity history (placeholder)"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        # TODO: Implement user history
        return Response({
            'history': [],
            'message': 'History feature coming soon'
        })

class SearchView(APIView):
    """Search hairstyles with advanced filtering"""
    permission_classes = [AllowAny]  # Add this line
    
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
            queryset = queryset.order_by('-trend_score', '-popularity_score', 'name')
            
            # Paginate
            paginator = Paginator(queryset, per_page)
            page_obj = paginator.get_page(page)
            
            serializer = HairstyleSerializer(page_obj.object_list, many=True, context={'request': request})
            
            # Track search
            if analytics_service:
                analytics_service.track_event(
                    user=request.user if request.user.is_authenticated else None,
                    event_type='search_performed',
                    event_data={
                        'query': query,
                        'filters': {
                            'face_shape': face_shape,
                            'occasion': occasion,
                            'hair_type': hair_type,
                            'maintenance': maintenance
                        },
                        'results_count': paginator.count
                    },
                    request=request
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
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class CacheStatsView(APIView):
    """Get cache statistics (admin only)"""
    permission_classes = [IsAuthenticated]  # Changed from IsAdminUser for testing
    
    def get(self, request):
        try:
            if cache_manager:
                stats = cache_manager.get_cache_stats()
                return Response({'cache_stats': stats})
            else:
                return Response({'cache_stats': {'message': 'Cache manager not available'}})
        except Exception as e:
            logger.error(f"Error fetching cache stats: {str(e)}")
            return Response(
                {"error": "Failed to fetch cache stats"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class CacheCleanupView(APIView):
    """Clean up expired cache entries (admin only)"""
    permission_classes = [IsAuthenticated]  # Changed from IsAdminUser for testing
    
    def post(self, request):
        try:
            if cache_manager:
                cleaned_count = cache_manager.cleanup_expired_cache()
                return Response({
                    'message': f'Cleaned up {cleaned_count} expired cache entries',
                    'cleaned_count': cleaned_count
                })
            else:
                return Response({'message': 'Cache manager not available'})
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")
            return Response(
                {"error": "Cache cleanup failed"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class SystemAnalyticsView(APIView):
    """Get system analytics (admin only)"""
    permission_classes = [IsAuthenticated]  # Changed from IsAdminUser for testing
    
    def get(self, request):
        try:
            days = int(request.query_params.get('days', 7))
            if analytics_service:
                analytics_data = analytics_service.get_system_analytics(days)
                return Response({'analytics': analytics_data})
            else:
                return Response({'analytics': {'message': 'Analytics service not available'}})
        except Exception as e:
            logger.error(f"Error fetching system analytics: {str(e)}")
            return Response(
                {"error": "Failed to fetch analytics"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class FaceShapesView(APIView):
    """Get available face shapes with descriptions"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
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
        except Exception as e:
            logger.error(f"Error fetching face shapes: {str(e)}")
            return Response(
                {"error": "Failed to fetch face shapes"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class OccasionsView(APIView):
    """Get available occasions"""
    permission_classes = [AllowAny]
    
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
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ListHairstylesView(APIView):
    permission_classes = [AllowAny]  # Add this line
    
    def get(self, request):
        try:
            # Get query parameters
            category = request.query_params.get('category')
            face_shape = request.query_params.get('face_shape')
            occasion = request.query_params.get('occasion')
            maintenance = request.query_params.get('maintenance')
            featured_only = request.query_params.get('featured', '').lower() == 'true'
            limit = min(int(request.query_params.get('limit', 50)), 100)
            
            # Build query
            queryset = Hairstyle.objects.filter(is_active=True).select_related('category')
            
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
            
            # Order by relevance
            queryset = queryset.order_by('-trend_score', '-popularity_score', 'name')[:limit]
            
            serializer = HairstyleSerializer(queryset, many=True, context={'request': request})
            
            # Log analytics event
            analytics_service.track_event(
                user=request.user if request.user.is_authenticated else None,
                event_type='hairstyles_browsed',
                event_data={
                    'filters': {
                        'category': category,
                        'face_shape': face_shape,
                        'occasion': occasion,
                        'maintenance': maintenance,
                        'featured_only': featured_only
                    },
                    'results_count': len(serializer.data)
                },
                request=request
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
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

# Additional utility views
class HairstyleCategoriesView(APIView):
    permission_classes = [AllowAny]  # Add this line
    
    def get(self, request):
        try:
            categories = HairstyleCategory.objects.filter(is_active=True).order_by('sort_order', 'name')
            serializer = HairstyleCategorySerializer(categories, many=True)
            return Response({'categories': serializer.data})
        except Exception as e:
            logger.error(f"Error fetching categories: {str(e)}")
            return Response(
                {"error": "Failed to fetch categories"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
