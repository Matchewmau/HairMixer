from django.shortcuts import get_object_or_404
from django.db import transaction
from django.db.models import Q, Avg, Count
from django.utils import timezone
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import (
    api_view,
    permission_classes,
    throttle_classes,
    authentication_classes,
)
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.exceptions import ValidationError
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
from django.core.paginator import Paginator
from django.db import connection
from pathlib import Path
from datetime import timedelta
import logging
import traceback  # Add this
import uuid
from .models import (
    CustomUser,
    UserProfile,
    UploadedImage,
    UserPreference,
    Hairstyle,
    HairstyleCategory,
    RecommendationLog,
    Feedback,
)
from .serializers import (
    UserSerializer,
    UserRegistrationSerializer,
    HairstyleSerializer,
    FeedbackSerializer,
    RecommendationLogSerializer,
    AnalyticsEventSerializer,
    HairstyleCategorySerializer,
    UserPreferenceSerializer,
    UploadedImageSerializer,
    RecommendRequestSerializer,
    OverlayRequestSerializer,
)
from .services.image_service import ImageService
from .services.recommendation_service import (
    RecommendationService,
)
from .services.overlay_service import OverlayService
from .services.analytics_utils import track_event_safe
from .services.pipeline_service import RecommendationOverlayPipeline

# Initialize logger
logger = logging.getLogger(__name__)

# Track what's already imported to prevent duplicates
_ML_IMPORTS_DONE = False

if not _ML_IMPORTS_DONE:
    # Import ML components conditionally to prevent startup issues
    try:
        from .ml.preprocess import (
            read_image,
            detect_face,
            validate_image_quality,
        )
        ML_PREPROCESS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"ML preprocessing not available: {e}")
        ML_PREPROCESS_AVAILABLE = False

    try:
        from .ml.model import load_model
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

    _ML_IMPORTS_DONE = True

# Custom throttle classes


class ImageUploadThrottle(UserRateThrottle):
    rate = '10/hour'


class RecommendationThrottle(UserRateThrottle):
    rate = '20/hour'

# Global instances - Initialize conditionally


MODEL = None
recommendation_engine = (
    EnhancedRecommendationEngine() if RECOMMENDATION_ENGINE_AVAILABLE else None
)
overlay_processor = (
    AdvancedOverlayProcessor() if OVERLAY_PROCESSOR_AVAILABLE else None
)
analytics_service = AnalyticsService() if ANALYTICS_AVAILABLE else None
cache_manager = CacheManager() if CACHE_MANAGER_AVAILABLE else None
image_service = ImageService()
recommendation_service = RecommendationService()
overlay_service = OverlayService()
pipeline_service = RecommendationOverlayPipeline()

# Admin permission helper (env-gated)
try:
    ADMIN_PERMISSION_CLASS = (
        IsAuthenticated
        if getattr(settings, 'RELAX_ADMIN_PERMS', True)
        else IsAdminUser
    )
except Exception:
    ADMIN_PERMISSION_CLASS = IsAuthenticated


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
@authentication_classes([])
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
                    password=make_password(
                        serializer.validated_data['password']
                    )
                )
                
                # Create user profile
                UserProfile.objects.create(user=user)
                
                # Log analytics event
                track_event_safe(
                    analytics_service,
                    user=user,
                    event_type='user_registered',
                    event_data={'source': 'web'},
                    request=request,
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
        # Serializer invalid: return structured error with details
        try:
            logger.error(f"Signup validation failed: {serializer.errors}")
        except Exception:
            pass
        return Response({
            'message': 'Validation failed',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return Response({
            'message': 'Registration failed',
            'error': 'An unexpected error occurred'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

 
@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
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
            track_event_safe(
                analytics_service,
                user=user,
                event_type='user_login',
                event_data={'source': 'web'},
                request=request,
            )
            
            # Generate tokens
            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token
            
            user_data = UserSerializer(user).data
            
            resp_data = {
                'message': 'Login successful',
                'user': user_data,
                'access_token': str(access_token),
                'refresh_token': str(refresh),
            }

            response = Response(resp_data, status=status.HTTP_200_OK)

            # Optionally set HttpOnly cookies for tokens
            try:
                if getattr(settings, 'AUTH_COOKIES_ENABLED', False):
                    access_name = getattr(
                        settings, 'AUTH_COOKIE_ACCESS_NAME', 'access_token'
                    )
                    refresh_name = getattr(
                        settings, 'AUTH_COOKIE_REFRESH_NAME', 'refresh_token'
                    )
                    domain = getattr(
                        settings, 'AUTH_COOKIE_DOMAIN', None
                    ) or None
                    samesite = getattr(settings, 'AUTH_COOKIE_SAMESITE', 'Lax')
                    secure_flag = not settings.DEBUG
                    # Derive lifetimes from SIMPLE_JWT settings
                    access_lifetime = settings.SIMPLE_JWT.get(
                        'ACCESS_TOKEN_LIFETIME'
                    )
                    refresh_lifetime = settings.SIMPLE_JWT.get(
                        'REFRESH_TOKEN_LIFETIME'
                    )
                    access_max_age = (
                        int(access_lifetime.total_seconds())
                        if access_lifetime else None
                    )
                    refresh_max_age = (
                        int(refresh_lifetime.total_seconds())
                        if refresh_lifetime else None
                    )

                    response.set_cookie(
                        access_name,
                        str(access_token),
                        max_age=access_max_age,
                        httponly=True,
                        secure=secure_flag,
                        samesite=samesite,
                        domain=domain,
                        path='/'
                    )
                    response.set_cookie(
                        refresh_name,
                        str(refresh),
                        max_age=refresh_max_age,
                        httponly=True,
                        secure=secure_flag,
                        samesite=samesite,
                        domain=domain,
                        path='/'
                    )
            except Exception:
                # Cookie setting failure should not block login response
                pass

            return response
        
        return Response({
            'message': 'Invalid email or password'
        }, status=status.HTTP_401_UNAUTHORIZED)
    
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return Response(
            {
                'message': 'Login failed',
                'error': 'An unexpected error occurred',
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout(request):
    try:
        refresh_token = request.data.get('refresh_token')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        
        # Log analytics event
        if request.user.is_authenticated:
            track_event_safe(
                analytics_service,
                user=request.user,
                event_type='user_logout',
                request=request,
            )
        
        response = Response({
            'message': 'Logout successful'
        }, status=status.HTTP_200_OK)

        # Clear HttpOnly cookies if enabled
        try:
            if getattr(settings, 'AUTH_COOKIES_ENABLED', False):
                access_name = getattr(
                    settings, 'AUTH_COOKIE_ACCESS_NAME', 'access_token'
                )
                refresh_name = getattr(
                    settings, 'AUTH_COOKIE_REFRESH_NAME', 'refresh_token'
                )
                domain = getattr(settings, 'AUTH_COOKIE_DOMAIN', None) or None
                samesite = getattr(settings, 'AUTH_COOKIE_SAMESITE', 'Lax')
                response.delete_cookie(
                    access_name, path='/', domain=domain, samesite=samesite
                )
                response.delete_cookie(
                    refresh_name, path='/', domain=domain, samesite=samesite
                )
        except Exception:
            pass

        return response
    
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return Response(
            {
                'message': 'Logout failed',
                'error': str(e),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
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
    permission_classes = [AllowAny]
    # Disable authentication to avoid 401 when frontend sends no/invalid token
    authentication_classes = []
    throttle_classes = [ImageUploadThrottle]
    
    # schema annotations
    from drf_spectacular.utils import (
        extend_schema,
        OpenApiResponse,
        OpenApiExample,
        OpenApiParameter,
    )

    @extend_schema(
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'image': {
                        'type': 'string',
                        'format': 'binary',
                    }
                },
            }
        },
        responses={
            200: OpenApiResponse(
                description='Image uploaded and analyzed successfully'
            ),
            400: OpenApiResponse(
                description='Validation failed or no face detected'
            ),
            500: OpenApiResponse(
                description='Server error uploading image'
            ),
        },
        examples=[
            OpenApiExample(
                'Upload image (multipart)',
                value=None,
                request_only=True,
            ),
        ],
    )
    def post(self, request):
        try:
            if 'image' not in request.FILES:
                return Response(
                    {"error": "No image provided"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            image_file = request.FILES['image']
            
            # Validate file
            if not image_file.content_type.startswith('image/'):
                return Response(
                    {"error": "Invalid file type"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if image_file.size > 10 * 1024 * 1024:  # 10MB limit
                return Response(
                    {"error": "File too large"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Create upload record with serializer for validation/metadata
            serializer = UploadedImageSerializer(data={"image": image_file})
            try:
                serializer.is_valid(raise_exception=True)
            except ValidationError as ve:
                return Response(ve.detail, status=status.HTTP_400_BAD_REQUEST)
            # Save with user only if authenticated; otherwise keep it null
            uploaded = serializer.save(
                user=(
                    request.user
                    if request.user and request.user.is_authenticated
                    else None
                ),
                processing_status='processing',
            )
            
            # Process image immediately for better UX
            success, payload = ImageService.analyze_uploaded_image(uploaded)
            if not success:
                payload.update({
                    'success': False,
                    'image_id': str(uploaded.id),
                    'suggestions': [
                        'Make sure your face is clearly visible',
                        'Ensure good lighting',
                        'Face the camera directly',
                        'Remove sunglasses, hats, or face coverings',
                        'Try taking the photo from a different angle'
                    ]
                })
                return Response(payload, status=status.HTTP_400_BAD_REQUEST)

            payload.update({
                'message': 'Image uploaded and analyzed successfully',
                'face_shape_description': self.get_face_shape_description(
                    payload.get('face_shape', {}).get('shape')
                ),
            })
            return Response(payload)
                
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            return Response(
                {"error": "Failed to upload image"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
            logger.error(
                "Error processing image %s: %s", image_instance.id, e
            )
            image_instance.processing_status = 'failed'
            image_instance.error_message = str(e)
            image_instance.save()

    def get_face_shape_description(self, face_shape):
        """Get description for detected face shape"""
        descriptions = {
            'oval': 'Balanced proportions - most hairstyles suit you!',
            'round': (
                'Soft, curved features - try angular cuts '
                'and long layers'
            ),
            'square': 'Strong jawline - soft waves and layers work great',
            'heart': (
                'Wider forehead - styles with volume at the jaw are perfect'
            ),
            'diamond': (
                'Prominent cheekbones - textured styles balance your features'
            ),
            'oblong': 'Longer face - styles with width and volume are ideal',
        }
        return descriptions.get(
            face_shape, 'Unique face shape with many styling options!'
        )


class SetPreferencesView(APIView):
    parser_classes = (JSONParser,)
    # Allow anonymous submissions
    # Authenticated users will be linked automatically
    permission_classes = [AllowAny]
    authentication_classes = []
    
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
            preference_data = {
                k: v
                for k, v in preference_data.items()
                if v != ''
            }
            
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
                error_msg = (
                    "Required fields missing: "
                    f"{', '.join(missing_fields)}"
                )
                logger.error(f"Validation error: {error_msg}")
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validate field values against choices
            valid_hair_types = ['straight', 'wavy', 'curly', 'coily']
            if preference_data['hair_type'] not in valid_hair_types:
                error_msg = (
                    "Invalid hair_type "
                    f"'{preference_data['hair_type']}'. Must be one of: "
                    f"{valid_hair_types}"
                )
                logger.error(f"Validation error: {error_msg}")
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            valid_hair_lengths = [
                'pixie', 'short', 'medium', 'long', 'extra_long'
            ]
            if preference_data['hair_length'] not in valid_hair_lengths:
                error_msg = (
                    "Invalid hair_length "
                    f"'{preference_data['hair_length']}'. Must be one of: "
                    f"{valid_hair_lengths}"
                )
                logger.error(f"Validation error: {error_msg}")
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            valid_maintenance = ['low', 'medium', 'high']
            if preference_data['maintenance'] not in valid_maintenance:
                error_msg = (
                    "Invalid maintenance "
                    f"'{preference_data['maintenance']}'. Must be one of: "
                    f"{valid_maintenance}"
                )
                logger.error(f"Validation error: {error_msg}")
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validate gender if provided
            if preference_data.get('gender'):
                valid_genders = ['male', 'female', 'nb', 'other']
                if preference_data['gender'] not in valid_genders:
                    error_msg = (
                        "Invalid gender "
                        f"'{preference_data['gender']}'. Must be one of: "
                        f"{valid_genders}"
                    )
                    logger.error(f"Validation error: {error_msg}")
                    return Response(
                        {"error": error_msg},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            # Normalize and validate lifestyle if provided
            if preference_data.get('lifestyle'):
                # Map UI values to model choices
                lifestyle_map = {
                    'moderate': 'casual',
                    'relaxed': 'casual',
                }
                if preference_data['lifestyle'] in lifestyle_map:
                    preference_data['lifestyle'] = lifestyle_map[
                        preference_data['lifestyle']
                    ]
                valid_lifestyles = [
                    choice[0] for choice in UserPreference.LIFESTYLE_CHOICES
                ]
                if preference_data['lifestyle'] not in valid_lifestyles:
                    error_msg = (
                        "Invalid lifestyle "
                        f"'{preference_data['lifestyle']}'. Must be one of: "
                        f"{valid_lifestyles}"
                    )
                    logger.error(f"Validation error: {error_msg}")
                    return Response(
                        {"error": error_msg},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            # Validate occasions if provided (align with model choices)
            if preference_data.get('occasions'):
                valid_occasions = [
                    choice[0] for choice in UserPreference.OCCASION_CHOICES
                ]
                invalid_occasions = [
                    occ for occ in preference_data['occasions']
                    if occ not in valid_occasions
                ]
                if invalid_occasions:
                    error_msg = (
                        f"Invalid occasions {invalid_occasions}. "
                        f"Must be from: {valid_occasions}"
                    )
                    logger.error(f"Validation error: {error_msg}")
                    return Response(
                        {"error": error_msg},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            serializer = UserPreferenceSerializer(data=preference_data)
            if not serializer.is_valid():
                logger.error(
                    "Preference validation errors: %s", serializer.errors
                )
                return Response(
                    {
                        "error": "Invalid preferences",
                        "details": serializer.errors,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            # Save with user when authenticated, else anonymous record
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
            logger.info(
                "Created user preference %s with data: %s",
                preference.id,
                preference_data,
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


class RecommendView(APIView):
    parser_classes = (JSONParser,)
    # Allow anonymous recommendations; throttle applies regardless
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [RecommendationThrottle]
    
    from drf_spectacular.utils import (
        extend_schema,
        OpenApiResponse,
        OpenApiExample,
    )
    from .serializers import RecommendRequestSerializer

    @extend_schema(
        request=RecommendRequestSerializer,
        responses={
            200: OpenApiResponse(description='Recommendations generated'),
            400: OpenApiResponse(description='Bad request'),
            404: OpenApiResponse(
                description='Image or preferences not found'
            ),
            500: OpenApiResponse(
                description='Server error generating recommendations'
            ),
        },
        examples=[
            OpenApiExample(
                'Generate recommendations',
                value={
                    'image_id': (
                        '11111111-1111-1111-1111-111111111111'
                    ),
                    'preference_id': (
                        '33333333-3333-3333-3333-333333333333'
                    ),
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
            
            # Get objects with proper error handling
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
            
            # Only attach user if authenticated; else pass None
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


class OverlayView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [IsAuthenticated]
    
    # Schema annotations for API docs
    from drf_spectacular.utils import (
        extend_schema,
        OpenApiResponse,
        OpenApiExample,
    )
    from .serializers import (
        OverlayRequestSerializer,
        OverlayResponseSerializer,
    )

    @extend_schema(
        request=OverlayRequestSerializer,
        responses={
            200: OpenApiResponse(
                response=OverlayResponseSerializer,
                description='Overlay created successfully',
            ),
            400: OpenApiResponse(description='Bad request / validation error'),
            401: OpenApiResponse(description='Authentication required'),
            500: OpenApiResponse(description='Server error creating overlay'),
        },
        examples=[
            OpenApiExample(
                'Basic overlay request',
                value={
                    'image_id': '11111111-1111-1111-1111-111111111111',
                    'hairstyle_id': '22222222-2222-2222-2222-222222222222',
                    'overlay_type': 'basic',
                },
                request_only=True,
            ),
            OpenApiExample(
                'Successful response',
                value={
                    'overlay_url': '/media/overlays/<image>_<style>_basic.png',
                    'overlay_type': 'basic',
                },
                response_only=True,
            ),
        ],
    )
    def post(self, request):
        try:
            req_ser = OverlayRequestSerializer(data=request.data)
            req_ser.is_valid(raise_exception=True)
            image_id = req_ser.validated_data["image_id"]
            style_id = req_ser.validated_data["hairstyle_id"]
            overlay_type = req_ser.validated_data["overlay_type"]
            
            if not image_id or not style_id:
                return Response(
                    {"error": "Both image_id and hairstyle_id are required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            
            uploaded = get_object_or_404(UploadedImage, id=image_id)
            style = get_object_or_404(Hairstyle, id=style_id)

            overlay_url = overlay_service.generate(
                uploaded, style, overlay_type
            )

            # Log analytics event
            track_event_safe(
                analytics_service,
                user=(request.user if request.user.is_authenticated else None),
                event_type='overlay_generated',
                event_data={
                    'image_id': str(image_id),
                    'style_id': str(style_id),
                    'overlay_type': overlay_type,
                },
                request=request,
            )
            
            return Response({
                "overlay_url": overlay_url,
                "overlay_type": overlay_type
            })
            
        except ValidationError as e:
            # Return proper 400 for serializer validation issues
            return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)
        except ValueError as e:
            # Known input/state errors (e.g., missing hairstyle image)
            return Response(
                {"error": str(e)}, status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Error creating overlay: {str(e)}")
            return Response(
                {"error": "Failed to create overlay", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AutoOverlayView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]
    authentication_classes = []

    from drf_spectacular.utils import (
        extend_schema,
        OpenApiResponse,
        OpenApiExample,
    )
    from .serializers import RecommendRequestSerializer

    @extend_schema(
        request=RecommendRequestSerializer,
        responses={
            200: OpenApiResponse(description='Overlay generated from recommendation'),
            400: OpenApiResponse(description='Bad request'),
            404: OpenApiResponse(description='Not found'),
            500: OpenApiResponse(description='Server error'),
        },
        examples=[
            OpenApiExample(
                'Auto overlay',
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

            uploaded = get_object_or_404(UploadedImage, id=image_id)
            prefs = get_object_or_404(UserPreference, id=pref_id)

            overlay_type = 'advanced'
            if request.query_params.get('overlay') in ('basic', 'advanced'):
                overlay_type = request.query_params['overlay']

            result = pipeline_service.run(
                uploaded, prefs, overlay_type=overlay_type,
                user=(request.user if request.user.is_authenticated else None)
            )
            if 'error' in result:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            return Response(result)
        except ValidationError as e:
            return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error("Auto overlay error: %s", str(e))
            return Response(
                {"error": "Failed to generate auto overlay", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class FeedbackView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]  # Add this line
    
    def post(self, request):
        try:
            serializer = FeedbackSerializer(data=request.data)
            if serializer.is_valid():
                fb = serializer.save(
                    user=(
                        request.user if request.user.is_authenticated else None
                    )
                )
                
                # Update hairstyle popularity
                if fb.hairstyle:
                    fb.hairstyle.update_popularity()
                
                # Log analytics event
                analytics_service.track_event(
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


class AnalyticsEventView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Track custom analytics events from frontend"""
        try:
            serializer = AnalyticsEventSerializer(data=request.data)
            if serializer.is_valid():
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
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )
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
        'recommendation_available': RECOMMENDATION_ENGINE_AVAILABLE,
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
            "recommendation_available": RECOMMENDATION_ENGINE_AVAILABLE,
            "overlay_available": OVERLAY_PROCESSOR_AVAILABLE,
            "analytics_available": ANALYTICS_AVAILABLE
        }
    }
    
    return Response(endpoints)

# Add these missing views at the end of your views.py file:


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
    permission_classes = [AllowAny]  # Add this line
    
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
    permission_classes = [AllowAny]  # Add this line
    
    def get(self, request, style_id):
        try:
            style = get_object_or_404(
                Hairstyle.objects.select_related('category'),
                id=style_id,
                is_active=True,
            )
            
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
            # Batch-load candidate hairstyles to avoid N+1 in serializer
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
    
    from drf_spectacular.utils import (
        extend_schema,
        OpenApiParameter,
        OpenApiResponse,
    )

    @extend_schema(
        parameters=[
            OpenApiParameter(
                'q',
                str,
                OpenApiParameter.QUERY,
                description='Search text',
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
                    'casual', 'formal', 'party', 'business', 'wedding',
                    'date', 'work'
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
            OpenApiParameter(
                'page', int, OpenApiParameter.QUERY, default=1
            ),
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
            
            # Build query
            queryset = Hairstyle.objects.filter(is_active=True)
            engine = connection.settings_dict.get('ENGINE', '')
            supports_json = 'postgresql' in engine
            
            # Text search
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
            
            # Filters
            if face_shape:
                if supports_json:
                    queryset = queryset.filter(
                        face_shapes__contains=[face_shape]
                    )
            
            if occasion:
                if supports_json:
                    queryset = queryset.filter(
                        occasions__contains=[occasion]
                    )
            
            if hair_type:
                if supports_json:
                    queryset = queryset.filter(
                        hair_types__contains=[hair_type]
                    )
            
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
                page_obj.object_list,
                many=True,
                context={'request': request},
            )
            
            # Track search (safe)
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
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
            featured_only = (
                request.query_params.get('featured', '').lower() == 'true'
            )
            limit = min(int(request.query_params.get('limit', 50)), 100)
            
            # Build query
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
            
            # Order by relevance
            queryset = queryset.order_by(
                '-trend_score', '-popularity_score', 'name'
            )[:limit]
            
            serializer = HairstyleSerializer(
                queryset, many=True, context={'request': request}
            )
            
            # Log analytics event
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

# Additional utility views

class HairstyleCategoriesView(APIView):
    permission_classes = [AllowAny]  # Add this line
    
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


@api_view(['POST'])
@permission_classes([AllowAny])
def debug_face_detection(request):
    """Debug endpoint to test face detection components"""
    try:
        from .ml.face_analyzer import FacialFeatureAnalyzer
        
        analyzer = FacialFeatureAnalyzer()
        
        debug_info = {
            'detector_type': analyzer.detector_type,
            'mediapipe_available': analyzer.detector_type == 'mediapipe',
            'facenet_available': analyzer.detector_type == 'facenet',
            'face_detector_initialized': analyzer.face_detector is not None,
            'face_mesh_initialized': analyzer.face_mesh is not None,
            'device': str(analyzer.device),
        }
        
        # Test with a sample if image provided
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            
            # Save temporarily
            temp_path = f"/tmp/debug_face_{uuid.uuid4()}.jpg"
            with open(temp_path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)
            
            # Test detection
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
            
            # Cleanup
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
        
        from .ml.face_analyzer import FacialFeatureAnalyzer
        
        # Save temp image
        image_file = request.FILES['image']
        temp_path = f"/tmp/debug_resnet_{uuid.uuid4()}.jpg"
        with open(temp_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)
        
        # Initialize analyzer
        analyzer = FacialFeatureAnalyzer()
        
        # Check if ResNet50 is loaded
        debug_info = {
            'resnet_available': (
                hasattr(analyzer, 'feature_extractor') and (
                    analyzer.feature_extractor is not None
                )
            ),
            'classifier_type': getattr(analyzer, 'shape_classifier', 'None'),
            'device': str(analyzer.device),
        }
        
        # Test face analysis
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
        
        # Cleanup
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
