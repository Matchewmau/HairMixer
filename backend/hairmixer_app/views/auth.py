from django.db import transaction
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import (
    api_view,
    permission_classes,
    throttle_classes,
    authentication_classes,
)
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
import logging

from ..models import CustomUser, UserProfile
from ..serializers import UserSerializer, UserRegistrationSerializer
from ..services.analytics_utils import track_event_safe

# Import analytics_service from a common place or initialize it here if needed
# For now, we'll assume it's available or import it from the main package if we move it to a common config
# To avoid circular imports, we might need to move the service initialization to a separate file like `services/__init__.py` or `apps.py`
# But for this refactor, let's try to import it from where it was, or re-initialize it safely.
# Since `views.py` had it global, we should probably put it in a `config.py` or similar.
# For now, let's re-initialize it here or import it from a new `common.py` if we create one.
# Actually, let's create a `common.py` in views/ to hold these shared instances.

from ..services.analytics import AnalyticsService
try:
    analytics_service = AnalyticsService()
except ImportError:
    analytics_service = None

logger = logging.getLogger(__name__)

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
        user = authenticate(request=request, username=email, password=password)
        
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
@permission_classes([AllowAny])  # Allow unauthenticated logout
def logout(request):
    try:
        refresh_token = request.data.get('refresh_token')
        if refresh_token:
            try:
                # Try to blacklist token if blacklist app is available
                token = RefreshToken(refresh_token)
                if hasattr(token, 'blacklist'):
                    token.blacklist()
                else:
                    logger.debug(
                        "Token blacklist not available. "
                        "Install rest_framework_simplejwt.token_blacklist "
                        "to enable token blacklisting."
                    )
            except Exception as e:
                # Token might be invalid/expired, but still allow logout
                logger.debug(f"Token blacklist failed: {str(e)}")
        
        # Log analytics event only if user is authenticated
        if request.user and request.user.is_authenticated:
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


@api_view(['GET', 'PUT', 'PATCH'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    try:
        if request.method == 'GET':
            user_data = UserSerializer(request.user).data
            return Response({
                'user': user_data
            }, status=status.HTTP_200_OK)
        
        elif request.method in ['PUT', 'PATCH']:
            # Update user profile
            serializer = UserSerializer(
                request.user,
                data=request.data,
                partial=(request.method == 'PATCH')
            )
            
            if serializer.is_valid():
                serializer.save()
                return Response({
                    'user': serializer.data,
                    'message': 'Profile updated successfully'
                }, status=status.HTTP_200_OK)
            
            return Response({
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
    
    except Exception as e:
        logger.error(f"User profile error: {str(e)}")
        return Response({
            'message': 'Failed to process user profile',
            'error': 'An unexpected error occurred'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
