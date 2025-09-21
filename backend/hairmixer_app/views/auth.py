from django.conf import settings
from django.db import transaction
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
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

from ..models import CustomUser, UserProfile
from ..serializers import UserSerializer, UserRegistrationSerializer
from ..services.analytics_utils import track_event_safe
from .base import logger, analytics_service
from drf_spectacular.utils import extend_schema, OpenApiResponse


@extend_schema(
    request=UserRegistrationSerializer,
    responses={
        201: OpenApiResponse(description='User created successfully'),
        400: OpenApiResponse(description='Validation failed'),
    },
)
@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
@throttle_classes([AnonRateThrottle])
def signup(request):
    try:
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            with transaction.atomic():
                user = CustomUser.objects.create(
                    username=serializer.validated_data['email'],
                    email=serializer.validated_data['email'],
                    first_name=serializer.validated_data['firstName'],
                    last_name=serializer.validated_data['lastName'],
                    password=make_password(
                        serializer.validated_data['password']
                    ),
                )

                UserProfile.objects.create(user=user)

                track_event_safe(
                    analytics_service,
                    user=user,
                    event_type='user_registered',
                    event_data={'source': 'web'},
                    request=request,
                )

                refresh = RefreshToken.for_user(user)
                access_token = refresh.access_token

                user_data = UserSerializer(user).data

                return Response(
                    {
                        'message': 'User created successfully',
                        'user': user_data,
                        'access_token': str(access_token),
                        'refresh_token': str(refresh),
                    },
                    status=status.HTTP_201_CREATED,
                )
        try:
            logger.error(f"Signup validation failed: {serializer.errors}")
        except Exception:
            pass
        return Response(
            {'message': 'Validation failed', 'errors': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return Response(
            {
                'message': 'Registration failed',
                'error': 'An unexpected error occurred',
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@extend_schema(
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'email': {'type': 'string', 'format': 'email'},
                'password': {'type': 'string'},
            },
            'required': ['email', 'password'],
        }
    },
    responses={
        200: OpenApiResponse(description='Login successful'),
        401: OpenApiResponse(description='Invalid credentials'),
    },
)
@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
@throttle_classes([AnonRateThrottle])
def login(request):
    try:
        email = request.data.get('email')
        password = request.data.get('password')

        if not email or not password:
            return Response(
                {'message': 'Email and password are required'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user = authenticate(username=email, password=password)

        if user:
            track_event_safe(
                analytics_service,
                user=user,
                event_type='user_login',
                event_data={'source': 'web'},
                request=request,
            )

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

            try:
                if getattr(settings, 'AUTH_COOKIES_ENABLED', False):
                    access_name = getattr(
                        settings, 'AUTH_COOKIE_ACCESS_NAME', 'access_token'
                    )
                    refresh_name = getattr(
                        settings, 'AUTH_COOKIE_REFRESH_NAME', 'refresh_token'
                    )
                    domain = (
                        getattr(settings, 'AUTH_COOKIE_DOMAIN', None) or None
                    )
                    samesite = getattr(settings, 'AUTH_COOKIE_SAMESITE', 'Lax')
                    secure_flag = not settings.DEBUG
                    access_lifetime = settings.SIMPLE_JWT.get(
                        'ACCESS_TOKEN_LIFETIME'
                    )
                    refresh_lifetime = settings.SIMPLE_JWT.get(
                        'REFRESH_TOKEN_LIFETIME'
                    )
                    access_max_age = (
                        int(access_lifetime.total_seconds())
                        if access_lifetime
                        else None
                    )
                    refresh_max_age = (
                        int(refresh_lifetime.total_seconds())
                        if refresh_lifetime
                        else None
                    )

                    response.set_cookie(
                        access_name,
                        str(access_token),
                        max_age=access_max_age,
                        httponly=True,
                        secure=secure_flag,
                        samesite=samesite,
                        domain=domain,
                        path='/',
                    )
                    response.set_cookie(
                        refresh_name,
                        str(refresh),
                        max_age=refresh_max_age,
                        httponly=True,
                        secure=secure_flag,
                        samesite=samesite,
                        domain=domain,
                        path='/',
                    )
            except Exception:
                pass

            return response

        return Response(
            {'message': 'Invalid email or password'},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return Response(
            {
                'message': 'Login failed',
                'error': 'An unexpected error occurred',
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@extend_schema(
    request={
        'application/json': {
            'type': 'object',
            'properties': {'refresh_token': {'type': 'string'}},
        }
    },
    responses={200: OpenApiResponse(description='Logout successful')},
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout(request):
    try:
        refresh_token = request.data.get('refresh_token')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()

        track_event_safe(
            analytics_service,
            user=request.user if request.user.is_authenticated else None,
            event_type='user_logout',
            request=request,
        )

        response = Response(
            {'message': 'Logout successful'}, status=status.HTTP_200_OK
        )

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
            {'message': 'Logout failed', 'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST,
        )


@extend_schema(responses={200: UserSerializer})
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    try:
        user_data = UserSerializer(request.user).data
        return Response({'user': user_data}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"User profile error: {str(e)}")
        return Response(
            {
                'message': 'Failed to fetch user profile',
                'error': 'An unexpected error occurred',
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
