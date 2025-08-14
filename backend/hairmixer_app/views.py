from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
from .models import CustomUser, UserProfile
from .serializers import UserSerializer, UserRegistrationSerializer

@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    try:
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
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
        return Response({
            'message': 'Registration failed',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
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
        return Response({
            'message': 'Login failed',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def logout(request):
    try:
        refresh_token = request.data.get('refresh_token')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        
        return Response({
            'message': 'Logout successful'
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
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
        return Response({
            'message': 'Failed to fetch user profile',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
