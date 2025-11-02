from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone

from ..models import PreferenceProfile
from ..serializers import PreferenceProfileSerializer


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
            # Update last_used_at timestamp
            profile.last_used_at = timezone.now()
            # Set this as default (will automatically unset other defaults)
            profile.is_default = True
            profile.save()
            
            serializer = PreferenceProfileSerializer(profile)
            return Response(serializer.data)
        except PreferenceProfile.DoesNotExist:
            return Response(
                {'error': 'Profile not found'},
                status=status.HTTP_404_NOT_FOUND
            )
