from pathlib import Path  # noqa: F401
from django.shortcuts import get_object_or_404
from django.db import connection  # noqa: F401
from rest_framework import status
from rest_framework.exceptions import ValidationError
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.core.paginator import Paginator  # noqa: F401
from drf_spectacular.utils import (
    extend_schema,
    OpenApiResponse,
    inline_serializer,
)
from rest_framework import serializers

from ..models import UploadedImage, UserPreference, Hairstyle
from ..serializers import (
    UploadedImageSerializer,
    UserPreferenceSerializer,
    RecommendRequestSerializer,
    OverlayRequestSerializer,
)
from ..services.analytics_utils import track_event_safe
from .base import analytics_service
from .base import (
    logger,
    ImageUploadThrottle,
    RecommendationThrottle,
    image_service,
    recommendation_service,
    overlay_service,
)
from ..services.pipeline_service import RecommendationOverlayPipeline


class UploadImageView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [ImageUploadThrottle]

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
                    'image': {'type': 'string', 'format': 'binary'}
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
            500: OpenApiResponse(description='Server error uploading image'),
        },
        examples=[
            OpenApiExample(
                'Upload image (multipart)', value=None, request_only=True
            )
        ],
    )
    def post(self, request):
        try:
            if 'image' not in request.FILES:
                return Response(
                    {"error": "No image provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            image_file = request.FILES['image']

            if not image_file.content_type.startswith('image/'):
                return Response(
                    {"error": "Invalid file type"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if image_file.size > 10 * 1024 * 1024:
                return Response(
                    {"error": "File too large"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            serializer = UploadedImageSerializer(data={"image": image_file})
            try:
                serializer.is_valid(raise_exception=True)
            except ValidationError as ve:
                return Response(ve.detail, status=status.HTTP_400_BAD_REQUEST)

            uploaded = serializer.save(
                user=(
                    request.user
                    if getattr(request.user, 'is_authenticated', False)
                    else None
                ),
                processing_status='processing',
            )

            success, payload = image_service.analyze_uploaded_image(uploaded)
            if not success:
                payload.update(
                    {
                        'success': False,
                        'image_id': str(uploaded.id),
                        'suggestions': [
                            'Make sure your face is clearly visible',
                            'Ensure good lighting',
                            'Face the camera directly',
                            'Remove sunglasses, hats, or face coverings',
                            'Try taking the photo from a different angle',
                        ],
                    }
                )
                return Response(payload, status=status.HTTP_400_BAD_REQUEST)

            payload.update(
                {'message': 'Image uploaded and analyzed successfully'}
            )
            return Response(payload)

        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            return Response(
                {"error": "Failed to upload image"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SetPreferencesView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]
    authentication_classes = []

    @extend_schema(
        request=UserPreferenceSerializer,
        responses={
            200: inline_serializer(
                name='PreferencesSaveResponse',
                fields={
                    'success': serializers.BooleanField(),
                    'preference_id': serializers.CharField(),
                    'message': serializers.CharField(),
                    'preferences': UserPreferenceSerializer(),
                },
            ),
            400: OpenApiResponse(description='Invalid preferences'),
        },
    )
    def post(self, request):
        try:
            data = request.data
            logger.info(f"Received preferences data: {data}")

            preference_data = {
                'hair_type': data.get('hair_type', ''),
                'hair_length': data.get('hair_length', ''),
                'lifestyle': data.get('lifestyle', ''),
                'maintenance': data.get('maintenance', ''),
                'occasions': data.get('occasions', []),
                'gender': data.get('gender', ''),
                'hair_color': data.get('hair_color', ''),
                'color_preference': data.get('color_preference', ''),
                'budget_range': data.get('budget_range', ''),
            }

            preference_data = {
                k: v for k, v in preference_data.items() if v != ''
            }
            if not isinstance(preference_data.get('occasions', []), list):
                preference_data['occasions'] = []

            required_fields = ['hair_type', 'hair_length', 'maintenance']
            missing_fields = [
                f for f in required_fields if not preference_data.get(f)
            ]
            if missing_fields:
                return Response(
                    {
                        "error": (
                            "Required fields missing: "
                            f"{', '.join(missing_fields)}"
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            valid_hair_types = ['straight', 'wavy', 'curly', 'coily']
            if preference_data['hair_type'] not in valid_hair_types:
                return Response(
                    {
                        "error": (
                            "Invalid hair_type '"
                            + preference_data['hair_type']
                            + "'. Must be one of: "
                            + str(valid_hair_types)
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            valid_hair_lengths = [
                'pixie', 'short', 'medium', 'long', 'extra_long'
            ]
            if preference_data['hair_length'] not in valid_hair_lengths:
                return Response(
                    {
                        "error": (
                            "Invalid hair_length '"
                            + preference_data['hair_length']
                            + "'. Must be one of: "
                            + str(valid_hair_lengths)
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            valid_maintenance = ['low', 'medium', 'high']
            if preference_data['maintenance'] not in valid_maintenance:
                return Response(
                    {
                        "error": (
                            "Invalid maintenance "
                            (
                                "'" + preference_data['maintenance'] + "'"
                                + ". Must be one of: " + str(valid_maintenance)
                            )
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if preference_data.get('gender'):
                valid_genders = ['male', 'female', 'nb', 'other']
                if preference_data['gender'] not in valid_genders:
                    return Response(
                        {
                            "error": (
                                "Invalid gender "
                                (
                                    "'" + preference_data['gender'] + "'"
                                    + ". Must be one of: " + str(valid_genders)
                                )
                            )
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

            if preference_data.get('lifestyle'):
                lifestyle_map = {'moderate': 'casual', 'relaxed': 'casual'}
                if preference_data['lifestyle'] in lifestyle_map:
                    preference_data['lifestyle'] = lifestyle_map[
                        preference_data['lifestyle']
                    ]
                from ..models import UserPreference

                valid_lifestyles = [
                    choice[0] for choice in UserPreference.LIFESTYLE_CHOICES
                ]
                if preference_data['lifestyle'] not in valid_lifestyles:
                    return Response(
                        {
                            "error": (
                                "Invalid lifestyle "
                                (
                                    "'" + preference_data['lifestyle'] + "'"
                                    + ". Must be one of: "
                                    + str(valid_lifestyles)
                                )
                            )
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

            if preference_data.get('occasions'):
                from ..models import UserPreference

                valid_occasions = [
                    choice[0] for choice in UserPreference.OCCASION_CHOICES
                ]
                invalid_occasions = [
                    occ
                    for occ in preference_data['occasions']
                    if occ not in valid_occasions
                ]
                if invalid_occasions:
                    return Response(
                        {
                            "error": (
                                (
                                    "Invalid occasions "
                                    + str(invalid_occasions)
                                    + ". Must be from: "
                                    + str(valid_occasions)
                                )
                            )
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

            serializer = UserPreferenceSerializer(data=preference_data)
            if not serializer.is_valid():
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
                    if getattr(request.user, 'is_authenticated', False)
                    else None
                )
            )

            return Response(
                {
                    'success': True,
                    'preference_id': str(preference.id),
                    'message': 'Preferences saved successfully',
                    'preferences': serializer.data,
                }
            )

        except Exception as e:
            logger.error(f"Error saving preferences: {str(e)}")
            return Response(
                {"error": "Failed to save preferences", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class RecommendView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [RecommendationThrottle]

    from drf_spectacular.utils import (
        extend_schema,
        OpenApiResponse,
        OpenApiExample,
    )
    from ..serializers import RecommendRequestSerializer

    @extend_schema(
        request=RecommendRequestSerializer,
        responses={
            200: OpenApiResponse(description='Recommendations generated'),
            400: OpenApiResponse(description='Bad request'),
            404: OpenApiResponse(description='Image or preferences not found'),
            500: OpenApiResponse(
                description='Server error generating recommendations'
            ),
        },
        examples=[
            OpenApiExample(
                'Generate recommendations',
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

            if not image_id or not pref_id:
                return Response(
                    {"error": "Both image_id and preference_id are required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

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

    from drf_spectacular.utils import (
        extend_schema,
        OpenApiResponse,
        OpenApiExample,
    )
    from ..serializers import (
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

            return Response(
                {"overlay_url": overlay_url, "overlay_type": overlay_type}
            )

        except ValidationError as e:
            return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)
        except ValueError as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Error creating overlay: {str(e)}")
            return Response(
                {"error": "Failed to create overlay", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
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

    @extend_schema(
        request=RecommendRequestSerializer,
        responses={
            200: OpenApiResponse(
                description='Overlay generated from recommendation'
            ),
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
            q_overlay = request.query_params.get('overlay')
            if q_overlay in ('basic', 'advanced'):
                overlay_type = q_overlay

            pipeline = RecommendationOverlayPipeline()
            result = pipeline.run(
                uploaded,
                prefs,
                overlay_type=overlay_type,
                user=(
                    request.user
                    if getattr(request.user, 'is_authenticated', False)
                    else None
                ),
            )
            if 'error' in result:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            return Response(result)
        except ValidationError as e:
            return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error("Auto overlay error: %s", str(e))
            return Response(
                {
                    "error": "Failed to generate auto overlay",
                    "details": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
