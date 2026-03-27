from django.shortcuts import get_object_or_404
from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.throttling import UserRateThrottle
from rest_framework.exceptions import ValidationError
from drf_spectacular.utils import (
    extend_schema,
    OpenApiResponse,
    OpenApiExample,
)
import logging
from pathlib import Path

from ..models import UploadedImage, UserPreference, Hairstyle
from ..serializers import (
    UploadedImageSerializer, OverlayRequestSerializer, OverlayResponseSerializer,
    RecommendRequestSerializer
)
from ..services.image_service import ImageService
from ..services.overlay_service import OverlayService
from ..services.pipeline_service import RecommendationOverlayPipeline
from ..services.analytics_utils import track_event_safe
from ..services.analytics import AnalyticsService

# Initialize services
try:
    analytics_service = AnalyticsService()
except ImportError:
    analytics_service = None

overlay_service = OverlayService()
pipeline_service = RecommendationOverlayPipeline()

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from ..ml.preprocess import (
        read_image,
        detect_face,
        validate_image_quality,
    )
    ML_PREPROCESS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML preprocessing not available: {e}")
    ML_PREPROCESS_AVAILABLE = False

try:
    from ..overlay import AdvancedOverlayProcessor
    OVERLAY_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Overlay processor not available: {e}")
    OVERLAY_PROCESSOR_AVAILABLE = False


class ImageUploadThrottle(UserRateThrottle):
    scope = 'uploads'

class UploadImageView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [ImageUploadThrottle]
    
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
            200: OpenApiResponse(description='Image uploaded and analyzed successfully'),
            400: OpenApiResponse(description='Validation failed or no face detected'),
            500: OpenApiResponse(description='Server error uploading image'),
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
            
            if not image_file.content_type.startswith('image/'):
                return Response(
                    {"error": "Invalid file type"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validate extension
            import os
            ext = os.path.splitext(image_file.name)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
                return Response(
                    {"error": "Unsupported file extension. Use JPG, PNG, or WEBP."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if image_file.size > 10 * 1024 * 1024:  # 10MB limit
                return Response(
                    {"error": "File too large"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            serializer = UploadedImageSerializer(data={"image": image_file})
            try:
                serializer.is_valid(raise_exception=True)
            except ValidationError as ve:
                return Response(ve.detail, status=status.HTTP_400_BAD_REQUEST)
            
            uploaded = serializer.save(
                user=(
                    request.user
                    if request.user and request.user.is_authenticated
                    else None
                ),
                processing_status='processing',
            )
            
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
            
            quality_check = validate_image_quality(img)
            if not quality_check['is_valid']:
                image_instance.processing_status = 'failed'
                image_instance.error_message = quality_check['error']
                image_instance.save()
                return
            
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
        descriptions = {
            'oval': 'Balanced proportions - most hairstyles suit you!',
            'round': 'Soft, curved features - try angular cuts and long layers',
            'square': 'Strong jawline - soft waves and layers work great',
            'heart': 'Wider forehead - styles with volume at the jaw are perfect',
            'diamond': 'Prominent cheekbones - textured styles balance your features',
            'oblong': 'Longer face - styles with width and volume are ideal',
        }
        return descriptions.get(
            face_shape, 'Unique face shape with many styling options!'
        )


class OverlayView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [IsAuthenticated]

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
            use_hair_color = req_ser.validated_data.get("use_hair_color", False)
            
            if not image_id or not style_id:
                return Response(
                    {"error": "Both image_id and hairstyle_id are required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            
            uploaded = get_object_or_404(UploadedImage, id=image_id)
            style = get_object_or_404(Hairstyle, id=style_id)

            hair_color = None
            hair_type = None
            hair_length = None
            if use_hair_color and request.user.is_authenticated:
                try:
                    preference = UserPreference.objects.filter(
                        user=request.user
                    ).order_by('-updated_at').first()
                    
                    if preference:
                        hair_color = preference.hair_color if preference.hair_color else None
                        if hair_color == 'other' and preference.color_preference:
                            hair_color = preference.color_preference.lower()
                        hair_type = preference.hair_type if preference.hair_type else None
                        hair_length = preference.hair_length if preference.hair_length else None
                    else:
                        logger.warning("No preferences found for user")
                except Exception as e:
                    logger.error(f"Error fetching preferences: {e}")
            else:
                logger.info("Not using hair attributes (Discover page or unauthenticated)")
            
            overlay_url = overlay_service.generate(
                uploaded, 
                style, 
                overlay_type, 
                hair_color=hair_color,
                hair_type=hair_type,
                hair_length=hair_length
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
            
            return Response({
                "overlay_url": overlay_url,
                "overlay_type": overlay_type
            })
            
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
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AutoOverlayView(APIView):
    parser_classes = (JSONParser,)
    permission_classes = [AllowAny]
    authentication_classes = []

    @extend_schema(
        request=RecommendRequestSerializer,
        responses={
            200: OpenApiResponse(description='Overlay generated from recommendation'),
            400: OpenApiResponse(description='Bad request'),
            404: OpenApiResponse(description='Not found'),
            500: OpenApiResponse(description='Server error'),
        },
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
