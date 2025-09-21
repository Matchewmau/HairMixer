import os
import uuid
import traceback

from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes

from ..serializers import FeedbackSerializer, AnalyticsEventSerializer
from .base import logger, analytics_service
from drf_spectacular.utils import extend_schema, OpenApiResponse
from drf_spectacular.types import OpenApiTypes


class FeedbackView(APIView):
    permission_classes = [AllowAny]

    @extend_schema(
        request=FeedbackSerializer,
        responses={
            200: OpenApiResponse(
                description='Feedback submitted successfully'
            ),
            400: OpenApiResponse(description='Validation error'),
        },
    )
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
                if analytics_service:
                    analytics_service.track_event(
                        user=(
                            request.user
                            if request.user.is_authenticated
                            else None
                        ),
                        event_type='feedback_submitted',
                        event_data={
                            'feedback_id': str(fb.id),
                            'liked': fb.liked,
                            'rating': fb.rating,
                            'has_note': bool(fb.note),
                        },
                        request=request,
                    )
                return Response(
                    {
                        "feedback_id": fb.id,
                        "message": "Feedback submitted successfully",
                    }
                )
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            return Response(
                {"error": "Failed to save feedback", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AnalyticsEventView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        request=AnalyticsEventSerializer,
        responses={200: OpenApiResponse(description='Event tracked')},
    )
    def post(self, request):
        try:
            serializer = AnalyticsEventSerializer(data=request.data)
            if serializer.is_valid():
                if analytics_service:
                    analytics_service.track_event(
                        user=(
                            request.user
                            if request.user.is_authenticated
                            else None
                        ),
                        event_type=serializer.validated_data['event_type'],
                        event_data=serializer.validated_data.get(
                            'event_data', {}
                        ),
                        session_id=serializer.validated_data.get(
                            'session_id', ''
                        ),
                        request=request,
                    )
                return Response({"status": "event_tracked"})
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Error tracking analytics event: {str(e)}")
            return Response(
                {"error": "Failed to track event"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@extend_schema(
    request={
        'multipart/form-data': {
            'type': 'object',
            'properties': {'image': {'type': 'string', 'format': 'binary'}},
        }
    },
    responses={200: OpenApiTypes.OBJECT},
)
@api_view(['POST'])
@permission_classes([AllowAny])
def debug_face_detection(request):
    try:
        from ..ml.face_analyzer import FacialFeatureAnalyzer

        analyzer = FacialFeatureAnalyzer()
        debug_info = {
            'detector_type': analyzer.detector_type,
            'mediapipe_available': analyzer.detector_type == 'mediapipe',
            'facenet_available': analyzer.detector_type == 'facenet',
            'face_detector_initialized': analyzer.face_detector is not None,
            'face_mesh_initialized': analyzer.face_mesh is not None,
            'device': str(analyzer.device),
        }
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            temp_path = f"/tmp/debug_face_{uuid.uuid4()}.jpg"
            with open(temp_path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            result, error = analyzer.detect_and_analyze_face(temp_path)
            debug_info['detection_result'] = {
                'success': result is not None,
                'error': error,
                'face_detected': (
                    result.get('face_detected', False) if result else False
                ),
                'detection_method': (
                    result.get('detection_method', 'none')
                    if result
                    else 'none'
                ),
                'confidence': result.get('confidence', 0) if result else 0,
            }
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        return Response({'debug_info': debug_info})
    except Exception as e:
        return Response({'error': str(e), 'traceback': traceback.format_exc()})


@extend_schema(
    request={
        'multipart/form-data': {
            'type': 'object',
            'properties': {'image': {'type': 'string', 'format': 'binary'}},
            'required': ['image'],
        }
    },
    responses={200: OpenApiTypes.OBJECT},
)
@api_view(['POST'])
@permission_classes([AllowAny])
def debug_resnet_features(request):
    try:
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'})

        from ..ml.face_analyzer import FacialFeatureAnalyzer

        image_file = request.FILES['image']
        temp_path = f"/tmp/debug_resnet_{uuid.uuid4()}.jpg"
        with open(temp_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        analyzer = FacialFeatureAnalyzer()

        debug_info = {
            'resnet_available': (
                hasattr(analyzer, 'feature_extractor')
                and (analyzer.feature_extractor is not None)
            ),
            'classifier_type': getattr(analyzer, 'shape_classifier', 'None'),
            'device': str(analyzer.device),
        }

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

        try:
            os.unlink(temp_path)
        except Exception:
            pass

        return Response({'debug_info': debug_info})
    except Exception as e:
        return Response(
            {'error': str(e), 'traceback': traceback.format_exc()}
        )
