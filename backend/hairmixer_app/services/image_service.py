import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from django.conf import settings
from ..models import UploadedImage

logger = logging.getLogger(__name__)

try:
    from ..ml.preprocess import read_image, validate_image_quality
    from ..ml.face_analyzer import analyze_face_comprehensive
    ML_AVAILABLE = True
except Exception as e:
    logger.warning(f"Image ML components unavailable: {e}")
    ML_AVAILABLE = False


class ImageService:
    """Handle image upload metadata extraction and analysis."""

    @staticmethod
    def analyze_uploaded_image(uploaded: UploadedImage) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze an UploadedImage for face and quality.
        Returns (success, payload) where payload contains response fields.
        """
        try:
            img_path = Path(settings.MEDIA_ROOT) / uploaded.image.name

            if not ML_AVAILABLE:
                uploaded.processing_status = 'failed'
                uploaded.error_message = 'ML components unavailable'
                uploaded.save(update_fields=['processing_status', 'error_message'])
                return False, {"error": "ML components unavailable"}

            img_array = read_image(img_path)
            if img_array is None:
                uploaded.processing_status = 'failed'
                uploaded.error_message = 'Could not read image file'
                uploaded.save(update_fields=['processing_status', 'error_message'])
                return False, {"error": "Could not process image file"}

            quality_check = validate_image_quality(img_array)
            face_analysis = analyze_face_comprehensive(img_path)
            face_detected = face_analysis.get('face_detected', False)

            if not face_detected:
                err = face_analysis.get('error', 'No face detected')
                uploaded.processing_status = 'no_face'
                uploaded.face_detected = False
                uploaded.error_message = err
                uploaded.save(update_fields=['processing_status', 'face_detected', 'error_message'])

                return False, {
                    'face_detected': False,
                    'error': 'Could not detect a face in this image',
                    'detailed_error': err,
                    'processing_status': 'no_face',
                    'quality_check': quality_check
                }

            # success path
            uploaded.processing_status = 'completed'
            uploaded.face_detected = True
            # Optionally count faces if provided
            face_count = face_analysis.get('face_count') or 1
            uploaded.face_count = face_count if isinstance(face_count, int) else 1
            uploaded.save(update_fields=['processing_status', 'face_detected', 'face_count'])

            payload = {
                'success': True,
                'image_id': str(uploaded.id),
                'face_detected': True,
                'face_shape': face_analysis.get('face_shape', {}),
                'confidence': face_analysis.get('confidence', face_analysis.get('face_shape', {}).get('confidence', 0)),
                'quality_score': quality_check.get('quality_score', 0),
                'quality_metrics': face_analysis.get('quality_metrics', {}),
                'is_good_quality': quality_check.get('is_valid', False),
                'processing_status': 'completed',
                'facial_features': face_analysis.get('facial_features', {}),
                'detection_method': face_analysis.get('detection_method', 'unknown'),
            }
            return True, payload

        except Exception as e:
            logger.exception("Error analyzing uploaded image")
            uploaded.processing_status = 'failed'
            uploaded.error_message = str(e)
            uploaded.save(update_fields=['processing_status', 'error_message'])
            return False, {"error": "Failed to process image", "detailed_error": str(e)}
