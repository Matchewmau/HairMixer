import torch
import torchvision.transforms as transforms
import numpy as np
import logging
from PIL import Image
import time

# Import the MobileNetV3 classifier
try:
    from .mobilenet_classifier import mobile_net_loader
    MOBILENET_AVAILABLE = True
    logging.getLogger(__name__).debug("MobileNetV3 classifier available")
except ImportError as e:
    MOBILENET_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "MobileNetV3 classifier not available: %s", e
    )

# At the top of face_analyzer.py, add a module-level flag
_MEDIAPIPE_INITIALIZED = False
_FACENET_INITIALIZED = False

# Try importing optional packages with better error handling - ONLY ONCE
FACENET_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False

# Reduce absl (mediapipe/tflite) verbosity if available
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

if not _FACENET_INITIALIZED:
    try:
        from facenet_pytorch import MTCNN
        FACENET_AVAILABLE = True
        _FACENET_INITIALIZED = True
        logging.getLogger(__name__).debug("FaceNet PyTorch available")
    except ImportError as e:
        logging.getLogger(__name__).warning(
            "FaceNet PyTorch not available: %s", e
        )

if not _MEDIAPIPE_INITIALIZED:
    try:
        # Suppress verbose TF/absl logs before importing mediapipe
        import os as _os
        _os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        _os.environ.setdefault('GLOG_minloglevel', '2')
        import mediapipe as mp
        try:
            from absl import logging as _absl_logging
            _absl_logging.set_verbosity(_absl_logging.ERROR)
        except Exception:
            pass
        MEDIAPIPE_AVAILABLE = True
        _MEDIAPIPE_INITIALIZED = True
        logging.getLogger(__name__).debug("MediaPipe available")
    except ImportError as e:
        logging.getLogger(__name__).warning(
            "MediaPipe not available: %s", e
        )

logger = logging.getLogger(__name__)

# Import face shapes from model.py
try:
    from .model import FACE_SHAPES
except ImportError:
    # Fallback if model.py import fails
    FACE_SHAPES = {
        0: "oval", 1: "round", 2: "square",
        3: "heart", 4: "diamond", 5: "oblong"
    }


class FacialFeatureAnalyzer:
    """Advanced facial feature analyzer using multiple detection methods"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to prevent multiple MediaPipe initializations"""
        if cls._instance is None:
            cls._instance = super(FacialFeatureAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        logger.debug("Initializing FacialFeatureAnalyzer (singleton)...")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.face_detector = None
        self.face_mesh = None
        self.shape_classifier = None
        
        # Initialize face detector with better error handling
        try:
            if MEDIAPIPE_AVAILABLE and not hasattr(self, '_mp_initialized'):
                logger.debug("Initializing MediaPipe components...")
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_drawing = mp.solutions.drawing_utils
                
                # Use more relaxed confidence threshold
                self.face_detector = self.mp_face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=0.3
                )
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                self.detector_type = 'mediapipe'
                self._mp_initialized = True
                logger.debug("MediaPipe detector initialized")
                
            elif FACENET_AVAILABLE and not hasattr(self, '_fn_initialized'):
                logger.debug("Initializing FaceNet...")
                self.face_detector = MTCNN(
                    keep_all=True,
                    device=self.device,
                    min_face_size=40,
                    thresholds=[0.6, 0.7, 0.7]
                )
                self.detector_type = 'facenet'
                self._fn_initialized = True
                logger.debug("FaceNet detector initialized successfully")
                
            else:
                logger.warning("No advanced face detection library available")
                self.detector_type = 'none'
                
        except Exception as e:
            logger.error(f"Error initializing face detectors: {str(e)}")
            self.detector_type = 'none'
        
        # Initialize transforms for MobileNet/Imagenet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load shape classifier if available
        try:
            logger.debug("Attempting to load shape classifier...")
            self._load_shape_classifier()
            logger.debug("Shape classifier loading completed")
        except Exception as e:
            logger.warning(f"Could not load shape classifier: {str(e)}")
            import traceback
            logger.warning(
                "Shape classifier traceback:\n%s",
                traceback.format_exc(),
            )
            
        FacialFeatureAnalyzer._initialized = True
        logger.debug(
            "FacialFeatureAnalyzer detector: %s",
            self.detector_type,
        )

    def detect_and_analyze_face(self, image_path):
        """Main face detection and analysis with detailed logging"""
        start = time.perf_counter()
        logger.info("face_analysis_start img=%s", image_path)
        try:
            # Load image with PIL to avoid OpenCV dependency
            with Image.open(str(image_path)) as pil_img:
                img = np.array(pil_img.convert('RGB'))
            if img is None:
                logger.error(f"Could not load image file: {image_path}")
                return None, "Could not load image file"
            logger.debug(
                "Loaded image: %s; detector: %s",
                img.shape,
                self.detector_type,
            )

            img_rgb = img

            # Try MediaPipe
            if MEDIAPIPE_AVAILABLE and self.face_detector:
                logger.debug("Trying MediaPipe face detection...")
                result, error = self._detect_with_mediapipe_simple(img_rgb)
                if result:
                    logger.debug("MediaPipe detection successful")
                    return result, None
                logger.warning(f"MediaPipe detection failed: {error}")

            # Try FaceNet
            if FACENET_AVAILABLE and self.face_detector:
                logger.debug("Trying FaceNet face detection...")
                result, error = self._detect_with_facenet_simple(img_rgb)
                if result:
                    logger.debug("FaceNet detection successful")
                    return result, None
                logger.warning(f"FaceNet detection failed: {error}")

            # No OpenCV fallback; return not detected
            logger.debug("No OpenCV fallback configured")

            # (Optional) Try Dlib/face_recognition as last resort
            # result, error = self._detect_with_dlib(img_rgb)
            # if result:
            #     logger.info("Dlib detection successful")
            #     return result, None

            return None, "No face detected by any method"

        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, f"Detection error: {str(e)}"
        # After computing result and before return:
        elapsed = time.perf_counter() - start
        if result and result.get('face_detected'):
            logger.info(
                "face_analysis_ok method=%s conf=%.3f shape=%s "
                "shape_conf=%.3f elapsed=%.3fs",
                result.get('detection_method'),
                float(result.get('confidence', 0.0)),
                (result.get('face_shape') or {}).get('shape', 'n/a'),
                float((result.get('face_shape') or {}).get('confidence', 0.0)),
                elapsed,
            )
        else:
            logger.info(
                "face_analysis_no_face method=%s elapsed=%.3fs err=%s",
                (result or {}).get('detection_method', 'n/a'),
                elapsed,
                error or '',
            )
        return result, error

    def _detect_with_mediapipe_simple(self, img_rgb):
        """Simplified MediaPipe detection"""
        try:
            logger.debug("MediaPipe - input image shape: %s", img_rgb.shape)
            logger.debug("MediaPipe - input image dtype: %s", img_rgb.dtype)
            logger.debug(
                "MediaPipe - input image range: %s-%s",
                img_rgb.min(),
                img_rgb.max(),
            )
            
            results = self.face_detector.process(img_rgb)
            
            if not results.detections:
                logger.debug("MediaPipe - no detections found")
                return None, "No face detected by MediaPipe"
            
            logger.debug(
                "MediaPipe - found %s detections",
                len(results.detections),
            )
            
            # Get the most confident detection
            best_detection = max(results.detections, key=lambda d: d.score[0])
            confidence = float(best_detection.score[0])
            
            logger.debug(
                "MediaPipe - best detection confidence: %s",
                confidence,
            )
            
            if confidence < 0.3:
                return None, f"Detection confidence too low: {confidence:.3f}"
            
            # Extract face region
            bbox = best_detection.location_data.relative_bounding_box
            h, w, _ = img_rgb.shape
            
            logger.debug(
                "MediaPipe - bbox: x=%.3f y=%.3f w=%.3f h=%.3f",
                bbox.xmin,
                bbox.ymin,
                bbox.width,
                bbox.height,
            )
            
            # Calculate coordinates with padding
            padding = 0.1
            x = max(0, int((bbox.xmin - padding * bbox.width) * w))
            y = max(0, int((bbox.ymin - padding * bbox.height) * h))
            width = min(w - x, int((1 + 2 * padding) * bbox.width * w))
            height = min(h - y, int((1 + 2 * padding) * bbox.height * h))
            
            logger.debug(
                "MediaPipe - calculated face region: x=%s, y=%s, w=%s, h=%s",
                x, y, width, height,
            )
            
            if width <= 0 or height <= 0 or width < 50 or height < 50:
                logger.debug(
                    "MediaPipe - invalid face region: %sx%s",
                    width,
                    height,
                )
                return None, f"Invalid face region: {width}x{height}"
            
            # Extract face
            face_img = img_rgb[y:y+height, x:x+width]
            
            if face_img.size == 0:
                logger.debug("MediaPipe - empty face region")
                return None, "Empty face region after extraction"
            
            logger.debug(
                "MediaPipe - extracted face shape: %s",
                face_img.shape,
            )
            
            # Analyze the detected face
            logger.debug("MediaPipe - calling face shape prediction...")
            face_shape = self._predict_face_shape_simple(face_img)
            logger.debug("MediaPipe - face shape result: %s", face_shape)
            
            # Use inline quality metrics to bypass singleton issues
            logger.debug("MediaPipe - calculating quality metrics inline...")
            try:
                # Simple inline quality calculation (numpy only)
                gray = np.dot(
                    face_img[..., :3],
                    [0.299, 0.587, 0.114],
                ).astype(np.float32)
                brightness = float(np.mean(gray))
                contrast = float(np.std(gray))
                gy, gx = np.gradient(gray)
                sharpness = float((gx**2 + gy**2).mean())
                quality_score = min(
                    brightness / 255.0 * 0.4
                    + contrast / 100.0 * 0.3
                    + min(sharpness / 100.0, 1.0) * 0.3,
                    1.0,
                )

                quality_metrics = {
                    'overall_quality': quality_score,
                    'brightness_score': brightness / 255.0,
                    'contrast_score': min(contrast / 50.0, 1.0),
                    'sharpness_score': min(sharpness / 100.0, 1.0),
                    'is_good_quality': quality_score > 0.5
                }
                logger.debug(
                    "MediaPipe - inline quality metrics: %s",
                    quality_metrics,
                )
            except Exception as e:
                logger.debug("MediaPipe - inline quality failed: %s", str(e))
                quality_metrics = {
                    'overall_quality': 0.7,
                    'brightness_score': 0.7,
                    'contrast_score': 0.7,
                    'sharpness_score': 0.7,
                    'is_good_quality': True
                }
            
            result = {
                'face_detected': True,
                'confidence': confidence,
                'face_shape': face_shape,
                'facial_features': {
                    'face_width': width,
                    'face_height': height,
                },
                'quality_metrics': quality_metrics,
                'landmarks': [],
                'face_box': [x, y, x+width, y+height],
                'detection_method': 'mediapipe'
            }
            
            logger.debug("MediaPipe - final result: %s", result)
            return result, None
            
        except Exception as e:
            logger.error(f"MediaPipe detection error: {str(e)}")
            return None, f"MediaPipe error: {str(e)}"

    # Removed OpenCV Haar fallback

    def _load_shape_classifier(self):
        """Load MobileNetV3 face shape classifier only (no other fallbacks)."""
        try:
            if not MOBILENET_AVAILABLE:
                logger.error("MobileNetV3 classifier module not available")
                self.feature_extractor = None
                self.shape_classifier = None
                return False

            logger.debug(
                "Attempting to load MobileNetV3 face shape classifier..."
            )
            if mobile_net_loader.load_model():
                self.shape_classifier = 'mobilenet_v3'
                self.feature_extractor = mobile_net_loader
                logger.info("model_load_ok model=MobileNetV3 weights=%s", str(self.model_path))
                logger.debug("MobileNetV3 classifier loaded successfully")
                return True

            logger.error("Failed to load MobileNetV3 model")
            self.feature_extractor = None
            self.shape_classifier = None
            return False

        except Exception as e:
            logger.error(f"Error loading MobileNetV3 classifier: {str(e)}")
            self.feature_extractor = None
            self.shape_classifier = None
            return False
    
    # Removed simple CNN and ResNet fallbacks

    def _predict_face_shape_simple(self, face_img):
        """Face shape prediction using MobileNetV3 only.
        (No model fallbacks)
        """
        try:
            if not (
                MOBILENET_AVAILABLE
                and getattr(self, 'shape_classifier', None) == 'mobilenet_v3'
                and getattr(self, 'feature_extractor', None)
            ):
                logger.error(
                    "MobileNetV3 classifier unavailable; returning unavailable"
                )
                return {
                    'shape': 'oval',
                    'confidence': 0.0,
                    'method': 'unavailable',
                }

            mobilenet_result = self._predict_with_mobilenet(face_img)
            logger.debug(
                "MobileNetV3 prediction: %s (conf: %.3f)",
                mobilenet_result['shape'],
                mobilenet_result['confidence'],
            )
            return mobilenet_result

        except Exception as e:
            logger.error("Face shape prediction error: %s", str(e))
            return {'shape': 'oval', 'confidence': 0.0, 'method': 'error'}

    def _predict_with_mobilenet(self, face_img):
        """Predict face shape using trained MobileNetV3 model"""
        t0 = time.perf_counter()
        try:
            # Prepare image for MobileNetV3 (same preprocessing as training)
            face_pil = Image.fromarray(face_img)
            
            # Apply standard ImageNet preprocessing
            # (adjust if training used different preprocessing)
            face_tensor = self.transform(face_pil).unsqueeze(0)
            
            # Get prediction from MobileNetV3
            prediction_result = self.feature_extractor.predict(face_tensor)
            
            if prediction_result is None:
                raise ValueError("MobileNetV3 prediction returned None")
            
            # Map class index to face shape name
            predicted_class = prediction_result['predicted_class']
            confidence = prediction_result['confidence']
            probabilities = prediction_result['probabilities']
            
            # Get face shape name from FACE_SHAPES mapping
            from .model import FACE_SHAPES
            face_shape = FACE_SHAPES.get(predicted_class, 'oval')
            
            # Create probability dictionary for all classes
            all_probabilities = {}
            for i, prob in enumerate(probabilities):
                shape_name = FACE_SHAPES.get(i, f'class_{i}')
                all_probabilities[shape_name] = float(prob)
            
            dt = (time.perf_counter() - t0) * 1000.0
            try:
                pred_shape = out['shape']
                pred_conf = float(out.get('confidence', 0.0))
            except Exception:
                pred_shape = 'n/a'
                pred_conf = 0.0
            logger.info(
                "model_predict_ok model=MobileNetV3 shape=%s conf=%.3f "
                "latency_ms=%.1f",
                pred_shape,
                pred_conf,
                dt,
            )
            
            return {
                'shape': face_shape,
                'confidence': confidence,
                'method': 'mobilenet_v3',
                'all_probabilities': all_probabilities,
                'raw_prediction': prediction_result
            }
            
        except Exception as e:
            logger.error(f"MobileNetV3 prediction error: {str(e)}")
            raise

    # Removed ResNet/geometric analysis helpers (MobileNet-only)

    def _detect_with_facenet_simple(self, img_rgb):
        """FaceNet (MTCNN) detection fallback"""
        try:
            boxes, probs = self.face_detector.detect(img_rgb)
            if boxes is None or len(boxes) == 0:
                return None, "No face detected by FaceNet"
            # Use the most confident face
            idx = np.argmax(probs)
            x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
            face_img = img_rgb[y1:y2, x1:x2]
            if face_img.size == 0:
                return None, "Empty face region"
            face_shape = self._predict_face_shape_simple(face_img)
            quality_metrics = self._calculate_quality_metrics_simple(face_img)
            return {
                'face_detected': True,
                'confidence': float(probs[idx]),
                'face_shape': face_shape,
                'facial_features': {'face_width': x2-x1, 'face_height': y2-y1},
                'quality_metrics': quality_metrics,
                'landmarks': [],
                'face_box': [x1, y1, x2, y2],
                'detection_method': 'facenet'
            }, None
        except Exception as e:
            logger.error(f"FaceNet detection error: {str(e)}")
            return None, f"FaceNet error: {str(e)}"

    def _calculate_quality_metrics_simple(self, face_img):
        """Simple quality metrics calculation for face images"""
        try:
            logger.debug("Quality metrics - starting calculation")
            
            # Basic quality assessments
            h, w = face_img.shape[:2]
            logger.debug("Quality metrics - image dimensions: %sx%s", w, h)
            
            # Size quality
            min_size = 100
            size_score = min(1.0, min(w, h) / min_size)
            logger.debug("Quality metrics - size score: %s", size_score)
            
            # Brightness quality
            # grayscale via numpy formula
            gray = np.dot(
                face_img[..., :3],
                [0.299, 0.587, 0.114],
            ).astype(np.float32)
            mean_brightness = np.mean(gray)
            # Optimal brightness around 128
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            brightness_score = max(0.0, min(1.0, brightness_score))
            logger.debug(
                "Quality metrics - brightness score: %s",
                brightness_score,
            )
            
            # Contrast quality
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 50)  # Good contrast above 50
            logger.debug(
                "Quality metrics - contrast score: %s",
                contrast_score,
            )
            
            # Sharpness (simplified Laplacian variance)
            # approximate sharpness via gradient variance (Sobel-like)
            gy, gx = np.gradient(gray)
            lap_var = (gx**2 + gy**2).mean()
            sharpness = float(lap_var)
            # Good sharpness above 100
            sharpness_score = min(1.0, sharpness / 100)
            logger.debug(
                "Quality metrics - sharpness score: %s",
                sharpness_score,
            )
            
            # Overall quality (weighted average)
            overall_quality = (
                size_score * 0.3
                + brightness_score * 0.25
                + contrast_score * 0.25
                + sharpness_score * 0.2
            )
            
            quality_metrics = {
                'overall_quality': round(overall_quality, 3),
                'is_good_quality': overall_quality >= 0.6,
                'size_score': round(size_score, 3),
                'brightness_score': round(brightness_score, 3),
                'contrast_score': round(contrast_score, 3),
                'sharpness_score': round(sharpness_score, 3),
                'dimensions': f"{w}x{h}"
            }
            
            logger.debug("Quality metrics - final result: %s", quality_metrics)
            return quality_metrics
            
        except Exception as e:
            logger.debug("Quality metrics error: %s", str(e))
            return {
                'overall_quality': 0.5,
                'is_good_quality': True,
                'error': str(e)
            }


# Removed debug_resnet_features (MobileNet is the only classifier).


def analyze_face_comprehensive(image_path):
    """
    Comprehensive face analysis that returns all needed data.
    This is the main function called by views.py
    """
    try:
        logger.debug(
            "Starting comprehensive face analysis for: %s",
            image_path,
        )
        
        # Initialize analyzer
        analyzer = FacialFeatureAnalyzer()
        logger.debug(
            "Analyzer detector: %s",
            analyzer.detector_type,
        )
        
        # Perform face detection and analysis
        result, error = analyzer.detect_and_analyze_face(image_path)
        
        if result is None:
            logger.warning(f"Face analysis failed: {error}")
            # Return structured failure response
            return {
                'face_detected': False,
                'error': error or 'No face detected',
                'face_shape': {'shape': 'oval', 'confidence': 0.0},
                'quality_metrics': {
                    'overall_quality': 0.0,
                    'is_good_quality': False,
                },
                'facial_features': {},
                'confidence': 0.0,
                'detection_method': 'failed',
            }
        
        # Ensure all required fields are present
        face_shape = result.get(
            'face_shape',
            {'shape': 'oval', 'confidence': 0.6},
        )
        if not isinstance(face_shape, dict):
            face_shape = {'shape': str(face_shape), 'confidence': 0.6}
            
        quality_metrics = result.get(
            'quality_metrics',
            {'overall_quality': 0.7, 'is_good_quality': True},
        )
        
        response_data = {
            'face_detected': result.get('face_detected', True),
            'face_shape': face_shape,
            'facial_features': result.get('facial_features', {}),
            'confidence': result.get(
                'confidence',
                face_shape.get('confidence', 0.6),
            ),
            'face_box': result.get('face_box', []),
            'landmarks': result.get('landmarks', []),
            'quality_metrics': quality_metrics,
            'detection_method': result.get('detection_method', 'unknown')
        }
        
        logger.debug(
            "Face analysis completed: shape=%s, conf=%.3f, method=%s",
            face_shape['shape'],
            face_shape['confidence'],
            response_data['detection_method'],
        )
        
        return response_data
        
    except Exception as e:
        logger.exception("Error in comprehensive face analysis")
        
        return {
            'face_detected': False,
            'error': f"Analysis error: {str(e)}",
            'face_shape': {'shape': 'oval', 'confidence': 0.0},
            'quality_metrics': {
                'overall_quality': 0.0,
                'is_good_quality': False,
            },
            'facial_features': {},
            'confidence': 0.0,
            'detection_method': 'error',
        }
