import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import logging
from PIL import Image
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import uuid

# Import the MobileNetV3 classifier
try:
    from .mobilenet_classifier import mobile_net_loader
    MOBILENET_AVAILABLE = True
    print("✅ MobileNetV3 classifier available")
except ImportError as e:
    MOBILENET_AVAILABLE = False
    print(f"⚠️ MobileNetV3 classifier not available: {e}")

# At the top of face_analyzer.py, add a module-level flag
_MEDIAPIPE_INITIALIZED = False
_FACENET_INITIALIZED = False

# Try importing optional packages with better error handling - ONLY ONCE
FACENET_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False

if not _FACENET_INITIALIZED:
    try:
        from facenet_pytorch import MTCNN
        FACENET_AVAILABLE = True
        _FACENET_INITIALIZED = True
        print("✅ FaceNet PyTorch available")
    except ImportError as e:
        print(f"⚠️  FaceNet PyTorch not available: {e}")

if not _MEDIAPIPE_INITIALIZED:
    try:
        import mediapipe as mp
        MEDIAPIPE_AVAILABLE = True
        _MEDIAPIPE_INITIALIZED = True
        print("✅ MediaPipe available")
    except ImportError as e:
        print(f"⚠️  MediaPipe not available: {e}")

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
            
        logger.info("Initializing FacialFeatureAnalyzer (singleton)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_detector = None
        self.face_mesh = None
        self.shape_classifier = None
        
        # Initialize face detector with better error handling
        try:
            if MEDIAPIPE_AVAILABLE and not hasattr(self, '_mp_initialized'):
                logger.info("Initializing MediaPipe components...")
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_drawing = mp.solutions.drawing_utils
                
                # Use more relaxed confidence threshold
                self.face_detector = self.mp_face_detection.FaceDetection(
                    model_selection=0,  # 0 for close-range, 1 for full-range
                    min_detection_confidence=0.3  # Lower threshold for better detection
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
                logger.info("MediaPipe face detector initialized successfully")
                
            elif FACENET_AVAILABLE and not hasattr(self, '_fn_initialized'):
                logger.info("Initializing FaceNet...")
                self.face_detector = MTCNN(
                    keep_all=True, 
                    device=self.device,
                    min_face_size=40,
                    thresholds=[0.6, 0.7, 0.7]
                )
                self.detector_type = 'facenet'
                self._fn_initialized = True
                logger.info("FaceNet detector initialized successfully")
                
            else:
                logger.warning("No advanced face detection library available, using OpenCV")
                self.detector_type = 'opencv'
                
        except Exception as e:
            logger.error(f"Error initializing face detectors: {str(e)}")
            self.detector_type = 'opencv'  # Fallback to OpenCV
        
        # Initialize transforms for MobileNet/Imagenet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load shape classifier if available
        try:
            logger.info("Attempting to load shape classifier...")
            self._load_shape_classifier()
            logger.info("Shape classifier loading completed")
        except Exception as e:
            logger.warning(f"Could not load shape classifier: {str(e)}")
            import traceback
            logger.warning(f"Shape classifier error traceback: {traceback.format_exc()}")
            
        FacialFeatureAnalyzer._initialized = True
        logger.info(f"FacialFeatureAnalyzer initialized with detector: {self.detector_type}")

    def detect_and_analyze_face(self, image_path):
        """Main face detection and analysis method with multiple fallbacks and better logging"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Could not load image file: {image_path}")
                return None, "Could not load image file"
            logger.info(f"Loaded image: {img.shape}, using detector: {self.detector_type}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Try MediaPipe
            if MEDIAPIPE_AVAILABLE and self.face_detector:
                logger.info("Trying MediaPipe face detection...")
                result, error = self._detect_with_mediapipe_simple(img_rgb)
                if result:
                    logger.info("MediaPipe detection successful")
                    return result, None
                logger.warning(f"MediaPipe detection failed: {error}")

            # Try FaceNet
            if FACENET_AVAILABLE and self.face_detector:
                logger.info("Trying FaceNet face detection...")
                result, error = self._detect_with_facenet_simple(img_rgb)
                if result:
                    logger.info("FaceNet detection successful")
                    return result, None
                logger.warning(f"FaceNet detection failed: {error}")

            # Try OpenCV
            logger.info("Trying OpenCV fallback detection...")
            result, error = self._detect_with_opencv_simple(img, img_rgb)
            if result:
                logger.info("OpenCV detection successful")
                return result, None
            logger.warning(f"OpenCV detection failed: {error}")

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

    def _detect_with_mediapipe_simple(self, img_rgb):
        """Simplified MediaPipe detection"""
        try:
            print(f"DEBUG: MediaPipe - input image shape: {img_rgb.shape}")
            print(f"DEBUG: MediaPipe - input image dtype: {img_rgb.dtype}")
            print(f"DEBUG: MediaPipe - input image range: {img_rgb.min()}-{img_rgb.max()}")
            
            results = self.face_detector.process(img_rgb)
            
            if not results.detections:
                print("DEBUG: MediaPipe - no detections found")
                return None, "No face detected by MediaPipe"
            
            print(f"DEBUG: MediaPipe - found {len(results.detections)} detections")
            
            # Get the most confident detection
            best_detection = max(results.detections, key=lambda d: d.score[0])
            confidence = float(best_detection.score[0])
            
            print(f"DEBUG: MediaPipe - best detection confidence: {confidence}")
            
            if confidence < 0.3:
                return None, f"Detection confidence too low: {confidence:.3f}"
            
            # Extract face region
            bbox = best_detection.location_data.relative_bounding_box
            h, w, _ = img_rgb.shape
            
            print(f"DEBUG: MediaPipe - bbox: xmin={bbox.xmin:.3f}, ymin={bbox.ymin:.3f}, "
                  f"width={bbox.width:.3f}, height={bbox.height:.3f}")
            
            # Calculate coordinates with padding
            padding = 0.1
            x = max(0, int((bbox.xmin - padding * bbox.width) * w))
            y = max(0, int((bbox.ymin - padding * bbox.height) * h))
            width = min(w - x, int((1 + 2 * padding) * bbox.width * w))
            height = min(h - y, int((1 + 2 * padding) * bbox.height * h))
            
            print(f"DEBUG: MediaPipe - calculated face region: "
                  f"x={x}, y={y}, w={width}, h={height}")
            
            if width <= 0 or height <= 0 or width < 50 or height < 50:
                print(f"DEBUG: MediaPipe - invalid face region: {width}x{height}")
                return None, f"Invalid face region: {width}x{height}"
            
            # Extract face
            face_img = img_rgb[y:y+height, x:x+width]
            
            if face_img.size == 0:
                print("DEBUG: MediaPipe - empty face region")
                return None, "Empty face region after extraction"
            
            print(f"DEBUG: MediaPipe - extracted face shape: {face_img.shape}")
            
            # Analyze the detected face
            print("DEBUG: MediaPipe - calling face shape prediction...")
            face_shape = self._predict_face_shape_simple(face_img)
            print(f"DEBUG: MediaPipe - face shape result: {face_shape}")
            
            # Use inline quality metrics to bypass singleton issues
            print("DEBUG: MediaPipe - calculating quality metrics inline...")
            try:
                # Simple inline quality calculation
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                brightness = float(np.mean(gray))
                contrast = float(np.std(gray))
                quality_score = min(brightness / 255.0 + contrast / 100.0, 1.0)
                
                quality_metrics = {
                    'overall_quality': quality_score,
                    'brightness_score': brightness / 255.0,
                    'contrast_score': min(contrast / 50.0, 1.0),
                    'is_good_quality': quality_score > 0.5
                }
                print(f"DEBUG: MediaPipe - inline quality metrics: {quality_metrics}")
            except Exception as e:
                print(f"DEBUG: MediaPipe - inline quality failed: {str(e)}")
                quality_metrics = {
                    'overall_quality': 0.7, 
                    'brightness_score': 0.7, 
                    'contrast_score': 0.7,
                    'is_good_quality': True
                }
            
            result = {
                'face_detected': True,
                'confidence': confidence,
                'face_shape': face_shape,
                'facial_features': {'face_width': width, 'face_height': height},
                'quality_metrics': quality_metrics,
                'landmarks': [],
                'face_box': [x, y, x+width, y+height],
                'detection_method': 'mediapipe'
            }
            
            print(f"DEBUG: MediaPipe - final result: {result}")
            return result, None
            
        except Exception as e:
            logger.error(f"MediaPipe detection error: {str(e)}")
            return None, f"MediaPipe error: {str(e)}"

    def _detect_with_opencv_simple(self, img_bgr, img_rgb):
        """Simplified OpenCV detection"""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if face_cascade.empty():
                return None, "Could not load OpenCV face cascade"
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None, "No face detected by OpenCV"
            
            # Use the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Add padding
            padding = int(0.1 * min(w, h))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_rgb.shape[1] - x, w + 2*padding)
            h = min(img_rgb.shape[0] - y, h + 2*padding)
            
            face_img = img_rgb[y:y+h, x:x+w]
            
            if face_img.size == 0:
                return None, "Empty face region"
            
            # Analyze the detected face
            face_shape = self._predict_face_shape_simple(face_img)
            quality_metrics = self._calculate_quality_metrics_simple(face_img)
            
            return {
                'face_detected': True,
                'confidence': 0.7,
                'face_shape': face_shape,
                'facial_features': {'face_width': w, 'face_height': h},
                'quality_metrics': quality_metrics,
                'landmarks': [],
                'face_box': [x, y, x+w, y+h],
                'detection_method': 'opencv'
            }, None
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {str(e)}")
            return None, f"OpenCV error: {str(e)}"

    def _load_shape_classifier(self):
        """Load MobileNetV3 face shape classifier only (no other fallbacks)."""
        try:
            if not MOBILENET_AVAILABLE:
                logger.error("MobileNetV3 classifier module not available")
                self.feature_extractor = None
                self.shape_classifier = None
                return False

            logger.info("Attempting to load MobileNetV3 face shape classifier...")
            if mobile_net_loader.load_model():
                self.shape_classifier = 'mobilenet_v3'
                self.feature_extractor = mobile_net_loader
                logger.info("MobileNetV3 face shape classifier loaded successfully")
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
        """Face shape prediction using MobileNetV3 only (no model fallbacks)."""
        try:
            if not (MOBILENET_AVAILABLE and getattr(self, 'shape_classifier', None) == 'mobilenet_v3' and getattr(self, 'feature_extractor', None)):
                logger.error("MobileNetV3 classifier unavailable; returning unavailable result")
                return {'shape': 'oval', 'confidence': 0.0, 'method': 'unavailable'}

            mobilenet_result = self._predict_with_mobilenet(face_img)
            logger.info(f"MobileNetV3 prediction: {mobilenet_result['shape']} (confidence: {mobilenet_result['confidence']:.3f})")
            return mobilenet_result

        except Exception as e:
            logger.error(f"Face shape prediction error: {str(e)}")
            return {'shape': 'oval', 'confidence': 0.0, 'method': 'error'}

    def _predict_with_mobilenet(self, face_img):
        """Predict face shape using trained MobileNetV3 model"""
        try:
            # Prepare image for MobileNetV3 (same preprocessing as training)
            face_pil = Image.fromarray(face_img)
            
            # Apply standard ImageNet preprocessing (adjust if your training used different preprocessing)
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

    # Removed ResNet/geometric analysis helpers to enforce MobileNet-only classification

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
            print("DEBUG: Quality metrics - starting calculation")
            
            # Basic quality assessments
            h, w = face_img.shape[:2]
            print(f"DEBUG: Quality metrics - image dimensions: {w}x{h}")
            
            # Size quality
            min_size = 100
            size_score = min(1.0, min(w, h) / min_size)
            print(f"DEBUG: Quality metrics - size score: {size_score}")
            
            # Brightness quality
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128  # Optimal around 128
            brightness_score = max(0.0, min(1.0, brightness_score))
            print(f"DEBUG: Quality metrics - brightness score: {brightness_score}")
            
            # Contrast quality 
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 50)  # Good contrast above 50
            print(f"DEBUG: Quality metrics - contrast score: {contrast_score}")
            
            # Sharpness (simplified Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_score = min(1.0, sharpness / 100)  # Good sharpness above 100
            print(f"DEBUG: Quality metrics - sharpness score: {sharpness_score}")
            
            # Overall quality (weighted average)
            overall_quality = (
                size_score * 0.3 + 
                brightness_score * 0.25 + 
                contrast_score * 0.25 + 
                sharpness_score * 0.2
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
            
            print(f"DEBUG: Quality metrics - final result: {quality_metrics}")
            return quality_metrics
            
        except Exception as e:
            print(f"DEBUG: Quality metrics error: {str(e)}")
            return {
                'overall_quality': 0.5,
                'is_good_quality': True,
                'error': str(e)
            }


# Removed debug_resnet_features endpoint; MobileNet is the only classifier used.


def analyze_face_comprehensive(image_path):
    """
    Comprehensive face analysis that returns all needed data.
    This is the main function called by views.py
    """
    try:
        logger.info(f"Starting comprehensive face analysis for: {image_path}")
        
        # Initialize analyzer
        analyzer = FacialFeatureAnalyzer()
        logger.info(f"Analyzer initialized with detector type: {analyzer.detector_type}")
        
        # Perform face detection and analysis
        result, error = analyzer.detect_and_analyze_face(image_path)
        
        if result is None:
            logger.warning(f"Face analysis failed: {error}")
            # Return structured failure response
            return {
                'face_detected': False,
                'error': error or 'No face detected',
                'face_shape': {'shape': 'oval', 'confidence': 0.0},
                'quality_metrics': {'overall_quality': 0.0, 'is_good_quality': False},
                'facial_features': {},
                'confidence': 0.0,
                'detection_method': 'failed'
            }
        
        # Ensure all required fields are present
        face_shape = result.get('face_shape', {'shape': 'oval', 'confidence': 0.6})
        if not isinstance(face_shape, dict):
            face_shape = {'shape': str(face_shape), 'confidence': 0.6}
            
        quality_metrics = result.get('quality_metrics', {'overall_quality': 0.7, 'is_good_quality': True})
        
        response_data = {
            'face_detected': result.get('face_detected', True),
            'face_shape': face_shape,
            'facial_features': result.get('facial_features', {}),
            'confidence': result.get('confidence', face_shape.get('confidence', 0.6)),
            'face_box': result.get('face_box', []),
            'landmarks': result.get('landmarks', []),
            'quality_metrics': quality_metrics,
            'detection_method': result.get('detection_method', 'unknown')
        }
        
        logger.info(f"Face analysis completed successfully: shape={face_shape['shape']}, "
                   f"confidence={face_shape['confidence']:.3f}, method={response_data['detection_method']}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in comprehensive face analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'face_detected': False,
            'error': f"Analysis error: {str(e)}",
            'face_shape': {'shape': 'oval', 'confidence': 0.0},
            'quality_metrics': {'overall_quality': 0.0, 'is_good_quality': False},
            'facial_features': {},
            'confidence': 0.0,
            'detection_method': 'error'
        }