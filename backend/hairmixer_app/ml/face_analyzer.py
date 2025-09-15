import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image
from django.conf import settings
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

class FaceShapeClassifier(nn.Module):
    """Deep learning model for face shape classification"""
    
    def __init__(self, num_classes=6):
        super(FaceShapeClassifier, self).__init__()
        self.backbone = resnet50(pretrained=True)
        # Replace final layer for face shape classification
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

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
        
        # Initialize transforms for ResNet model
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
        """Load MobileNetV3 or fallback to ResNet50 for feature extraction"""
        try:
            # First try to load the MobileNetV3 model
            if MOBILENET_AVAILABLE:
                logger.info("Attempting to load MobileNetV3 face shape classifier...")
                if mobile_net_loader.load_model():
                    self.shape_classifier = 'mobilenet_v3'
                    self.feature_extractor = mobile_net_loader
                    logger.info("MobileNetV3 face shape classifier loaded successfully")
                    return True
                else:
                    logger.warning("MobileNetV3 model loading failed, falling back to ResNet50")
            
            # Fallback to ResNet50 feature extraction
            logger.info("Loading ResNet50 feature extractor...")
            
            # Try different approaches to load ResNet50
            try:
                # First try: Standard pretrained loading
                logger.info("Attempting standard ResNet50 loading...")
                self.feature_extractor = resnet50(pretrained=True)
                logger.info("Standard ResNet50 loading successful")
                
            except Exception as e1:
                logger.warning(f"Standard ResNet50 loading failed: {str(e1)}")
                try:
                    # Second try: Load without pretrained weights
                    logger.info("Attempting ResNet50 without pretrained weights...")
                    self.feature_extractor = resnet50(pretrained=False)
                    logger.info("ResNet50 loaded without pretrained weights")
                    
                except Exception as e2:
                    logger.error(f"ResNet50 loading completely failed: {str(e2)}")
                    # Use a simple CNN instead
                    logger.info("Falling back to simple feature extractor...")
                    self.feature_extractor = self._create_simple_feature_extractor()
            
            if self.feature_extractor is not None:
                # Remove the final classification layer to get features
                if hasattr(self.feature_extractor, 'fc'):
                    # For ResNet models
                    self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
                
                # Move to device and set to evaluation mode
                self.feature_extractor.to(self.device)
                self.feature_extractor.eval()
                
                logger.info("Feature extractor loaded and configured successfully")
                
                # Mark that we have feature-based analysis available
                self.shape_classifier = 'feature_based'
                
                return True
            
        except Exception as e:
            logger.warning(f"Could not load any feature extractor: {str(e)}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            
        # If all else fails
        self.feature_extractor = None
        self.shape_classifier = None
        return False
    
    def _create_simple_feature_extractor(self):
        """Create a simple CNN for feature extraction as fallback"""
        try:
            class SimpleCNN(nn.Module):
                def __init__(self):
                    super(SimpleCNN, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(128, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1, 1))
                    )
                
                def forward(self, x):
                    x = self.features(x)
                    x = torch.flatten(x, 1)
                    return x
            
            logger.info("Created simple CNN feature extractor")
            return SimpleCNN()
            
        except Exception as e:
            logger.error(f"Could not create simple feature extractor: {str(e)}")
            return None

    def _predict_face_shape_simple(self, face_img):
        """Enhanced face shape prediction using MobileNetV3 or ResNet50 features + geometric analysis"""
        try:
            # First get basic geometric analysis
            geometric_result = self._geometric_face_shape_analysis(face_img)
            
            # Check if MobileNetV3 is available and loaded
            if (MOBILENET_AVAILABLE and 
                hasattr(self, 'shape_classifier') and 
                self.shape_classifier == 'mobilenet_v3'):
                
                try:
                    # Use MobileNetV3 for direct prediction
                    mobilenet_result = self._predict_with_mobilenet(face_img)
                    logger.info(f"MobileNetV3 prediction: {mobilenet_result['shape']} "
                               f"(confidence: {mobilenet_result['confidence']:.3f})")
                    return mobilenet_result
                    
                except Exception as e:
                    logger.warning(f"MobileNetV3 prediction failed: {str(e)}, using geometric only")
            
            # If ResNet50 feature extractor is available, enhance the prediction
            elif (hasattr(self, 'feature_extractor') and 
                  self.feature_extractor is not None and 
                  self.shape_classifier == 'feature_based'):
                
                try:
                    # Extract deep features using ResNet50
                    enhanced_result = self._predict_with_resnet_features(face_img, geometric_result)
                    logger.info(f"ResNet50-enhanced prediction: {enhanced_result['shape']} "
                               f"(confidence: {enhanced_result['confidence']:.3f})")
                    return enhanced_result
                    
                except Exception as e:
                    logger.warning(f"ResNet50 feature extraction failed: {str(e)}, using geometric only")
            
            # Fallback to geometric analysis only
            return geometric_result
            
        except Exception as e:
            logger.error(f"Face shape prediction error: {str(e)}")
            return {'shape': 'oval', 'confidence': 0.5, 'method': 'fallback'}

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

    def _predict_with_resnet_features(self, face_img, geometric_result):
        """Predict face shape using ResNet50 features to enhance geometric analysis"""
        try:
            # Prepare image for ResNet50
            face_pil = Image.fromarray(face_img)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Extract features using ResNet50
            with torch.no_grad():
                features = self.feature_extractor(face_tensor)
                features = features.flatten().cpu().numpy()
            
            # Analyze feature characteristics
            feature_stats = self._analyze_resnet_features(features)
            
            # Enhanced geometric analysis with ResNet features
            enhanced_result = self._combine_features_with_geometry(
                face_img, geometric_result, feature_stats
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"ResNet50 feature prediction error: {str(e)}")
            return geometric_result

    def _analyze_resnet_features(self, features):
        """Analyze ResNet50 features to extract facial characteristics"""
        try:
            # Calculate feature statistics
            feature_mean = np.mean(features)
            feature_std = np.std(features)
            feature_var = np.var(features)
            feature_max = np.max(features)
            feature_min = np.min(features)
            
            # Calculate feature distribution characteristics
            feature_skewness = self._calculate_skewness(features)
            feature_kurtosis = self._calculate_kurtosis(features)
            
            # Analyze feature activation patterns
            high_activation_ratio = np.sum(features > feature_mean + feature_std) / len(features)
            low_activation_ratio = np.sum(features < feature_mean - feature_std) / len(features)
            
            # Feature quality indicators
            feature_diversity = feature_std / (abs(feature_mean) + 1e-8)
            feature_complexity = np.sum(np.abs(features) > 0.1) / len(features)
            
            return {
                'mean': feature_mean,
                'std': feature_std,
                'variance': feature_var,
                'max': feature_max,
                'min': feature_min,
                'skewness': feature_skewness,
                'kurtosis': feature_kurtosis,
                'high_activation_ratio': high_activation_ratio,
                'low_activation_ratio': low_activation_ratio,
                'diversity': feature_diversity,
                'complexity': feature_complexity,
                'quality_score': min(1.0, feature_diversity * feature_complexity)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ResNet features: {str(e)}")
            return {'quality_score': 0.5}

    def _combine_features_with_geometry(self, face_img, geometric_result, feature_stats):
        """Combine ResNet50 features with geometric analysis for enhanced prediction"""
        try:
            h, w = face_img.shape[:2]
            
            # Get basic measurements
            aspect_ratio = w / h
            
            # Enhanced geometric measurements
            enhanced_measurements = self._get_enhanced_measurements(face_img)
            
            # Feature-informed face shape classification
            base_shape = geometric_result['shape']
            base_confidence = geometric_result['confidence']
            
            # Use feature statistics to refine classification
            refined_result = self._refine_classification_with_features(
                base_shape, base_confidence, aspect_ratio, 
                enhanced_measurements, feature_stats
            )
            
            # Calculate final confidence based on feature quality and geometric consistency
            feature_quality = feature_stats.get('quality_score', 0.5)
            geometric_confidence = base_confidence
            
            # Weighted confidence combining both approaches
            final_confidence = (geometric_confidence * 0.6 + feature_quality * 0.4)
            final_confidence = min(0.95, max(0.3, final_confidence))  # Clamp between 0.3 and 0.95
            
            return {
                'shape': refined_result['shape'],
                'confidence': final_confidence,
                'method': 'resnet50_enhanced_geometric',
                'geometric_base': base_shape,
                'feature_quality': feature_quality,
                'measurements': enhanced_measurements,
                'feature_stats': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                 for k, v in feature_stats.items()}
            }
            
        except Exception as e:
            logger.error(f"Error combining features with geometry: {str(e)}")
            return geometric_result

    def _get_enhanced_measurements(self, face_img):
        """Get enhanced facial measurements using edge detection and contour analysis"""
        try:
            h, w = face_img.shape[:2]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY) if len(face_img.shape) == 3 else face_img
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Measure widths at different heights
            measurements = {}
            
            # Forehead width (top 25%)
            forehead_y = int(h * 0.25)
            measurements['forehead_width'] = self._measure_width_at_height(edges, forehead_y)
            
            # Eye level width (around 40%)
            eye_y = int(h * 0.4)
            measurements['eye_level_width'] = self._measure_width_at_height(edges, eye_y)
            
            # Cheekbone width (around 50-60%)
            cheekbone_y = int(h * 0.55)
            measurements['cheekbone_width'] = self._measure_width_at_height(edges, cheekbone_y)
            
            # Mouth level width (around 70%)
            mouth_y = int(h * 0.7)
            measurements['mouth_level_width'] = self._measure_width_at_height(edges, mouth_y)
            
            # Jawline width (bottom 15%)
            jaw_y = int(h * 0.85)
            measurements['jaw_width'] = self._measure_width_at_height(edges, jaw_y)
            
            # Calculate ratios
            max_width = max(measurements.values()) if measurements.values() else w
            measurements['forehead_ratio'] = measurements['forehead_width'] / max_width if max_width > 0 else 0
            measurements['cheekbone_ratio'] = measurements['cheekbone_width'] / max_width if max_width > 0 else 0
            measurements['jaw_ratio'] = measurements['jaw_width'] / max_width if max_width > 0 else 0
            
            # Face proportions
            measurements['aspect_ratio'] = w / h
            measurements['face_length'] = h
            measurements['face_width'] = w
            
            return measurements
            
        except Exception as e:
            logger.error(f"Error in enhanced measurements: {str(e)}")
            return {'aspect_ratio': 1.0}

    def _refine_classification_with_features(self, base_shape, base_confidence, aspect_ratio, measurements, feature_stats):
        """Refine face shape classification using feature insights"""
        try:
            feature_complexity = feature_stats.get('complexity', 0.5)
            feature_diversity = feature_stats.get('diversity', 0.5)
            
            # Feature-informed refinements
            refinements = {
                'oval': self._check_oval_features(measurements, feature_stats),
                'round': self._check_round_features(measurements, feature_stats),
                'square': self._check_square_features(measurements, feature_stats),
                'heart': self._check_heart_features(measurements, feature_stats),
                'diamond': self._check_diamond_features(measurements, feature_stats),
                'oblong': self._check_oblong_features(measurements, feature_stats)
            }
            
            # Get confidence scores for each shape
            shape_scores = {}
            for shape, score in refinements.items():
                shape_scores[shape] = score
            
            # Find the best match
            best_shape = max(shape_scores.items(), key=lambda x: x[1])
            
            # If the feature analysis strongly suggests a different shape, consider switching
            if best_shape[1] > 0.7 and best_shape[0] != base_shape:
                return {'shape': best_shape[0]}
            
            # Otherwise, stick with geometric analysis but maybe adjust confidence
            return {'shape': base_shape}
            
        except Exception as e:
            logger.error(f"Error in feature-based refinement: {str(e)}")
            return {'shape': base_shape}

    def _check_oval_features(self, measurements, feature_stats):
        """Check if features suggest oval face shape"""
        aspect_ratio = measurements.get('aspect_ratio', 1.0)
        forehead_ratio = measurements.get('forehead_ratio', 0.5)
        jaw_ratio = measurements.get('jaw_ratio', 0.5)
        
        # Oval faces have balanced proportions
        balance_score = 1.0 - abs(forehead_ratio - jaw_ratio)
        aspect_score = 1.0 - abs(aspect_ratio - 1.2) / 0.5  # Ideal aspect ratio around 1.2
        
        return (balance_score * 0.6 + aspect_score * 0.4) * feature_stats.get('quality_score', 0.5)

    def _check_round_features(self, measurements, feature_stats):
        """Check if features suggest round face shape"""
        aspect_ratio = measurements.get('aspect_ratio', 1.0)
        cheekbone_ratio = measurements.get('cheekbone_ratio', 0.5)
        
        # Round faces have aspect ratio close to 1.0 and prominent cheekbones
        aspect_score = 1.0 - abs(aspect_ratio - 1.0) / 0.3
        cheekbone_score = cheekbone_ratio
        
        return (aspect_score * 0.7 + cheekbone_score * 0.3) * feature_stats.get('quality_score', 0.5)

    def _check_square_features(self, measurements, feature_stats):
        """Check if features suggest square face shape"""
        aspect_ratio = measurements.get('aspect_ratio', 1.0)
        jaw_ratio = measurements.get('jaw_ratio', 0.5)
        forehead_ratio = measurements.get('forehead_ratio', 0.5)
        
        # Square faces have strong jawline and similar forehead width
        jaw_strength = jaw_ratio
        proportion_balance = 1.0 - abs(jaw_ratio - forehead_ratio)
        aspect_score = 1.0 - abs(aspect_ratio - 1.0) / 0.2
        
        return (jaw_strength * 0.4 + proportion_balance * 0.3 + aspect_score * 0.3) * feature_stats.get('quality_score', 0.5)

    def _check_heart_features(self, measurements, feature_stats):
        """Check if features suggest heart face shape"""
        forehead_ratio = measurements.get('forehead_ratio', 0.5)
        jaw_ratio = measurements.get('jaw_ratio', 0.5)
        
        # Heart faces have wide forehead and narrow jaw
        forehead_prominence = forehead_ratio
        jaw_narrowness = 1.0 - jaw_ratio
        ratio_difference = max(0, forehead_ratio - jaw_ratio) / 0.3
        
        return (forehead_prominence * 0.4 + jaw_narrowness * 0.3 + ratio_difference * 0.3) * feature_stats.get('quality_score', 0.5)

    def _check_diamond_features(self, measurements, feature_stats):
        """Check if features suggest diamond face shape"""
        cheekbone_ratio = measurements.get('cheekbone_ratio', 0.5)
        forehead_ratio = measurements.get('forehead_ratio', 0.5)
        jaw_ratio = measurements.get('jaw_ratio', 0.5)
        
        # Diamond faces have prominent cheekbones with narrower forehead and jaw
        cheekbone_prominence = cheekbone_ratio
        forehead_narrowness = 1.0 - forehead_ratio
        jaw_narrowness = 1.0 - jaw_ratio
        
        return (cheekbone_prominence * 0.5 + forehead_narrowness * 0.25 + jaw_narrowness * 0.25) * feature_stats.get('quality_score', 0.5)

    def _check_oblong_features(self, measurements, feature_stats):
        """Check if features suggest oblong face shape"""
        aspect_ratio = measurements.get('aspect_ratio', 1.0)
        
        # Oblong faces have high aspect ratio (taller than wide)
        elongation_score = max(0, (1.5 - aspect_ratio) / 0.7)  # Higher score for lower aspect ratios
        
        return elongation_score * feature_stats.get('quality_score', 0.5)

    def _measure_width_at_height(self, edges, y):
        """Measure the width of face at a specific height using edge detection"""
        try:
            if y >= edges.shape[0] or y < 0:
                return 0
            
            row = edges[y, :]
            edge_pixels = np.where(row > 0)[0]
            
            if len(edge_pixels) >= 2:
                return edge_pixels[-1] - edge_pixels[0]
            return 0
        except:
            return 0

    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0

    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        except:
            return 0

    def _geometric_face_shape_analysis(self, face_img):
        """Enhanced geometric face shape analysis"""
        try:
            h, w = face_img.shape[:2]
            aspect_ratio = w / h
            
            # Get enhanced measurements
            measurements = self._get_enhanced_measurements(face_img)
            
            forehead_ratio = measurements.get('forehead_ratio', 0.5)
            cheekbone_ratio = measurements.get('cheekbone_ratio', 0.5)
            jaw_ratio = measurements.get('jaw_ratio', 0.5)
            
            logger.info(f"Enhanced measurements - aspect: {aspect_ratio:.2f}, "
                       f"forehead: {forehead_ratio:.2f}, cheekbone: {cheekbone_ratio:.2f}, jaw: {jaw_ratio:.2f}")
            
            # Enhanced classification rules
            confidence = 0.7
            
            # Round face: close to square aspect ratio + soft features
            if 0.9 < aspect_ratio < 1.15 and abs(forehead_ratio - jaw_ratio) < 0.1:
                shape = 'round'
                confidence = 0.8
                
            # Square face: close to square aspect ratio + strong jaw
            elif 0.85 < aspect_ratio < 1.15 and jaw_ratio > 0.75:
                shape = 'square'
                confidence = 0.75
                
            # Oblong/Long face: tall aspect ratio
            elif aspect_ratio < 0.8:
                shape = 'oblong'
                confidence = 0.8
                
            # Heart face: wide forehead, narrow jaw
            elif forehead_ratio > jaw_ratio + 0.15 and forehead_ratio > 0.7:
                shape = 'heart'
                confidence = 0.75
                
            # Diamond face: wide cheekbones, narrow forehead and jaw
            elif (cheekbone_ratio > forehead_ratio + 0.1 and 
                  cheekbone_ratio > jaw_ratio + 0.1 and 
                  cheekbone_ratio > 0.7):
                shape = 'diamond'
                confidence = 0.7
                
            # Default to oval for balanced features
            else:
                shape = 'oval'
                confidence = 0.8
            
            logger.info(f"Enhanced geometric analysis: {shape} (confidence: {confidence})")
            
            return {
                'shape': shape,
                'confidence': confidence,
                'method': 'geometric_enhanced',
                'measurements': measurements
            }
            
        except Exception as e:
            logger.error(f"Enhanced geometric analysis error: {str(e)}")
            return {'shape': 'oval', 'confidence': 0.5, 'method': 'geometric_fallback'}

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


@api_view(['POST'])
@permission_classes([AllowAny])
def debug_resnet_features(request):
    """Debug endpoint to verify ResNet50 feature extraction"""
    try:
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'})
        
        from .ml.face_analyzer import FacialFeatureAnalyzer
        
        # Save temp image
        image_file = request.FILES['image']
        temp_path = f"/tmp/debug_resnet_{uuid.uuid4()}.jpg"
        with open(temp_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)
        
        # Initialize analyzer
        analyzer = FacialFeatureAnalyzer()
        
        # Check if ResNet50 is loaded
        debug_info = {
            'resnet_available': hasattr(analyzer, 'feature_extractor') and analyzer.feature_extractor is not None,
            'classifier_type': getattr(analyzer, 'shape_classifier', 'None'),
            'device': str(analyzer.device),
        }
        
        # Test face analysis
        result, error = analyzer.detect_and_analyze_face(temp_path)
        
        if result:
            debug_info['face_detected'] = True
            debug_info['face_shape_result'] = result.get('face_shape', {})
            debug_info['detection_method'] = result.get('detection_method', 'unknown')
            
            # Check if ResNet50 method was used
            face_shape_info = result.get('face_shape', {})
            debug_info['used_resnet'] = face_shape_info.get('method') == 'resnet50_enhanced_geometric'
            debug_info['feature_quality'] = face_shape_info.get('feature_quality', 'N/A')
            
        else:
            debug_info['face_detected'] = False
            debug_info['error'] = error
        
        # Cleanup
        import os
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return Response({'debug_info': debug_info})
        
    except Exception as e:
        import traceback
        return Response({
            'error': str(e), 
            'traceback': traceback.format_exc()
        })


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