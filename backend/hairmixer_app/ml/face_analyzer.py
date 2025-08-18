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

# Try importing optional packages with better error handling
FACENET_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False

try:
    from facenet_pytorch import MTCNN
    FACENET_AVAILABLE = True
    print("✅ FaceNet PyTorch available")
except ImportError as e:
    print(f"⚠️  FaceNet PyTorch not available: {e}")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
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
    """Advanced facial feature analysis using deep learning"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_detector = None
        self.shape_classifier = None
        
        # Initialize face detector
        if FACENET_AVAILABLE:
            self.face_detector = MTCNN(keep_all=True, device=self.device)
            self.detector_type = 'facenet'
        elif MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
            self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
            self.detector_type = 'mediapipe'
        else:
            logger.warning("No face detection library available, falling back to OpenCV")
            self.detector_type = 'opencv'
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load face shape classifier if available
            model_path = Path(settings.BASE_DIR) / 'ml_models' / 'face_shape_classifier.pth'
            if model_path.exists():
                self.shape_classifier = FaceShapeClassifier()
                self.shape_classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                self.shape_classifier.eval()
                logger.info("Face shape classifier loaded successfully")
            else:
                logger.warning("Face shape model not found, will use geometric analysis")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def detect_and_analyze_face(self, image_path):
        """Detect face and analyze facial structure"""
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return None, "Could not load image"
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect face using available method
            if self.detector_type == 'facenet':
                return self._detect_with_facenet(img_rgb)
            elif self.detector_type == 'mediapipe':
                return self._detect_with_mediapipe(img_rgb)
            else:
                return self._detect_with_opencv(img)
            
        except Exception as e:
            logger.error(f"Error in face analysis: {str(e)}")
            return None, str(e)
    
    def _detect_with_facenet(self, img_rgb):
        """Use FaceNet for face detection"""
        boxes, probs, landmarks = self.face_detector.detect(img_rgb, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            return None, "No face detected"
        
        # Use the most confident detection
        best_idx = torch.argmax(probs)
        face_box = boxes[best_idx]
        face_landmarks = landmarks[best_idx] if landmarks is not None else None
        
        # Extract face region
        x1, y1, x2, y2 = face_box.int()
        face_img = img_rgb[y1:y2, x1:x2]
        
        # Analyze facial structure
        analysis = self._analyze_facial_structure_basic(face_img, face_landmarks)
        
        # Predict face shape
        face_shape = self._predict_face_shape(face_img)
        
        return {
            'face_detected': True,
            'confidence': float(probs[best_idx]),
            'face_shape': face_shape,
            'facial_features': analysis,
            'landmarks': face_landmarks.tolist() if face_landmarks is not None else [],
            'face_box': face_box.tolist()
        }, None
    
    def _detect_with_mediapipe(self, img_rgb):
        """Use MediaPipe for face detection"""
        results = self.face_detector.process(img_rgb)
        
        if not results.detections:
            return None, "No face detected"
        
        # Get the most confident detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w, _ = img_rgb.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        face_img = img_rgb[y:y+height, x:x+width]
        
        # Get face landmarks
        mesh_results = self.face_mesh.process(img_rgb)
        landmarks = None
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
        
        # Analyze facial structure
        analysis = self._analyze_facial_structure_basic(face_img, landmarks)
        
        # Predict face shape
        face_shape = self._predict_face_shape(face_img)
        
        return {
            'face_detected': True,
            'confidence': detection.score[0],
            'face_shape': face_shape,
            'facial_features': analysis,
            'landmarks': [],
            'face_box': [x, y, x+width, y+height]
        }, None
    
    def _detect_with_opencv(self, img):
        """Fallback to OpenCV face detection"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, "No face detected"
        
        # Use the first detected face
        x, y, w, h = faces[0]
        face_img = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        
        # Basic analysis without landmarks
        analysis = self._analyze_facial_structure_basic(face_img, None)
        
        # Predict face shape
        face_shape = self._predict_face_shape(face_img)
        
        return {
            'face_detected': True,
            'confidence': 0.8,
            'face_shape': face_shape,
            'facial_features': analysis,
            'landmarks': [],
            'face_box': [x, y, x+w, y+h]
        }, None
    
    def _analyze_facial_structure_basic(self, face_img, landmarks):
        """Basic facial structure analysis"""
        try:
            h, w = face_img.shape[:2]
            
            # Basic measurements from image dimensions
            features = {
                'face_width': w,
                'face_height': h,
                'face_ratio': w / h,
                'jawline_width': int(w * 0.8),
                'forehead_width': int(w * 0.9),
                'cheekbone_width': int(w * 0.95),
                'forehead_height': int(h * 0.3),
                'jawline_strength': 'medium',
                'cheekbone_prominence': 'medium',
                'symmetry_score': 0.85
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing facial structure: {str(e)}")
            return {}
    
    def _predict_face_shape(self, face_img):
        """Predict face shape using available method"""
        if self.shape_classifier is not None:
            try:
                # Use trained model
                face_tensor = self.transform(Image.fromarray(face_img)).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.shape_classifier(face_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                
                shape = FACE_SHAPES[predicted_class]
                
                return {
                    'shape': shape,
                    'confidence': confidence,
                    'all_probabilities': {
                        FACE_SHAPES[i]: float(probabilities[0, i]) 
                        for i in range(len(FACE_SHAPES))
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in ML face shape prediction: {str(e)}")
        
        # Fallback to geometric analysis
        return self._geometric_face_shape_analysis(face_img)
    
    def _geometric_face_shape_analysis(self, face_img):
        """Geometric face shape analysis as fallback"""
        try:
            h, w = face_img.shape[:2]
            ratio = w / h
            
            # Simple geometric rules
            if ratio > 0.95:
                shape = 'round'
                confidence = 0.7
            elif ratio < 0.75:
                shape = 'oblong'
                confidence = 0.7
            elif 0.85 < ratio < 0.95:
                shape = 'square'
                confidence = 0.6
            else:
                shape = 'oval'
                confidence = 0.6
            
            return {
                'shape': shape,
                'confidence': confidence,
                'all_probabilities': {s: 0.1 for s in FACE_SHAPES.values()}
            }
            
        except Exception as e:
            logger.error(f"Error in geometric analysis: {str(e)}")
            return {'shape': 'oval', 'confidence': 0.5, 'all_probabilities': {}}