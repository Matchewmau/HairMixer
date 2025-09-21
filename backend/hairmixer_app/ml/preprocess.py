import cv2
import numpy as np
import logging
from PIL import Image as PILImage

# Try to import face_recognition, fallback to alternatives
FACE_RECOGNITION_AVAILABLE = False
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

def read_image(path):
    """Read image from file path with error handling"""
    try:
        # First try with OpenCV
        img = cv2.imread(str(path))
        if img is not None:
            return img
        
        # Fallback to PIL
        pil_img = PILImage.open(path)
        # Convert PIL to OpenCV format
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
        
    except Exception as e:
        logger.error(f"Error reading image {path}: {str(e)}")
        raise

def validate_image_quality(img):
    """Validate image quality for face detection"""
    try:
        if img is None:
            return {"is_valid": False, "error": "Image is None"}
        
        height, width = img.shape[:2]
        
        # Check minimum resolution
        if width < 200 or height < 200:
            return {"is_valid": False, "error": "Image resolution too low (minimum 200x200)"}
        
        # Check if image is too blurry using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:  # Threshold for blur detection
            return {"is_valid": False, "error": "Image appears to be too blurry"}
        
        # Check brightness
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            return {"is_valid": False, "error": "Image is too dark"}
        elif mean_brightness > 225:
            return {"is_valid": False, "error": "Image is too bright/overexposed"}
        
        return {"is_valid": True, "quality_score": min(laplacian_var / 100, 10)}
        
    except Exception as e:
        logger.error(f"Error validating image quality: {str(e)}")
        return {"is_valid": False, "error": "Unable to validate image quality"}

def to_model_input(img, size=(224, 224)):
    """Convert image to model input format with normalization"""
    try:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        img_resized = cv2.resize(img_rgb, size)
        # Normalize pixel values to [0, 1]
        arr = img_resized.astype(np.float32) / 255.0
        
        # Apply standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std
        
        # Transpose to CHW format (channels first) for PyTorch
        arr = np.transpose(arr, (2, 0, 1))
        # Add batch dimension
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def detect_face(img):
    """Enhanced face detection using multiple methods"""
    try:
        # Method 1: Use face_recognition if available
        if FACE_RECOGNITION_AVAILABLE:
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(img_rgb)
                if face_locations:
                    return True, face_locations
            except Exception as e:
                logger.warning(f"face_recognition failed: {e}")
        
        # Method 2: OpenCV Haar Cascade (fallback)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_cv = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        if len(faces_cv) > 0:
            return True, faces_cv
        else:
            return False, []
            
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return False, []

def extract_face_landmarks(img):
    """Extract facial landmarks for better analysis"""
    try:
        if FACE_RECOGNITION_AVAILABLE:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_landmarks_list = face_recognition.face_landmarks(img_rgb)
            if face_landmarks_list:
                return face_landmarks_list[0]  # Return first face landmarks
        
        # If face_recognition is not available, return empty dict
        logger.warning("Face landmarks extraction not available without face_recognition library")
        return {}
        
    except Exception as e:
        logger.error(f"Error extracting face landmarks: {str(e)}")
        return {}

def crop_face_region(img, face_box, padding=0.2):
    """Crop face region from image with padding"""
    try:
        if FACE_RECOGNITION_AVAILABLE and len(face_box) == 4 and isinstance(face_box[0], int):
            # face_recognition format: (top, right, bottom, left)
            top, right, bottom, left = face_box
            
            # Add padding
            height, width = img.shape[:2]
            pad_h = int((bottom - top) * padding)
            pad_w = int((right - left) * padding)
            
            # Calculate crop coordinates with bounds checking
            y1 = max(0, top - pad_h)
            y2 = min(height, bottom + pad_h)
            x1 = max(0, left - pad_w)
            x2 = min(width, right + pad_w)
            
            return img[y1:y2, x1:x2]
        
        else:
            # OpenCV format: (x, y, w, h)
            x, y, w, h = face_box
            
            # Add padding
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            
            # Calculate crop coordinates
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img.shape[1], x + w + pad_w)
            y2 = min(img.shape[0], y + h + pad_h)
            
            return img[y1:y2, x1:x2]
        
    except Exception as e:
        logger.error(f"Error cropping face region: {str(e)}")
        return img