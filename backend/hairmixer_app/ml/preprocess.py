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
        pil_img = PILImage.open(path).convert('RGB')
        # Return as RGB numpy array
        return np.array(pil_img)
        
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
        # Ensure RGB; if grayscale, expand
        if img.ndim == 2:
            gray = img.astype(np.float32)
        else:
            # luminance grayscale
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        # Approximate sharpness using gradient variance
        gy, gx = np.gradient(gray)
        laplacian_var = (gx**2 + gy**2).var()
        
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
        # Ensure RGB and resize via PIL
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img
        img_resized = np.array(PILImage.fromarray(img_rgb).resize(size, PILImage.Resampling.BILINEAR))
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
                # face_recognition expects RGB
                img_rgb = img if (img.ndim == 3 and img.shape[2] == 3) else np.stack([img, img, img], axis=-1)
                face_locations = face_recognition.face_locations(img_rgb)
                if face_locations:
                    return True, face_locations
            except Exception as e:
                logger.warning(f"face_recognition failed: {e}")

        # No OpenCV fallback; report not detected
        return False, []
            
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return False, []

def extract_face_landmarks(img):
    """Extract facial landmarks for better analysis"""
    try:
        if FACE_RECOGNITION_AVAILABLE:
            img_rgb = img if (img.ndim == 3 and img.shape[2] == 3) else np.stack([img, img, img], axis=-1)
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
            # Generic (x, y, w, h)
            x, y, w, h = face_box
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img.shape[1], x + w + pad_w)
            y2 = min(img.shape[0], y + h + pad_h)
            return img[y1:y2, x1:x2]
        
    except Exception as e:
        logger.error(f"Error cropping face region: {str(e)}")
        return img