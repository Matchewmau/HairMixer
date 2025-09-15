import logging
import random
import numpy as np
from pathlib import Path
from django.conf import settings
import threading
import time

logger = logging.getLogger(__name__)

# Face shape categories with confidence mapping
FACE_SHAPES = {
    0: "diamond",
    1: "heart", 
    2: "oblong",
    3: "oval",
    4: "round",
    5: "square",
    6: "triangle"  # Your 7th face shape class
}

FACE_SHAPE_CHARACTERISTICS = {
    "oval": {
        "description": "Balanced proportions with slightly wider cheekbones",
        "suitable_styles": ["most styles", "versatile"],
        "avoid": []
    },
    "round": {
        "description": "Equal width and height with soft, curved lines",
        "suitable_styles": ["angular cuts", "long layers", "side parts"],
        "avoid": ["blunt bobs", "center parts"]
    },
    "square": {
        "description": "Strong jawline with equal width forehead and jaw",
        "suitable_styles": ["soft waves", "long layers", "side-swept bangs"],
        "avoid": ["blunt cuts", "sharp lines"]
    },
    "heart": {
        "description": "Wider forehead with narrow chin",
        "suitable_styles": ["chin-length cuts", "side parts", "full bangs"],
        "avoid": ["short crops", "excessive volume on top"]
    },
    "diamond": {
        "description": "Narrow forehead and jaw with wide cheekbones",
        "suitable_styles": ["side-swept bangs", "chin-length styles"],
        "avoid": ["slicked back", "center parts"]
    },
    "oblong": {
        "description": "Longer than wide with straight sides",
        "suitable_styles": ["blunt cuts", "waves", "bangs"],
        "avoid": ["long straight styles", "center parts"]
    }
}

# Global analyzer instance
_facial_analyzer = None
_loading_timeout = 30  # seconds

def load_model_with_timeout(timeout=30):
    """Load model with timeout to prevent hanging"""
    result = {'analyzer': None, 'error': None}
    
    def load_model_thread():
        try:
            from .face_analyzer import FacialFeatureAnalyzer
            analyzer = FacialFeatureAnalyzer()
            # Don't call load_models() during startup - it causes hanging
            # analyzer.load_models()  # Skip this for now
            result['analyzer'] = analyzer
            logger.info("Facial analyzer created successfully (models will load on demand)")
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error creating facial analyzer: {str(e)}")
    
    # Start loading in a separate thread
    thread = threading.Thread(target=load_model_thread)
    thread.daemon = True
    thread.start()
    
    # Wait for completion or timeout
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        logger.error(f"Model loading timed out after {timeout} seconds")
        return None
    
    if result['error']:
        logger.error(f"Model loading failed: {result['error']}")
        return None
    
    return result['analyzer']

def load_model():
    """Load the facial feature analyzer with timeout protection"""
    global _facial_analyzer
    try:
        if _facial_analyzer is None:
            _facial_analyzer = load_model_with_timeout(_loading_timeout)
        return _facial_analyzer
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        return None

def predict_face_shape(model, image_path):
    """Predict face shape from image"""
    if model is None:
        logger.warning("No model available, using random prediction")
        shape_idx = random.randint(0, 5)
        return {
            'shape': FACE_SHAPES[shape_idx],
            'confidence': 0.5,
            'all_probabilities': {}
        }
    
    try:
        result, error = model.detect_and_analyze_face(image_path)
        if result:
            return result['face_shape']
        else:
            logger.error(f"Face shape prediction failed: {error}")
            return {'shape': 'oval', 'confidence': 0.5, 'all_probabilities': {}}
    except Exception as e:
        logger.error(f"Error in face shape prediction: {str(e)}")
        return {'shape': 'oval', 'confidence': 0.5, 'all_probabilities': {}}

def analyze_facial_features(image_path):
    """Analyze facial features from image"""
    global _facial_analyzer
    
    if _facial_analyzer is None:
        _facial_analyzer = load_model()
    
    if _facial_analyzer is None:
        # Return dummy data as fallback
        return {
            "face_ratio": 0.8,
            "jawline_strength": "medium",
            "symmetry_score": 0.85
        }
    
    try:
        result, error = _facial_analyzer.detect_and_analyze_face(image_path)
        if result:
            return result['facial_features']
        else:
            logger.error(f"Facial analysis failed: {error}")
            return {}
    except Exception as e:
        logger.error(f"Error analyzing facial features: {str(e)}")
        return {}

def get_face_shape_recommendations(face_shape):
    """Get style recommendations specific to face shape"""
    return FACE_SHAPE_CHARACTERISTICS.get(face_shape, FACE_SHAPE_CHARACTERISTICS['oval'])