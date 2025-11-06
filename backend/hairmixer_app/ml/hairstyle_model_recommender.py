"""
Hairstyle Recommender using One-Hot Encoded Random Forest Model

This module provides hairstyle recommendations using a Random Forest classifier
with one-hot encoded features. The model predicts specific hairstyle names
based on user preferences.

Model Files:
    - hairstyle_model.joblib: Trained Random Forest classifier
    - model_columns.joblib: Feature column names (one-hot encoded)
    - hairstyle_family_label_encoder.joblib: Label encoder for target classes

Features:
    The model uses one-hot encoding for the following categorical features:
    - faceshape: heart, oblong, oval, round, square
    - gender: female, male
    - hair_type, hair_length, hair_color, hair_condition
    - lifestyle: active, casual, creative, moderate, professional, relaxed
    - maintenance: low, medium, high
    - styling_maintenance, styling_preference
    - texture_detail, thickness, volume
    - wants_bangs: yes, no
    - occasions: combinations of work, casual, formal, party, wedding, birthday
"""

import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class HairstyleModelRecommender:
    """Random Forest-based hairstyle recommender with one-hot encoding"""
    
    def __init__(self):
        """Initialize recommender and load model files"""
        self.rf_model = None
        self.model_columns = None
        self.label_encoder = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained Random Forest model and related files"""
        try:
            models_path = (
                Path(settings.BASE_DIR) / 'hairmixer_app' / 'ml' / 
                'models' / 'hairstyle_model'
            )
            
            # Verify directory exists
            if not models_path.exists():
                logger.error(f"Model directory not found: {models_path}")
                return
            
            # Load Random Forest model
            model_file = models_path / 'hairstyle_model.joblib'
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return
                
            self.rf_model = joblib.load(model_file)
            logger.info(f"✓ Loaded Random Forest model with {self.rf_model.n_estimators} estimators")
            
            # Load model columns (one-hot encoded feature names)
            columns_file = models_path / 'model_columns.joblib'
            if not columns_file.exists():
                logger.error(f"Model columns file not found: {columns_file}")
                self.rf_model = None
                return
                
            self.model_columns = joblib.load(columns_file)
            logger.info(f"✓ Loaded {len(self.model_columns)} one-hot encoded features")
            
            # Load label encoder for target classes
            encoder_file = models_path / 'hairstyle_family_label_encoder.joblib'
            if not encoder_file.exists():
                logger.error(f"Label encoder file not found: {encoder_file}")
                self.rf_model = None
                return
                
            self.label_encoder = joblib.load(encoder_file)
            logger.info(f"✓ Loaded label encoder with {len(self.label_encoder.classes_)} hairstyle classes")
            
        except Exception as e:
            logger.error(f"Failed to load hairstyle model: {e}", exc_info=True)
            self.rf_model = None
            self.model_columns = None
            self.label_encoder = None
    
    def predict_top_k(
        self, 
        user_preferences: Dict[str, Any], 
        face_shape: str, 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Predict top K hairstyle recommendations
        
        Args:
            user_preferences: Dict with user preference fields
            face_shape: Detected face shape from face analyzer
            k: Number of recommendations to return (default 10)
            
        Returns:
            List of dicts with keys: hairstyle_name, confidence, rank
        """
        if not self.is_available():
            logger.error("Model not loaded")
            return []
        
        try:
            # Prepare features
            features = self._prepare_features(user_preferences, face_shape)
            
            # Create one-hot encoded DataFrame
            X = self._create_feature_vector(features)
            
            if X is None:
                logger.error("Feature encoding failed")
                return []
            
            # Get predictions
            probabilities = self.rf_model.predict_proba(X)[0]
            
            # Get top K indices
            top_k_indices = np.argsort(probabilities)[-k:][::-1]
            
            # Decode hairstyle names
            recommendations = []
            for idx in top_k_indices:
                try:
                    hairstyle_name = self.label_encoder.inverse_transform([idx])[0]
                    confidence = float(probabilities[idx])
                    recommendations.append({
                        'hairstyle_name': hairstyle_name,
                        'confidence': confidence,
                        'rank': len(recommendations) + 1
                    })
                except Exception as e:
                    logger.warning(f"Error decoding hairstyle at index {idx}: {e}")
                    continue
            
            logger.info(
                f"Generated {len(recommendations)} recommendations for "
                f"face_shape={face_shape}, gender={features.get('gender', 'N/A')}"
            )
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}", exc_info=True)
            return []
    
    def _prepare_features(
        self, 
        user_preferences: Dict[str, Any], 
        face_shape: str
    ) -> Dict[str, Any]:
        """
        Prepare and normalize features from user preferences
        
        Args:
            user_preferences: Raw user preference data
            face_shape: Detected face shape
            
        Returns:
            Dictionary with normalized feature values
        """
        features = {}
        
        # Face shape (normalize to model's expected values)
        features['faceshape'] = self._normalize_faceshape(face_shape)
        
        # Gender
        features['gender'] = str(user_preferences.get('gender', 'female')).lower()
        if features['gender'] not in ['male', 'female']:
            features['gender'] = 'female'
        
        # Hair type
        hair_type = str(user_preferences.get('hair_type', 'straight')).lower()
        features['hair_type'] = hair_type if hair_type in ['straight', 'wavy', 'curly'] else 'straight'
        
        # Hair length
        hair_length = str(user_preferences.get('hair_length', 'medium')).lower()
        features['hair_length'] = hair_length if hair_length in ['short', 'medium', 'long'] else 'medium'
        
        # Hair color
        hair_color = str(user_preferences.get('hair_color', 'natural')).lower()
        valid_colors = ['natural', 'black', 'brown', 'blonde', 'red', 'gray', 'white', 'auburn', 'other']
        features['hair_color'] = hair_color if hair_color in valid_colors else 'natural'
        
        # Hair condition (can be multiple, comma-separated)
        condition = user_preferences.get('hair_condition', 'none')
        if isinstance(condition, list):
            # Join multiple conditions with comma
            features['hair_condition'] = ','.join(sorted(condition))
        else:
            features['hair_condition'] = str(condition).lower()
        
        # Lifestyle
        lifestyle = str(user_preferences.get('lifestyle', 'casual')).lower()
        valid_lifestyles = ['active', 'casual', 'creative', 'moderate', 'professional', 'relaxed']
        features['lifestyle'] = lifestyle if lifestyle in valid_lifestyles else 'casual'
        
        # Maintenance
        maintenance = str(user_preferences.get('maintenance', 'medium')).lower()
        features['maintenance'] = maintenance if maintenance in ['low', 'medium', 'high'] else 'medium'
        
        # Styling maintenance
        styling_maint = str(user_preferences.get('styling_maintenance', 'medium')).lower()
        features['styling_maintenance'] = styling_maint if styling_maint in ['low', 'medium', 'high'] else 'medium'
        
        # Styling preference
        styling_pref = str(user_preferences.get('styling_preference', 'natural')).lower()
        valid_prefs = ['natural', 'casual', 'classic', 'polished', 'elegant', 'glamorous', 'edgy', 'trendy']
        features['styling_preference'] = styling_pref if styling_pref in valid_prefs else 'natural'
        
        # Texture detail
        texture = str(user_preferences.get('hair_texture_detail', 'normal')).lower()
        valid_textures = ['fine', 'normal', 'thick', 'smooth', 'coarse', 'silky', 'frizzy']
        features['texture_detail'] = texture if texture in valid_textures else 'normal'
        
        # Thickness
        thickness = str(user_preferences.get('hair_thickness', 'medium')).lower()
        valid_thickness = ['thin', 'medium', 'thick', 'very_thick']
        features['thickness'] = thickness if thickness in valid_thickness else 'medium'
        
        # Volume
        volume = str(user_preferences.get('volume', 'medium')).lower()
        features['volume'] = volume if volume in ['low', 'medium', 'high'] else 'medium'
        
        # Wants bangs
        wants_bangs = user_preferences.get('wants_bangs', False)
        features['wants_bangs'] = 'yes' if wants_bangs else 'no'
        
        # Occasions (can be multiple, sorted alphabetically and joined with comma)
        occasions = user_preferences.get('occasions', [])
        if isinstance(occasions, list):
            # Sort and join occasions
            occasions = sorted([str(o).lower() for o in occasions if o])
            features['occasions'] = ','.join(occasions) if occasions else 'casual'
        else:
            features['occasions'] = str(occasions).lower() if occasions else 'casual'
        
        return features
    
    def _normalize_faceshape(self, face_shape: str) -> str:
        """Normalize face shape to model's expected values"""
        if not face_shape:
            return 'oval'
        
        face_shape = face_shape.lower().strip()
        
        # Model expects: heart, oblong, oval, round, square
        valid_shapes = ['heart', 'oblong', 'oval', 'round', 'square']
        
        if face_shape in valid_shapes:
            return face_shape
        
        # Map similar shapes
        shape_mapping = {
            'diamond': 'heart',
            'triangle': 'heart',
            'rectangle': 'oblong',
            'long': 'oblong',
        }
        
        return shape_mapping.get(face_shape, 'oval')
    
    def _create_feature_vector(self, features: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Create one-hot encoded feature vector matching model columns
        
        Args:
            features: Dictionary with feature names and values
            
        Returns:
            DataFrame with one-hot encoded features, or None if encoding fails
        """
        try:
            # Create a DataFrame with all model columns initialized to 0
            feature_vector = pd.DataFrame(0, index=[0], columns=self.model_columns)
            
            # Set one-hot encoded values
            for feature_name, value in features.items():
                # Construct the one-hot encoded column name
                if feature_name in ['faceshape', 'gender']:
                    # Simple prefix
                    column_name = f"{feature_name}_{value}"
                elif feature_name == 'hair_type':
                    column_name = f"hair_type_{value}"
                elif feature_name == 'hair_length':
                    column_name = f"hair_length_{value}"
                elif feature_name == 'hair_color':
                    column_name = f"hair_color_{value}"
                elif feature_name == 'hair_condition':
                    column_name = f"hair_condition_{value}"
                elif feature_name == 'lifestyle':
                    column_name = f"lifestyle_{value}"
                elif feature_name == 'maintenance':
                    column_name = f"maintenance_{value}"
                elif feature_name == 'styling_maintenance':
                    column_name = f"styling_maintenance_{value}"
                elif feature_name == 'styling_preference':
                    column_name = f"styling_preference_{value}"
                elif feature_name == 'texture_detail':
                    column_name = f"texture_detail_{value}"
                elif feature_name == 'thickness':
                    column_name = f"thickness_{value}"
                elif feature_name == 'volume':
                    column_name = f"volume_{value}"
                elif feature_name == 'wants_bangs':
                    column_name = f"wants_bangs_{value}"
                elif feature_name == 'occasions':
                    column_name = f"occasions_{value}"
                else:
                    logger.warning(f"Unknown feature: {feature_name}")
                    continue
                
                # Set the column to 1 if it exists in model columns
                if column_name in feature_vector.columns:
                    feature_vector.at[0, column_name] = 1
                else:
                    # Apply fallback for missing columns
                    fallback_applied = False
                    
                    # If 'natural' color not in model, use 'brown' as fallback
                    if (feature_name == 'hair_color' and value == 'natural'):
                        fallback_col = 'hair_color_brown'
                        if fallback_col in feature_vector.columns:
                            feature_vector.at[0, fallback_col] = 1
                            fallback_applied = True
                            logger.info(
                                f"Applied fallback: {column_name} -> {fallback_col}"
                            )
                    
                    if not fallback_applied:
                        logger.warning(
                            f"Column '{column_name}' not found in model columns. "
                            f"Feature: {feature_name}={value}"
                        )
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature vector creation failed: {e}", exc_info=True)
            return None
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready"""
        return (
            self.rf_model is not None and 
            self.model_columns is not None and
            self.label_encoder is not None
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_available():
            return {'loaded': False, 'error': 'Model not loaded'}
        
        return {
            'loaded': True,
            'n_features': len(self.model_columns),
            'n_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'model_type': type(self.rf_model).__name__,
            'n_estimators': self.rf_model.n_estimators
        }


# Global instance
_hairstyle_model_recommender = None


def get_hairstyle_model_recommender() -> HairstyleModelRecommender:
    """Get or create global hairstyle model recommender instance"""
    global _hairstyle_model_recommender
    if _hairstyle_model_recommender is None:
        _hairstyle_model_recommender = HairstyleModelRecommender()
    return _hairstyle_model_recommender
