"""
Random Forest-based Hairstyle Recommender (No-Family Model)

This module provides hairstyle recommendations using a Random Forest classifier
that predicts specific hairstyle names based on 15 user preference features.

Model Files:
    - rf_name_no_family.pkl: Trained Random Forest classifier
    - label_encoders_no_family.pkl: LabelEncoders for categorical features
    - feature_columns_no_family.pkl: Feature names in correct order
    - model_metadata_no_family.json: Model performance metrics

Features (15 total):
    1. faceshape, 2. gender, 3. hair_type, 4. hair_length, 5. hair_color,
    6. volume, 7. thickness, 8. texture_detail, 9. lifestyle, 10. maintenance,
    11. styling_maintenance, 12. styling_preference, 13. occasions,
    14. wants_bangs, 15. hair_condition
"""

import pickle
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class HairstyleRecommender:
    """Random Forest-based hairstyle recommender (No-Family Model)"""
    
    def __init__(self):
        """Initialize recommender and load model files"""
        self.rf_model = None
        self.label_encoders = None
        self.feature_columns = None
        self.metadata = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained Random Forest model and encoders"""
        try:
            models_path = (
                Path(settings.BASE_DIR) / 'hairmixer_app' / 'ml' / 
                'models' / 'hairstyle_recommender'
            )
            
            # Verify directory exists
            if not models_path.exists():
                logger.error(f"Model directory not found: {models_path}")
                return
            
            # Load Random Forest model
            model_file = models_path / 'rf_name_no_family.pkl'
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return
                
            with open(model_file, 'rb') as f:
                self.rf_model = pickle.load(f)
            logger.info("✓ Loaded Random Forest model (No-Family)")
            
            # Load label encoders
            encoders_file = models_path / 'label_encoders_no_family.pkl'
            if not encoders_file.exists():
                logger.error(f"Encoders file not found: {encoders_file}")
                self.rf_model = None
                return
                
            with open(encoders_file, 'rb') as f:
                self.label_encoders = pickle.load(f)
            logger.info(f"✓ Loaded {len(self.label_encoders)} label encoders")
            
            # Load feature columns
            features_file = models_path / 'feature_columns_no_family.pkl'
            if not features_file.exists():
                logger.error(f"Features file not found: {features_file}")
                self.rf_model = None
                return
                
            with open(features_file, 'rb') as f:
                self.feature_columns = pickle.load(f)
            logger.info(f"✓ Loaded {len(self.feature_columns)} feature columns")
            
            # Load metadata
            metadata_file = models_path / 'model_metadata_no_family.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(
                    f"✓ Model metadata: {self.metadata.get('n_classes', 'N/A')} classes, "
                    f"{self.metadata.get('top_10_accuracy', 0)*100:.1f}% Top-10 accuracy"
                )
            else:
                logger.warning("Model metadata file not found")
            
        except Exception as e:
            logger.error(f"Failed to load hairstyle recommender model: {e}", exc_info=True)
            self.rf_model = None
            self.label_encoders = None
            self.feature_columns = None
    
    def predict_top_k(
        self, 
        user_preferences: Dict[str, Any], 
        face_shape: str, 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Predict top K hairstyle recommendations
        
        Args:
            user_preferences: Dict with user preference fields matching model training
            face_shape: Detected face shape from face analyzer
            k: Number of recommendations to return (default 10)
            
        Returns:
            List of dicts with keys: hairstyle_name, confidence, rank
            
        Example:
            >>> prefs = {
            ...     'gender': 'female',
            ...     'hair_type': 'wavy',
            ...     'hair_length': 'medium',
            ...     'hair_color': 'brown',
            ...     'lifestyle': 'active',
            ...     'maintenance': 'low'
            ... }
            >>> results = recommender.predict_top_k(prefs, 'oval', k=10)
        """
        if self.rf_model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Prepare input features in correct order
            feature_values = {}
            
            # Map face_shape to 'faceshape' feature
            feature_values['faceshape'] = face_shape.lower() if face_shape else 'oval'
            
            # Map from UserPreference model fields to model features
            # Direct mappings (same field names)
            direct_mapping = [
                'gender', 'hair_type', 'hair_length', 'hair_color',
                'lifestyle', 'maintenance', 'volume'
            ]
            
            for field in direct_mapping:
                value = user_preferences.get(field, '')
                feature_values[field] = str(value).lower() if value else ''
            
            # Handle thickness mapping (UserPreference has 'hair_thickness')
            thickness = user_preferences.get('hair_thickness', 'medium')
            feature_values['thickness'] = str(thickness).lower() if thickness else 'medium'
            
            # Handle texture_detail mapping (UserPreference has 'hair_texture_detail')
            texture = user_preferences.get('hair_texture_detail', 'normal')
            feature_values['texture_detail'] = str(texture).lower() if texture else 'normal'
            
            # Handle styling fields
            styling_maint = user_preferences.get('styling_maintenance', 'low')
            feature_values['styling_maintenance'] = str(styling_maint).lower() if styling_maint else 'low'
            
            styling_pref = user_preferences.get('styling_preference', 'natural')
            feature_values['styling_preference'] = str(styling_pref).lower() if styling_pref else 'natural'
            
            # Handle occasions - take first if list, otherwise use as-is
            occasions = user_preferences.get('occasions', [])
            if isinstance(occasions, list):
                feature_values['occasions'] = occasions[0].lower() if occasions else 'casual'
            else:
                feature_values['occasions'] = str(occasions).lower() if occasions else 'casual'
            
            # Handle wants_bangs - convert to string
            wants_bangs = user_preferences.get('wants_bangs', False)
            if isinstance(wants_bangs, bool):
                feature_values['wants_bangs'] = 'true' if wants_bangs else 'false'
            else:
                feature_values['wants_bangs'] = str(wants_bangs).lower()
            
            # Handle hair_condition
            condition = user_preferences.get('hair_condition', 'none')
            feature_values['hair_condition'] = str(condition).lower() if condition else 'none'
            
            # Encode features
            X = self._encode_features(feature_values)
            
            if X is None:
                logger.error("Feature encoding failed")
                return []
            
            # Get predictions
            probabilities = self.rf_model.predict_proba(X)[0]
            
            # Get top K indices
            top_k_indices = np.argsort(probabilities)[-k:][::-1]
            
            # Decode hairstyle names
            recommendations = []
            hairstyle_name_encoder = self.label_encoders.get('hairstyle_name')
            
            if hairstyle_name_encoder is None:
                logger.error("hairstyle_name encoder not found")
                return []
            
            for idx in top_k_indices:
                try:
                    hairstyle_name = hairstyle_name_encoder.inverse_transform([idx])[0]
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
                f"face_shape={face_shape}, gender={feature_values.get('gender', 'N/A')}"
            )
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}", exc_info=True)
            return []
    
    def _encode_features(self, feature_values: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Encode feature values using label encoders
        
        Args:
            feature_values: Dict mapping feature names to their values
            
        Returns:
            numpy array of shape (1, 15) with encoded features, or None if encoding fails
        """
        try:
            # Feature names in the order expected by the model
            # Must match the order used during training
            feature_names = [
                'faceshape', 'gender', 'hair_type', 'hair_length', 'hair_color',
                'volume', 'thickness', 'texture_detail', 'lifestyle', 'maintenance',
                'styling_maintenance', 'styling_preference', 'occasions', 
                'wants_bangs', 'hair_condition'
            ]
            
            encoded = []
            
            for feature in feature_names:
                value = feature_values.get(feature, '')
                
                # Get the encoder for this feature
                encoder = self.label_encoders.get(feature)
                
                if encoder is None:
                    logger.warning(f"No encoder found for feature: {feature}")
                    encoded.append(0)
                    continue
                
                # Handle missing or empty values
                if not value or value == '':
                    # Use first class as default
                    if len(encoder.classes_) > 0:
                        value = encoder.classes_[0]
                    else:
                        logger.warning(f"Encoder for {feature} has no classes")
                        encoded.append(0)
                        continue
                
                # Try to encode the value
                try:
                    # Check if value exists in encoder classes
                    if value not in encoder.classes_:
                        logger.warning(
                            f"Value '{value}' not in encoder classes for {feature}. "
                            f"Available: {list(encoder.classes_)[:5]}... "
                            f"Using default: {encoder.classes_[0]}"
                        )
                        value = encoder.classes_[0]
                    
                    encoded_value = encoder.transform([value])[0]
                    encoded.append(encoded_value)
                    
                except Exception as e:
                    logger.warning(
                        f"Error encoding {feature}={value}: {e}. Using 0."
                    )
                    encoded.append(0)
            
            return np.array([encoded])
            
        except Exception as e:
            logger.error(f"Feature encoding failed: {e}", exc_info=True)
            return None
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready"""
        return (
            self.rf_model is not None and 
            self.label_encoders is not None and
            self.feature_columns is not None
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_available():
            return {'loaded': False, 'error': 'Model not loaded'}
        
        return {
            'loaded': True,
            'n_features': len(self.feature_columns) if self.feature_columns else 0,
            'n_classes': len(self.label_encoders.get('hairstyle_name', {}).classes_) if self.label_encoders else 0,
            'metadata': self.metadata or {},
            'model_type': type(self.rf_model).__name__
        }


# Global instance
_hairstyle_recommender = None


def get_hairstyle_recommender() -> HairstyleRecommender:
    """Get or create global hairstyle recommender instance"""
    global _hairstyle_recommender
    if _hairstyle_recommender is None:
        _hairstyle_recommender = HairstyleRecommender()
    return _hairstyle_recommender
