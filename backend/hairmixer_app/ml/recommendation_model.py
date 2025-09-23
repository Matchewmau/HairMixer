import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class MLRecommendationEngine:
    """Machine Learning based recommendation engine"""
    
    def __init__(self):
        self.model = None
        self.feature_encoders = {}
        self.is_trained = False
        
    def prepare_training_data(self):
        """Prepare training data from user feedback and interactions"""
        from ..models import Feedback
        
        # Get feedback data
        feedbacks = Feedback.objects.select_related(
            'recommendation', 'hairstyle'
        ).filter(rating__isnull=False)
        
        features = []
        targets = []
        
        for feedback in feedbacks:
            try:
                rec = feedback.recommendation
                style = feedback.hairstyle
                
                # Extract features
                feature_vector = self._extract_features(
                    rec.face_shape,
                    rec.detected_features,
                    style,
                    rec.preference
                )
                
                features.append(feature_vector)
                targets.append(feedback.rating)
                
            except Exception as e:
                logger.error(
                    "Error processing feedback %s: %s",
                    feedback.id, str(e)
                )
                continue
        
        return np.array(features), np.array(targets)
    
    def train_model(self):
        """Train the recommendation model"""
        try:
            # Prepare data
            X, y = self.prepare_training_data()
            
            if len(X) < 50:  # Need minimum data
                logger.warning(
                    "Insufficient training data, using rule-based system"
                )
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            logger.info(
                "Model trained - Train: %.3f Test: %.3f",
                train_score,
                test_score,
            )
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def predict_user_preference(
        self, face_shape, facial_features, hairstyle, user_preferences
    ):
        """Predict how much a user would like a hairstyle"""
        if not self.is_trained or self.model is None:
            # Fallback to rule-based scoring
            return self._rule_based_scoring(
                face_shape, facial_features, hairstyle, user_preferences
            )
        
        try:
            # Extract features
            feature_vector = self._extract_features(
                face_shape, facial_features, hairstyle, user_preferences
            )
            
            # Predict
            prediction = self.model.predict([feature_vector])[0]
            
            # Convert to 0-1 scale and add confidence
            # map 1-5 rating to 0-1 range
            normalized_score = max(0, min(1, (prediction - 1) / 4))
            
            return {
                'score': normalized_score,
                'confidence': 0.8,  # Model confidence
                'method': 'ml_prediction'
            }
            
        except Exception as e:
            logger.error("Error in ML prediction: %s", str(e))
            return self._rule_based_scoring(
                face_shape, facial_features, hairstyle, user_preferences
            )
    
    def _extract_features(
        self, face_shape, facial_features, hairstyle, preferences
    ):
        """Extract numerical features for ML model"""
        features = []
        
        # Face shape encoding
        face_shape_encoded = self._encode_categorical('face_shape', face_shape)
        features.extend(face_shape_encoded)
        
        # Facial measurements (if available)
        if facial_features:
            features.extend([
                facial_features.get('face_ratio', 1.0),
                facial_features.get('jawline_width', 100),
                facial_features.get('forehead_width', 100),
                facial_features.get('cheekbone_width', 100),
                facial_features.get('symmetry_score', 0.8)
            ])
        else:
            features.extend([1.0, 100, 100, 100, 0.8])  # Default values
        
        # Hairstyle features
        if hairstyle:
            # Hair length encoding
            length_encoded = self._encode_categorical(
                'hair_length',
                hairstyle.hair_lengths[0]
                if hairstyle.hair_lengths else 'medium'
            )
            features.extend(length_encoded)
            
            # Hair type encoding
            type_encoded = self._encode_categorical(
                'hair_type',
                hairstyle.hair_types[0]
                if hairstyle.hair_types else 'straight'
            )
            features.extend(type_encoded)
            
            # Maintenance level
            maintenance_encoded = self._encode_categorical(
                'maintenance', hairstyle.maintenance or 'medium'
            )
            features.extend(maintenance_encoded)
            
            # Style scores
            features.extend([
                hairstyle.trend_score or 0,
                hairstyle.popularity_score or 0
            ])
        
        # User preferences
        if preferences:
            pref_length = self._encode_categorical(
                'hair_length', preferences.hair_length or 'medium'
            )
            features.extend(pref_length)
            
            pref_type = self._encode_categorical(
                'hair_type', preferences.hair_type or 'straight'
            )
            features.extend(pref_type)
            
            pref_maintenance = self._encode_categorical(
                'maintenance', preferences.maintenance or 'medium'
            )
            features.extend(pref_maintenance)
        
        return features
    
    def _encode_categorical(self, category, value):
        """Encode categorical variables"""
        categories = {
            'face_shape': [
                'oval', 'round', 'square', 'heart', 'diamond', 'oblong',
                'triangle'
            ],
            'hair_length': ['pixie', 'short', 'medium', 'long', 'extra_long'],
            'hair_type': ['straight', 'wavy', 'curly', 'coily'],
            'maintenance': ['low', 'medium', 'high']
        }
        
        if category not in categories:
            return [0]
        
        # Minor normalization then one-hot encoding
        try:
            norm_value = str(value).strip().lower()
        except Exception:
            norm_value = ''
        cat_list = categories[category]
        encoded = [1 if norm_value == cat else 0 for cat in cat_list]
        
        return encoded

    def _rule_based_scoring(
        self, face_shape, facial_features, hairstyle, preferences
    ):
        """Simple compatibility scoring used when ML model isn't available."""
        try:
            score = 0.0
            # Face shape compatibility
            if face_shape and hasattr(hairstyle, 'face_shapes'):
                shapes = hairstyle.face_shapes or []
                fs = str(face_shape).strip().lower()
                if fs in shapes:
                    score += 30
                else:
                    # loose compatibility
                    compat = {
                        'oval': ['round', 'square', 'heart', 'triangle'],
                        'round': ['oval', 'heart', 'triangle'],
                        'square': ['oval', 'diamond', 'triangle'],
                        'heart': ['oval', 'round', 'diamond'],
                        'diamond': ['heart', 'square'],
                        'oblong': ['oval', 'square'],
                        'triangle': ['oval', 'square']
                    }
                    if fs in compat and any(s in compat[fs] for s in shapes):
                        score += 20
            # Hair type
            if (
                preferences and preferences.hair_type
                and hasattr(hairstyle, 'hair_types')
            ):
                if preferences.hair_type in (hairstyle.hair_types or []):
                    score += 20
            # Hair length
            if (
                preferences and preferences.hair_length
                and hasattr(hairstyle, 'hair_lengths')
            ):
                if preferences.hair_length in (hairstyle.hair_lengths or []):
                    score += 20
            # Maintenance
            if (
                preferences and preferences.maintenance
                and getattr(hairstyle, 'maintenance', None)
            ):
                if preferences.maintenance == hairstyle.maintenance:
                    score += 15
            # Occasions overlap
            try:
                pref_occasions = set(preferences.occasions or [])
                style_occasions = set((hairstyle.occasions or []))
                if pref_occasions and style_occasions:
                    score += min(len(pref_occasions & style_occasions) * 5, 10)
            except Exception:
                pass
            # Trend/popularity
            score += float(getattr(hairstyle, 'trend_score', 0) or 0) * 0.5
            score += (
                float(getattr(hairstyle, 'popularity_score', 0) or 0) * 0.5
            )

            # Normalize to 0-1 range roughly
            normalized = max(0.0, min(1.0, score / 100.0))
            return {
                'score': normalized,
                'confidence': 0.5,
                'method': 'rule_based'
            }
        except Exception as e:
            logger.error(f"Rule-based scoring error: {str(e)}")
            return {'score': 0.0, 'confidence': 0.0, 'method': 'error'}
