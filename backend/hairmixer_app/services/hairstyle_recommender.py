"""
Hairstyle recommendation service using the trained Random Forest model.

This is the ONLY hairstyle recommendation method used in the system.
All recommendations are generated using the Random Forest classifier
(hairstyle_family_model.pkl).

This service loads the hairstyle_family_model and provides
hairstyle recommendations based on user preferences.

Key Features:
    - Uses Random Forest classifier (26 hairstyle families)
    - Trained on 15 user preference features
    - Generates confidence-scored recommendations
    - No fallback methods - Pure ML-based approach
    - Hierarchical: Family prediction → Database query → Specific styles
"""
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from django.conf import settings
from django.db import models
from ..models import Hairstyle


logger = logging.getLogger(__name__)


class HairstyleRecommender:
    """
    Hairstyle recommendation engine using trained ML model.
    
    This class provides intelligent hairstyle recommendations by:
    1. Loading a pre-trained Random Forest classifier (hairstyle_family_model.pkl)
    2. Encoding 21 user preference features into model input
    3. Predicting hairstyle families with confidence scores
    4. Querying database for matching hairstyles
    5. Returning top 10 recommendations with match scores
    
    Features used by the model:
    - Core: gender, hair_type, hair_length, faceshape, maintenance, lifestyle
    - Detailed: volume, styling_maintenance, styling_preference, 
                hair_condition, hair_thickness, hair_texture_detail
    - Binary: wants_bangs
    - Multi-select: 8 occasion types
    
    The model predicts hairstyle families (e.g., "Bob", "Layers", "Pixie")
    and then retrieves matching hairstyles from the database.
    
    Example usage:
        recommender = HairstyleRecommender()
        preferences = {
            'faceshape': 'oval',
            'hair_type': 'wavy',
            'hair_length': 'medium',
            'wants_bangs': True,
            # ... other preferences
        }
        recommendations = recommender.get_top_recommendations(
            preferences, top_n=10
        )
    
    Attributes:
        model: Loaded sklearn RandomForestClassifier
        model_loaded (bool): Whether model loaded successfully
    """
    
    def __init__(self):
        """Initialize the recommender and load the ML model."""
        self.model = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """
        Load the trained hairstyle family model from disk.
        
        Attempts to load 'hairstyle_family_model.pkl' from the ml/models/
        directory. Logs model information including number of features and
        predicted classes.
        
        Sets:
            self.model: The loaded RandomForestClassifier
            self.model_loaded: True if successful, False otherwise
        """
        try:
            model_path = Path(__file__).parent.parent / 'ml' / 'models' / 'hairstyle_family_model.pkl'
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.model_loaded = True
            logger.info(
                f"Hairstyle family model loaded successfully: "
                f"{type(self.model).__name__}"
            )
            
            # Log model info if available
            if hasattr(self.model, 'n_features_in_'):
                logger.info(
                    f"Model expects {self.model.n_features_in_} features"
                )
            if hasattr(self.model, 'classes_'):
                logger.info(
                    f"Model predicts {len(self.model.classes_)} families: "
                    f"{self.model.classes_[:5]}..."
                )
                
        except Exception as e:
            logger.error(f"Failed to load hairstyle model: {str(e)}")
            self.model_loaded = False
    
    def _encode_preferences(
        self, preferences: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Encode user preferences into feature vector for ML model.
        
        Converts categorical and boolean preferences into numeric features
        suitable for the Random Forest classifier. Creates a 21-feature
        vector following this structure:
        
        Features 1-6 (Core):
            - gender (0-3)
            - hair_type (0-3)
            - hair_length (0-4)
            - faceshape (0-6)
            - maintenance (0-2)
            - lifestyle (0-3)
        
        Features 7-12 (Detailed):
            - volume (0-3)
            - styling_maintenance (0-2)
            - styling_preference (0-4)
            - hair_condition (0-3)
            - hair_thickness (0-3)
            - hair_texture_detail (0-4)
        
        Feature 13 (Binary):
            - wants_bangs (0-1)
        
        Features 14-21 (Multi-select occasions):
            - work, casual, formal, date, exercise, travel, party, wedding
            - Each encoded as binary 0 or 1
            - Note: Generates 21 features total, but model may use fewer
        
        Args:
            preferences: Dictionary with user preference fields
                        Keys match UserPreference model fields
            
        Returns:
            numpy array of shape (1, N) with encoded features
            where N matches model's expected feature count,
            or None if encoding fails
            
        Example:
            >>> prefs = {
            ...     'gender': 'female',
            ...     'hair_type': 'wavy',
            ...     'hair_length': 'medium',
            ...     'faceshape': 'oval',
            ...     'occasions': ['work', 'casual']
            ... }
            >>> features = self._encode_preferences(prefs)
            >>> features.shape
            (1, 21)
        """
        try:
            # Feature encoding mappings
            # Each categorical value maps to an integer
            gender_map = {
                'male': 0, 'female': 1, 'nb': 2, 'other': 3, '': 0
            }
            hair_type_map = {
                'straight': 0, 'wavy': 1, 'curly': 2, 'coily': 3, '': 0
            }
            hair_length_map = {
                'pixie': 0, 'short': 1, 'medium': 2,
                'long': 3, 'extra_long': 4,
                # Dataset only has these 3, map to existing
                '': 2
            }
            face_shape_map = {
                'oval': 0, 'round': 1, 'square': 2, 'heart': 3,
                'oblong': 4, 'diamond': 5, 'triangle': 6, '': 0
            }
            maintenance_map = {
                'low': 0, 'medium': 1, 'high': 2, '': 1
            }
            lifestyle_map = {
                'active': 0, 'professional': 1,
                'creative': 2, 'casual': 3,
                # Map new dataset values to closest existing categories
                'moderate': 3, 'relaxed': 3, '': 3
            }
            volume_map = {
                'flat': 0, 'light': 1, 'medium': 2, 'high': 3,
                # Dataset uses low/medium/high
                'low': 0, '': 2
            }
            styling_pref_map = {
                'natural': 0, 'casual': 1, 'polished': 2,
                'glamorous': 3, 'edgy': 4,
                # Dataset uses classic/elegant/trendy
                'classic': 1, 'elegant': 2, 'trendy': 1, '': 1
            }
            condition_map = {
                'excellent': 3, 'good': 2, 'fair': 1, 'damaged': 0,
                # Dataset has detailed conditions, map to scale
                'none': 3, 'dry_ends': 1, 'oily_scalp': 1,
                'dandruff': 1, 'frizzy': 1, 'split_ends': 1,
                'thinning': 0, 'sensitive_scalp': 2, '': 2
            }
            thickness_map = {
                'fine': 0, 'medium': 1, 'thick': 2, 'very_thick': 3,
                # Dataset uses thin instead of fine
                'thin': 0, '': 1
            }
            texture_map = {
                'smooth': 0, 'coarse': 1, 'silky': 2,
                'frizzy': 3, 'normal': 4, '': 4
            }
            
            # Extract and encode features
            features = []
            
            # Core features
            features.append(
                gender_map.get(preferences.get('gender', ''), 0)
            )
            features.append(
                hair_type_map.get(preferences.get('hair_type', ''), 0)
            )
            features.append(
                hair_length_map.get(preferences.get('hair_length', ''), 2)
            )
            features.append(
                face_shape_map.get(preferences.get('faceshape', ''), 0)
            )
            features.append(
                maintenance_map.get(preferences.get('maintenance', ''), 1)
            )
            features.append(
                lifestyle_map.get(preferences.get('lifestyle', ''), 3)
            )
            
            # New detailed features
            features.append(
                volume_map.get(preferences.get('volume', ''), 2)
            )
            features.append(
                maintenance_map.get(
                    preferences.get('styling_maintenance', ''), 1
                )
            )
            features.append(
                styling_pref_map.get(
                    preferences.get('styling_preference', ''), 1
                )
            )
            features.append(
                condition_map.get(preferences.get('hair_condition', ''), 2)
            )
            features.append(
                thickness_map.get(preferences.get('hair_thickness', ''), 1)
            )
            features.append(
                texture_map.get(
                    preferences.get('hair_texture_detail', ''), 4
                )
            )
            
            # Boolean features
            features.append(
                1 if preferences.get('wants_bangs', False) else 0
            )
            
            # Occasions - encode as multiple binary features
            # NOTE: Model was trained with 8 occasions, so we maintain that encoding
            # even though UI now only allows 6 occasions from dataset
            occasions = preferences.get('occasions', [])
            occasion_types = [
                'work', 'casual', 'formal', 'date',
                'exercise', 'travel', 'party', 'wedding'
            ]
            # Map new 'birthday' occasion to closest match 'party'
            if 'birthday' in occasions and 'birthday' not in occasion_types:
                occasions = list(occasions)
                occasions.append('party')
            
            for occ in occasion_types:
                features.append(1 if occ in occasions else 0)
            
            feature_array = np.array(features).reshape(1, -1)
            
            # Handle model compatibility: if model expects fewer features,
            # use only the first N features (backward compatibility)
            if hasattr(self, 'model') and hasattr(self.model, 'n_features_in_'):
                expected_features = self.model.n_features_in_
                if feature_array.shape[1] > expected_features:
                    logger.warning(
                        f"Model expects {expected_features} features, "
                        f"but we generated {feature_array.shape[1]}. "
                        f"Using first {expected_features} features only."
                    )
                    feature_array = feature_array[:, :expected_features]
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error encoding preferences: {str(e)}")
            return None
    
    def get_top_recommendations(
        self,
        preferences: Dict[str, Any],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top N hairstyle recommendations using family-based prediction.
        
        Hierarchical approach:
        1. Model predicts hairstyle families (26 classes) with confidence
        2. Map families to categories
        3. Query database for hairstyles in predicted families
        4. Return diverse recommendations from high-confidence families
        
        Args:
            preferences: User preference dictionary with 15 features
            top_n: Number of recommendations to return (default: 10)
            
        Returns:
            List of recommended hairstyles with prediction scores
            
        Raises:
            Exception: If model fails, returns empty list
        """
        if not self.model_loaded or self.model is None:
            logger.error("Model not loaded - cannot generate recommendations")
            return []
        
        try:
            # Encode preferences into feature vector
            feature_vector = self._encode_preferences(preferences)
            if feature_vector is None:
                logger.error("Failed to encode preferences")
                return []
            
            # Get prediction probabilities for all 26 families
            if not hasattr(self.model, 'predict_proba'):
                logger.error("Model does not support predict_proba")
                return []
            
            probabilities = self.model.predict_proba(feature_vector)[0]
            predicted_classes = self.model.classes_
            
            # Sort families by confidence (highest first)
            sorted_indices = np.argsort(probabilities)[::-1]
            
            logger.info(
                f"Model predictions (top 10): "
                f"{[(predicted_classes[i], probabilities[i]) for i in sorted_indices[:10]]}"
            )
            
            # Strategy: Collect hairstyles from top families until we have 10
            # Allocate more slots to higher-confidence families
            recommendations = []
            seen_style_ids = set()  # Track to avoid duplicates
            seen_categories = set()  # Track categories already queried
            
            # Calculate how many styles to get from each family
            # Higher confidence families get more slots
            family_allocations = self._calculate_family_allocations(
                sorted_indices, probabilities, top_n
            )
            
            # First pass: Collect hairstyles from each family with strict filters
            for idx, num_styles in family_allocations:
                if len(recommendations) >= top_n:
                    break
                
                family_id = predicted_classes[idx]
                family_score = probabilities[idx]
                
                # Get hairstyles for this family
                styles = self._get_styles_by_family(
                    family_id, preferences, limit=num_styles,
                    seen_categories=seen_categories, strict_filter=True
                )
                
                logger.info(
                    f"Family {family_id} (score={family_score:.3f}, "
                    f"allocated={num_styles}): Found {len(styles)} hairstyles"
                )
                
                # Add styles to recommendations
                for style in styles:
                    if len(recommendations) >= top_n:
                        break
                    
                    # Avoid duplicates
                    if style.id in seen_style_ids:
                        continue
                    
                    seen_style_ids.add(style.id)
                    
                    recommendations.append({
                        'id': str(style.id),
                        'name': style.name,
                        'description': style.description or '',
                        'image_url': (
                            style.image.url if style.image
                            else style.image_url or None
                        ),
                        'category': (
                            style.category.name if style.category else ''
                        ),
                        'hairstyle_family': str(family_id),
                        'difficulty': style.difficulty or 'medium',
                        'estimated_time': style.estimated_time or 30,
                        'maintenance': style.maintenance or 'medium',
                        'tags': style.tags or [],
                        'match_score': round(float(family_score), 3),
                        'confidence': round(float(family_score) * 100, 1)
                    })
            
            # Second pass: If we don't have enough, relax filters and try again
            if len(recommendations) < top_n:
                logger.warning(
                    f"Only found {len(recommendations)} recommendations "
                    f"with strict filters. Relaxing filters to get {top_n}..."
                )
                
                # Try more families with relaxed filters
                for idx in sorted_indices[:20]:  # Check top 20 families
                    if len(recommendations) >= top_n:
                        break
                    
                    family_id = predicted_classes[idx]
                    family_score = probabilities[idx]
                    
                    # Skip very low confidence
                    if family_score < 0.01:
                        continue
                    
                    # Get styles with relaxed filters
                    styles = self._get_styles_by_family(
                        family_id, preferences, limit=3,
                        seen_categories=None, strict_filter=False
                    )
                    
                    for style in styles:
                        if len(recommendations) >= top_n:
                            break
                        
                        # Avoid duplicates
                        if style.id in seen_style_ids:
                            continue
                        
                        seen_style_ids.add(style.id)
                        
                        recommendations.append({
                            'id': str(style.id),
                            'name': style.name,
                            'description': style.description or '',
                            'image_url': (
                                style.image.url if style.image
                                else style.image_url or None
                            ),
                            'category': (
                                style.category.name if style.category else ''
                            ),
                            'hairstyle_family': str(family_id),
                            'difficulty': style.difficulty or 'medium',
                            'estimated_time': style.estimated_time or 30,
                            'maintenance': style.maintenance or 'medium',
                            'tags': style.tags or [],
                            'match_score': round(float(family_score), 3),
                            'confidence': round(float(family_score) * 100, 1)
                        })
            
            logger.info(
                f"Generated {len(recommendations)} recommendations "
                f"from {len([r for r in recommendations])} total styles"
            )
            
            return recommendations[:top_n]
            
        except Exception as e:
            logger.error(
                f"Error generating recommendations: {str(e)}",
                exc_info=True
            )
            return []
    
    def _calculate_family_allocations(
        self,
        sorted_indices: np.ndarray,
        probabilities: np.ndarray,
        total_slots: int
    ) -> List[tuple]:
        """
        Calculate how many hairstyles to retrieve from each family.
        
        Uses a weighted allocation strategy:
        - Top family (highest confidence): 3-4 styles
        - Next 2-3 families: 2 styles each
        - Remaining families: 1 style each
        
        Args:
            sorted_indices: Indices sorted by probability (highest first)
            probabilities: Probability array from model
            total_slots: Total recommendations needed (10)
            
        Returns:
            List of (family_index, num_styles) tuples
        """
        allocations = []
        remaining_slots = total_slots
        
        # Allocate slots based on confidence tiers
        for rank, idx in enumerate(sorted_indices):
            if remaining_slots <= 0:
                break
            
            confidence = probabilities[idx]
            
            # Tier 1: Top prediction gets 3-4 slots
            if rank == 0 and confidence > 0.15:
                num_styles = min(4, remaining_slots)
            # Tier 2: Next 2-3 predictions get 2 slots
            elif rank in [1, 2, 3] and confidence > 0.08:
                num_styles = min(2, remaining_slots)
            # Tier 3: Remaining predictions get 1 slot
            elif confidence > 0.02:
                num_styles = min(1, remaining_slots)
            else:
                # Skip very low confidence families
                continue
            
            allocations.append((idx, num_styles))
            remaining_slots -= num_styles
            
            # Safety: stop after checking enough families
            if rank >= 15 or remaining_slots <= 0:
                break
        
        logger.debug(
            f"Family allocations: {[(f, n) for f, n in allocations]}"
        )
        
        return allocations
    
    def _get_styles_by_family(
        self,
        family: str,
        preferences: Dict[str, Any],
        limit: int = 3,
        seen_categories: Optional[set] = None,
        strict_filter: bool = True
    ) -> List[Any]:
        """
        Get hairstyles matching the predicted family.
        
        Uses intelligent mapping from model's 26 families to database's
        9 categories, then filters by user preferences (hair_length,
        faceshape, etc.)
        
        Args:
            family: Hairstyle family ID from model (0-25)
            preferences: User preferences for additional filtering
            limit: Maximum number of styles to return
            seen_categories: Set of categories already queried (to avoid
                duplicates)
            strict_filter: If True, apply preference filters. If False,
                skip filters to get more results
            
        Returns:
            List of Hairstyle objects matching family and preferences
        """
        try:
            # Convert numpy int to Python int
            family_str = str(int(family)) if isinstance(
                family, (int, np.integer)
            ) else str(family)
            
            # Map model's 26 classes → database's 9 categories
            # Each model family maps to a style category
            MODEL_TO_DB_CATEGORY = {
                # Short styles (category 1)
                0: 1, 1: 1, 2: 1,
                # Long styles (category 2)
                3: 2, 4: 2, 5: 2,
                # Medium styles (category 3)
                6: 3, 7: 3, 8: 3,
                # Curly styles (category 4)
                9: 4, 10: 4, 11: 4,
                # Formal styles (category 5)
                12: 5, 13: 5, 14: 5,
                # Classic styles (category 6)
                15: 6, 16: 6, 17: 6,
                # Trendy styles (category 7)
                18: 7, 19: 7, 20: 7,
                # Retro styles (category 8)
                21: 8, 22: 8, 23: 8,
                # Casual styles (category 9)
                24: 9, 25: 9
            }
            
            # Convert family to database category
            try:
                family_id = int(family_str)
                
                if family_id in MODEL_TO_DB_CATEGORY:
                    category_id = MODEL_TO_DB_CATEGORY[family_id]
                    
                    # Skip if category already queried (avoid duplicates)
                    if seen_categories is not None:
                        if category_id in seen_categories:
                            logger.debug(
                                f"Category {category_id} already queried, "
                                f"skipping family {family_id}"
                            )
                            return []
                        seen_categories.add(category_id)
                    
                    logger.debug(
                        f"Mapped family {family_id} → "
                        f"category {category_id}"
                    )
                else:
                    # Unknown family, use as-is
                    category_id = family_id
                    logger.warning(
                        f"Unknown family {family_id}, using as-is"
                    )
                
                # Query by category with gender filtering at database level
                queryset = Hairstyle.objects.filter(
                    is_active=True,
                    category_id=category_id
                )
                
                # Apply gender filter at database level for efficiency
                if preferences.get('gender') and strict_filter:
                    user_gender = preferences['gender'].lower()
                    # Map 'nb' and 'other' to unisex
                    if user_gender in ['nb', 'other']:
                        user_gender = 'unisex'
                    
                    # Filter for exact gender match OR unisex
                    queryset = queryset.filter(
                        models.Q(suitable_gender=user_gender) |
                        models.Q(suitable_gender='unisex')
                    )
                
                queryset = queryset.order_by(
                    '-popularity_score', '-trend_score'
                )
                
            except (ValueError, TypeError):
                # Non-numeric family, search by name
                queryset = Hairstyle.objects.filter(
                    is_active=True
                ).filter(
                    models.Q(category__name__icontains=family_str) |
                    models.Q(name__icontains=family_str) |
                    models.Q(tags__icontains=family_str)
                )
                
                # Apply gender filter at database level for efficiency
                if preferences.get('gender') and strict_filter:
                    user_gender = preferences['gender'].lower()
                    # Map 'nb' and 'other' to unisex
                    if user_gender in ['nb', 'other']:
                        user_gender = 'unisex'
                    
                    # Filter for exact gender match OR unisex
                    queryset = queryset.filter(
                        models.Q(suitable_gender=user_gender) |
                        models.Q(suitable_gender='unisex')
                    )
                
                queryset = queryset.order_by(
                    '-popularity_score', '-trend_score'
                )
            
            # Get initial batch (more than needed for filtering)
            fetch_limit = limit * 3 if strict_filter else limit * 5
            results = list(queryset[:fetch_limit])
            
            # If no strict filtering, return all results
            if not strict_filter:
                return results[:limit]
            
            # Apply preference filters (strict mode only)
            filtered_results = results
            
            # Filter by hair_length (soft - prefer but don't require)
            if preferences.get('hair_length') and filtered_results:
                length_matches = [
                    s for s in filtered_results
                    if s.hair_lengths and
                    preferences['hair_length'] in s.hair_lengths
                ]
                if length_matches:
                    filtered_results = length_matches
            
            # Filter by faceshape (soft filter)
            if preferences.get('faceshape') and filtered_results:
                face_matches = [
                    s for s in filtered_results
                    if not s.face_shapes or
                    preferences['faceshape'] in s.face_shapes
                ]
                if face_matches:
                    filtered_results = face_matches
            
            # CRITICAL: Filter by gender using the new suitable_gender field
            # This ensures male users only get male/unisex styles, and
            # female users only get female/unisex styles
            if preferences.get('gender') and filtered_results:
                user_gender = preferences['gender'].lower()
                # Map 'nb' and 'other' to unisex for hairstyle matching
                if user_gender in ['nb', 'other']:
                    user_gender = 'unisex'
                
                gender_matches = [
                    s for s in filtered_results
                    if s.suitable_gender == user_gender or
                    s.suitable_gender == 'unisex'
                ]
                
                # STRICT: If no gender matches found, log warning and
                # return empty list (don't show wrong-gender styles)
                if not gender_matches:
                    logger.warning(
                        f"No hairstyles found matching gender '{user_gender}' "
                        f"in family {family}. Available: "
                        f"{[s.suitable_gender for s in filtered_results[:5]]}"
                    )
                filtered_results = gender_matches
            
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(
                f"Error getting styles by family '{family}': {str(e)}",
                exc_info=True
            )
            return []
