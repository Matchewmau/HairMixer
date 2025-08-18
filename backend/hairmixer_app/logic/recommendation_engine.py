import logging
import math
from typing import List, Dict, Any
from django.db.models import Avg, Count, Q
from django.utils import timezone
from datetime import timedelta

logger = logging.getLogger(__name__)

class EnhancedRecommendationEngine:
    """Advanced recommendation engine with multiple scoring algorithms"""
    
    def __init__(self):
        self.weights = {
            'face_shape': 0.3,
            'occasion': 0.2,
            'hair_type': 0.15,
            'hair_length': 0.15,
            'maintenance': 0.1,
            'lifestyle': 0.05,
            'trend_score': 0.03,
            'popularity': 0.02
        }
    
    def get_recommendations(self, face_shape: str, preferences, all_styles, facial_features: Dict = None, limit: int = 10):
        """Get ranked hairstyle recommendations"""
        try:
            scored_styles = []
            
            for style in all_styles:
                score = self._calculate_style_score(
                    style, face_shape, preferences, facial_features
                )
                
                if score > 0:  # Only include styles with positive scores
                    scored_styles.append({
                        'hairstyle': style,
                        'score': score,
                        'breakdown': self._get_score_breakdown(
                            style, face_shape, preferences, facial_features
                        )
                    })
            
            # Sort by score (descending)
            scored_styles.sort(key=lambda x: x['score'], reverse=True)
            
            # Apply diversity filter to avoid too similar styles
            diverse_styles = self._apply_diversity_filter(scored_styles[:limit*2], limit)
            
            logger.info(f"Generated {len(diverse_styles)} recommendations for face shape: {face_shape}")
            return diverse_styles[:limit]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _calculate_style_score(self, style, face_shape: str, preferences, facial_features: Dict = None):
        """Calculate comprehensive score for a hairstyle"""
        try:
            score = 0.0
            
            # Face shape compatibility (highest weight)
            if face_shape in (style.face_shapes or []):
                score += self.weights['face_shape'] * 100
            elif self._is_compatible_face_shape(face_shape, style.face_shapes or []):
                score += self.weights['face_shape'] * 70
            
            # Occasion matching
            occasion_match = len(set(preferences.occasions or []) & set(style.occasions or []))
            if occasion_match > 0:
                score += self.weights['occasion'] * (occasion_match * 30)
            
            # Hair type compatibility
            if preferences.hair_type in (style.hair_types or []):
                score += self.weights['hair_type'] * 80
            elif self._is_compatible_hair_type(preferences.hair_type, style.hair_types or []):
                score += self.weights['hair_type'] * 50
            
            # Hair length compatibility
            if preferences.hair_length in (style.hair_lengths or []):
                score += self.weights['hair_length'] * 80
            elif self._is_compatible_length(preferences.hair_length, style.hair_lengths or []):
                score += self.weights['hair_length'] * 60
            
            # Maintenance level matching
            if preferences.maintenance == style.maintenance:
                score += self.weights['maintenance'] * 100
            elif self._is_compatible_maintenance(preferences.maintenance, style.maintenance):
                score += self.weights['maintenance'] * 60
            
            # Lifestyle compatibility
            if preferences.lifestyle and preferences.lifestyle in (style.tags or []):
                score += self.weights['lifestyle'] * 70
            
            # Trend and popularity scores
            score += self.weights['trend_score'] * (style.trend_score or 0)
            score += self.weights['popularity'] * (style.popularity_score or 0)
            
            # Facial feature bonuses
            if facial_features:
                score += self._get_facial_feature_bonus(style, facial_features)
            
            # Penalize styles user wants to avoid
            if hasattr(preferences, 'avoid_styles') and preferences.avoid_styles:
                for avoid_tag in preferences.avoid_styles:
                    if avoid_tag in (style.tags or []):
                        score *= 0.5  # 50% penalty
            
            return max(0, score)  # Ensure non-negative score
            
        except Exception as e:
            logger.error(f"Error calculating style score for {style.name}: {str(e)}")
            return 0
    
    def _is_compatible_face_shape(self, user_face_shape: str, style_face_shapes: List[str]):
        """Check if face shapes are compatible even if not exact match"""
        compatibility_map = {
            'oval': ['round', 'square', 'heart'],
            'round': ['oval', 'heart'],
            'square': ['oval', 'diamond'],
            'heart': ['oval', 'round', 'diamond'],
            'diamond': ['heart', 'square'],
            'oblong': ['oval', 'square']
        }
        
        compatible_shapes = compatibility_map.get(user_face_shape, [])
        return any(shape in style_face_shapes for shape in compatible_shapes)
    
    def _is_compatible_hair_type(self, user_hair_type: str, style_hair_types: List[str]):
        """Check hair type compatibility"""
        compatibility_map = {
            'straight': ['wavy'],
            'wavy': ['straight', 'curly'],
            'curly': ['wavy', 'coily'],
            'coily': ['curly']
        }
        
        compatible_types = compatibility_map.get(user_hair_type, [])
        return any(hair_type in style_hair_types for hair_type in compatible_types)
    
    def _is_compatible_length(self, user_length: str, style_lengths: List[str]):
        """Check length compatibility"""
        length_order = ['pixie', 'short', 'medium', 'long', 'extra_long']
        
        if user_length not in length_order or not style_lengths:
            return False
        
        user_idx = length_order.index(user_length)
        
        # Allow styles within 1 step of user's preference
        compatible_indices = [max(0, user_idx-1), user_idx, min(len(length_order)-1, user_idx+1)]
        compatible_lengths = [length_order[i] for i in compatible_indices]
        
        return any(length in style_lengths for length in compatible_lengths)
    
    def _is_compatible_maintenance(self, user_maintenance: str, style_maintenance: str):
        """Check maintenance compatibility"""
        if not style_maintenance:
            return False
        
        maintenance_order = ['low', 'medium', 'high']
        
        try:
            user_idx = maintenance_order.index(user_maintenance)
            style_idx = maintenance_order.index(style_maintenance)
            
            # Allow maintenance levels within 1 step
            return abs(user_idx - style_idx) <= 1
        except ValueError:
            return False
    
    def _get_facial_feature_bonus(self, style, facial_features: Dict):
        """Calculate bonus points based on facial features"""
        bonus = 0
        
        # Jawline considerations
        jawline = facial_features.get('jawline', {})
        if jawline.get('strength') == 'strong' and 'soft' in (style.tags or []):
            bonus += 5
        elif jawline.get('strength') == 'soft' and 'angular' in (style.tags or []):
            bonus += 5
        
        # Age considerations
        age = facial_features.get('age_estimate', 25)
        if age > 50 and 'mature' in (style.tags or []):
            bonus += 3
        elif age < 25 and 'youthful' in (style.tags or []):
            bonus += 3
        
        return bonus * 0.02  # Small bonus relative to main scores
    
    def _get_score_breakdown(self, style, face_shape: str, preferences, facial_features: Dict = None):
        """Get detailed breakdown of scoring for transparency"""
        breakdown = {
            'face_shape_score': 0,
            'occasion_score': 0,
            'hair_type_score': 0,
            'hair_length_score': 0,
            'maintenance_score': 0,
            'trend_score': style.trend_score or 0,
            'popularity_score': style.popularity_score or 0
        }
        
        # Calculate individual component scores
        if face_shape in (style.face_shapes or []):
            breakdown['face_shape_score'] = self.weights['face_shape'] * 100
        
        occasion_match = len(set(preferences.occasions or []) & set(style.occasions or []))
        if occasion_match > 0:
            breakdown['occasion_score'] = self.weights['occasion'] * (occasion_match * 30)
        
        if preferences.hair_type in (style.hair_types or []):
            breakdown['hair_type_score'] = self.weights['hair_type'] * 80
        
        if preferences.hair_length in (style.hair_lengths or []):
            breakdown['hair_length_score'] = self.weights['hair_length'] * 80
        
        if preferences.maintenance == style.maintenance:
            breakdown['maintenance_score'] = self.weights['maintenance'] * 100
        
        return breakdown
    
    def _apply_diversity_filter(self, scored_styles: List[Dict], limit: int):
        """Apply diversity filter to avoid too similar recommendations"""
        if len(scored_styles) <= limit:
            return scored_styles
        
        diverse_styles = []
        used_categories = set()
        used_lengths = set()
        used_maintenance = set()
        
        # First pass: select top styles ensuring diversity
        for item in scored_styles:
            style = item['hairstyle']
            
            # Check diversity criteria
            category_name = style.category.name if style.category else 'uncategorized'
            hair_length = style.hair_lengths[0] if style.hair_lengths else 'unknown'
            maintenance = style.maintenance or 'unknown'
            
            # Prioritize diversity for top picks
            if len(diverse_styles) < limit // 2:
                if (category_name not in used_categories or 
                    hair_length not in used_lengths or 
                    maintenance not in used_maintenance):
                    diverse_styles.append(item)
                    used_categories.add(category_name)
                    used_lengths.add(hair_length)
                    used_maintenance.add(maintenance)
                    continue
            
            # Fill remaining slots with highest scoring styles
            if len(diverse_styles) < limit:
                diverse_styles.append(item)
        
        return diverse_styles[:limit]