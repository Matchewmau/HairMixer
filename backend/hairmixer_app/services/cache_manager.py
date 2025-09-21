import logging
import hashlib
import json
from django.core.cache import cache
from django.utils import timezone
from django.conf import settings
from datetime import timedelta
from typing import Dict, Any, Optional, List
from ..models import CachedRecommendation, UploadedImage, UserPreference
from django.db.models import F

logger = logging.getLogger(__name__)


class CacheManager:
    """Advanced caching system for recommendations and other data"""
    
    def __init__(self):
        self.default_timeout = getattr(
            settings, 'RECOMMENDATION_CACHE_TIMEOUT', 3600
        )  # 1 hour
        self.cache_prefix = 'hairmixer'
    
    def get_recommendation_cache_key(
        self, uploaded_image: UploadedImage, preferences: UserPreference
    ) -> str:
        """Generate cache key for recommendations"""
        try:
            # Create hash of image and preferences
            image_hash = hashlib.md5(
                f"{uploaded_image.id}_{uploaded_image.updated_at}".encode()
            ).hexdigest()
            pref_hash = self._get_preference_hash(preferences)
            
            cache_key = f"{self.cache_prefix}_rec_{image_hash}_{pref_hash}"
            return cache_key
        except Exception as e:
            logger.error(f"Error generating cache key: {str(e)}")
            return f"{self.cache_prefix}_rec_fallback_{uploaded_image.id}"
    
    def _get_preference_hash(self, preferences: UserPreference) -> str:
        """Generate hash of user preferences"""
        pref_data = {
            'occasions': sorted(preferences.occasions or []),
            'hair_type': preferences.hair_type,
            'hair_length': preferences.hair_length,
            'lifestyle': preferences.lifestyle,
            'maintenance': preferences.maintenance,
            'version': preferences.version
        }
        pref_json = json.dumps(pref_data, sort_keys=True)
        return hashlib.md5(pref_json.encode()).hexdigest()
    
    def cache_recommendation(
        self,
        cache_key: str,
        recommendation_data: Dict,
        preferences: UserPreference,
        timeout: Optional[int] = None,
    ):
        """Cache recommendation results"""
        try:
            timeout = timeout or self.default_timeout
            expires_at = timezone.now() + timedelta(seconds=timeout)
            
            # Store in Django cache (fast access)
            cache.set(cache_key, recommendation_data, timeout)
            
            # Store in database (persistent)
            face_shape = recommendation_data.get('face_shape', 'unknown')
            pref_hash = self._get_preference_hash(preferences)
            
            CachedRecommendation.objects.update_or_create(
                cache_key=cache_key,
                defaults={
                    'face_shape': face_shape,
                    'preference_hash': pref_hash,
                    'recommendations': recommendation_data,
                    'expires_at': expires_at,
                    'hit_count': 0
                }
            )
            
            logger.info(f"Cached recommendation: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error caching recommendation: {str(e)}")
    
    def get_cached_recommendation(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached recommendation"""
        try:
            # Try Django cache first (fastest)
            cached_data = cache.get(cache_key)
            if cached_data:
                # Update hit count in database
                self._update_cache_hit_count(cache_key)
                logger.info(f"Cache hit (memory): {cache_key}")
                return cached_data
            
            # Try database cache
            try:
                cached_rec = CachedRecommendation.objects.get(
                    cache_key=cache_key,
                    expires_at__gt=timezone.now()
                )
                
                # Restore to memory cache
                cache.set(
                    cache_key, cached_rec.recommendations, self.default_timeout
                )
                
                # Update hit count
                cached_rec.hit_count += 1
                cached_rec.save(update_fields=['hit_count'])
                
                logger.info(f"Cache hit (database): {cache_key}")
                return cached_rec.recommendations
                
            except CachedRecommendation.DoesNotExist:
                pass
            
            logger.debug(f"Cache miss: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached recommendation: {str(e)}")
            return None
    
    def _update_cache_hit_count(self, cache_key: str):
        """Update hit count for cached recommendation"""
        try:
            CachedRecommendation.objects.filter(cache_key=cache_key).update(
                hit_count=F('hit_count') + 1
            )
        except Exception as e:
            logger.error(f"Error updating cache hit count: {str(e)}")
    
    def invalidate_recommendation_cache(
        self,
        user_id: Optional[int] = None,
        face_shape: Optional[str] = None,
    ):
        """Invalidate recommendation caches"""
        try:
            queryset = CachedRecommendation.objects.all()
            
            if face_shape:
                queryset = queryset.filter(face_shape=face_shape)
            
            cache_keys = list(queryset.values_list('cache_key', flat=True))
            
            # Remove from memory cache
            for key in cache_keys:
                cache.delete(key)
            
            # Remove from database
            deleted_count = queryset.delete()[0]
            
            logger.info(f"Invalidated {deleted_count} cached recommendations")
            
        except Exception as e:
            logger.error(f"Error invalidating recommendation cache: {str(e)}")
    
    def cache_hairstyles_by_category(
        self,
        category_id: str,
        hairstyles_data: List[Dict],
        timeout: Optional[int] = None,
    ):
        """Cache hairstyles by category"""
        try:
            cache_key = f"{self.cache_prefix}_styles_cat_{category_id}"
            timeout = timeout or (
                self.default_timeout * 4
            )  # Longer timeout for static data
            
            cache.set(cache_key, hairstyles_data, timeout)
            logger.info(f"Cached hairstyles for category: {category_id}")
            
        except Exception as e:
            logger.error(f"Error caching hairstyles by category: {str(e)}")
    
    def get_cached_hairstyles_by_category(
        self, category_id: str
    ) -> Optional[List[Dict]]:
        """Get cached hairstyles by category"""
        try:
            cache_key = f"{self.cache_prefix}_styles_cat_{category_id}"
            return cache.get(cache_key)
        except Exception as e:
            logger.error(f"Error getting cached hairstyles: {str(e)}")
            return None
    
    def cache_user_preferences(
        self,
        user_id: int,
        preferences_data: Dict,
        timeout: Optional[int] = None,
    ):
        """Cache user preferences"""
        try:
            cache_key = f"{self.cache_prefix}_prefs_{user_id}"
            timeout = timeout or (self.default_timeout * 2)
            
            cache.set(cache_key, preferences_data, timeout)
            
        except Exception as e:
            logger.error(f"Error caching user preferences: {str(e)}")
    
    def get_cached_user_preferences(self, user_id: int) -> Optional[Dict]:
        """Get cached user preferences"""
        try:
            cache_key = f"{self.cache_prefix}_prefs_{user_id}"
            return cache.get(cache_key)
        except Exception as e:
            logger.error(f"Error getting cached user preferences: {str(e)}")
            return None
    
    def cleanup_expired_cache(self):
        """Clean up expired cache entries from database"""
        try:
            expired_count = CachedRecommendation.objects.filter(
                expires_at__lt=timezone.now()
            ).delete()[0]
            
            logger.info(f"Cleaned up {expired_count} expired cache entries")
            return expired_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired cache: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            from django.db.models import Count, Avg, Sum
            
            stats = CachedRecommendation.objects.aggregate(
                total_entries=Count('id'),
                total_hits=Sum('hit_count'),
                avg_hits=Avg('hit_count')
            )
            
            # Active vs expired
            now = timezone.now()
            active_count = CachedRecommendation.objects.filter(
                expires_at__gt=now
            ).count()
            expired_count = CachedRecommendation.objects.filter(
                expires_at__lte=now
            ).count()
            
            stats.update({
                'active_entries': active_count,
                'expired_entries': expired_count,
                'hit_ratio': (
                    (stats['total_hits'] or 0)
                    / (stats['total_entries'] or 1)
                )
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}
