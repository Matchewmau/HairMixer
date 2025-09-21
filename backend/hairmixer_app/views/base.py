import logging

from django.conf import settings
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.throttling import UserRateThrottle
from ..services.image_service import ImageService
from ..services.recommendation_service import RecommendationService
from ..services.overlay_service import OverlayService

# Initialize logger
logger = logging.getLogger(__name__)

# Feature flags and optional imports
ML_PREPROCESS_AVAILABLE = False
ML_MODEL_AVAILABLE = False
RECOMMENDATION_ENGINE_AVAILABLE = False
OVERLAY_PROCESSOR_AVAILABLE = False
ANALYTICS_AVAILABLE = False
CACHE_MANAGER_AVAILABLE = False

try:
    from ..ml import preprocess as _pre  # noqa: F401
    ML_PREPROCESS_AVAILABLE = True
except Exception as e:
    logger.warning(f"ML preprocessing not available: {e}")

try:
    from ..ml.model import load_model  # noqa: F401
    ML_MODEL_AVAILABLE = True
except Exception as e:
    logger.warning(f"ML model not available: {e}")

try:
    from ..logic.recommendation_engine import (
        EnhancedRecommendationEngine,
    )  # noqa: F401
    RECOMMENDATION_ENGINE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Recommendation engine not available: {e}")

try:
    from ..overlay import AdvancedOverlayProcessor  # noqa: F401
    OVERLAY_PROCESSOR_AVAILABLE = True
except Exception as e:
    logger.warning(f"Overlay processor not available: {e}")

try:
    from ..services.analytics import AnalyticsService  # noqa: F401
    ANALYTICS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Analytics service not available: {e}")

try:
    from ..services.cache_manager import CacheManager  # noqa: F401
    CACHE_MANAGER_AVAILABLE = True
except Exception as e:
    logger.warning(f"Cache manager not available: {e}")

# Global instances - Initialize conditionally
MODEL = None
recommendation_engine = (
    EnhancedRecommendationEngine() if RECOMMENDATION_ENGINE_AVAILABLE else None
)
overlay_processor = (
    AdvancedOverlayProcessor() if OVERLAY_PROCESSOR_AVAILABLE else None
)
analytics_service = AnalyticsService() if ANALYTICS_AVAILABLE else None
cache_manager = CacheManager() if CACHE_MANAGER_AVAILABLE else None
image_service = ImageService()
recommendation_service = RecommendationService()
overlay_service = OverlayService()

# Helpers


def get_model():
    global MODEL
    if MODEL is None and ML_MODEL_AVAILABLE:
        try:
            from ..ml.model import load_model as _load_model

            MODEL = _load_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    return MODEL


def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


# Throttles
class ImageUploadThrottle(UserRateThrottle):
    rate = '10/hour'


class RecommendationThrottle(UserRateThrottle):
    rate = '20/hour'


# Admin permission helper (env-gated)
try:
    ADMIN_PERMISSION_CLASS = (
        IsAuthenticated
        if getattr(settings, 'RELAX_ADMIN_PERMS', True)
        else IsAdminUser
    )
except Exception:
    ADMIN_PERMISSION_CLASS = IsAuthenticated
