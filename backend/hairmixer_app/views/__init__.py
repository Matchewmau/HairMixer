from .auth import signup, login, logout, user_profile
from .analysis import (
    UploadImageView,
    SetPreferencesView,
    RecommendView,
    OverlayView,
    AutoOverlayView,
)
from .catalog import (
    FeaturedHairstylesView,
    TrendingHairstylesView,
    HairstyleDetailView,
    ListHairstylesView,
    HairstyleCategoriesView,
    SearchView,
    FaceShapesView,
    OccasionsView,
)
from .user import (
    UserRecommendationsView,
    UserFavoritesView,
    UserHistoryView,
)
from .misc import (
    FeedbackView,
    AnalyticsEventView,
    debug_face_detection,
    debug_resnet_features,
)
from .admin import (
    CacheStatsView,
    CacheCleanupView,
    SystemAnalyticsView,
    health_check,
    api_root,
)

__all__ = [
    # auth
    'signup',
    'login',
    'logout',
    'user_profile',
    # analysis
    'UploadImageView',
    'SetPreferencesView',
    'RecommendView',
    'OverlayView',
    'AutoOverlayView',
    # catalog
    'FeaturedHairstylesView',
    'TrendingHairstylesView',
    'HairstyleDetailView',
    'ListHairstylesView',
    'HairstyleCategoriesView',
    'SearchView',
    'FaceShapesView',
    'OccasionsView',
    # user
    'UserRecommendationsView',
    'UserFavoritesView',
    'UserHistoryView',
    # misc
    'FeedbackView',
    'AnalyticsEventView',
    'debug_face_detection',
    'debug_resnet_features',
    # admin/system
    'CacheStatsView',
    'CacheCleanupView',
    'SystemAnalyticsView',
    'health_check',
    'api_root',
]
