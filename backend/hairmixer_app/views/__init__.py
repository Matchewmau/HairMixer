from .auth import signup, login, logout, user_profile
from .analysis import (
    UploadImageView,
    SetPreferencesView,
    RecommendView,
    MLRecommendView,
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
from .hairstyle_detail_view import HairstyleDetailWithAIView
from .user import (
    UserRecommendationsView,
    UserFavoritesView,
    UserHistoryView,
)
from .preference_profiles import (
    PreferenceProfileListCreateView,
    PreferenceProfileDetailView,
    PreferenceProfileSetDefaultView,
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
    'MLRecommendView',
    'OverlayView',
    'AutoOverlayView',
    # catalog
    'FeaturedHairstylesView',
    'TrendingHairstylesView',
    'HairstyleDetailView',
    'HairstyleDetailWithAIView',
    'ListHairstylesView',
    'HairstyleCategoriesView',
    'SearchView',
    'FaceShapesView',
    'OccasionsView',
    # user
    'UserRecommendationsView',
    'UserFavoritesView',
    'UserHistoryView',
    # preference profiles
    'PreferenceProfileListCreateView',
    'PreferenceProfileDetailView',
    'PreferenceProfileSetDefaultView',
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
