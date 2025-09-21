from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

# Create router for viewsets if needed
router = DefaultRouter()

urlpatterns = [
    # API Root
    path('', views.api_root, name='api_root'),  # Add this line
    
    # Include router URLs
    path('', include(router.urls)),
    
    # Authentication endpoints
    path('auth/signup/', views.signup, name='signup'),
    path('auth/login/', views.login, name='login'),
    path('auth/logout/', views.logout, name='logout'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/profile/', views.user_profile, name='user_profile'),
    
    # Core functionality endpoints
    path('upload/', views.UploadImageView.as_view(), name='upload_image'),
    path(
        'preferences/',
        views.SetPreferencesView.as_view(),
        name='set_preferences',
    ),
    path('recommend/', views.RecommendView.as_view(), name='recommend'),
    path('overlay/', views.OverlayView.as_view(), name='overlay'),
    path('feedback/', views.FeedbackView.as_view(), name='feedback'),
    
    # Hairstyle endpoints
    path(
        'hairstyles/',
        views.ListHairstylesView.as_view(),
        name='list_hairstyles',
    ),
    path(
        'hairstyles/featured/',
        views.FeaturedHairstylesView.as_view(),
        name='featured_hairstyles',
    ),
    path(
        'hairstyles/trending/',
        views.TrendingHairstylesView.as_view(),
        name='trending_hairstyles',
    ),
    path(
        'hairstyles/<uuid:style_id>/',
        views.HairstyleDetailView.as_view(),
        name='hairstyle_detail',
    ),
    path(
        'hairstyles/categories/',
        views.HairstyleCategoriesView.as_view(),
        name='hairstyle_categories',
    ),
    
    # User-specific endpoints (require authentication)
    path(
        'user/recommendations/',
        views.UserRecommendationsView.as_view(),
        name='user_recommendations',
    ),
    path(
        'user/favorites/',
        views.UserFavoritesView.as_view(),
        name='user_favorites',
    ),
    path(
        'user/history/',
        views.UserHistoryView.as_view(),
        name='user_history',
    ),
    
    # Search and filter endpoints
    path('search/', views.SearchView.as_view(), name='search'),
    path(
        'filter/face-shapes/',
        views.FaceShapesView.as_view(),
        name='face_shapes',
    ),
    path('filter/occasions/', views.OccasionsView.as_view(), name='occasions'),
    
    # Analytics and admin endpoints
    path(
        'analytics/event/',
        views.AnalyticsEventView.as_view(),
        name='analytics_event',
    ),
    path(
        'admin/cache/stats/',
        views.CacheStatsView.as_view(),
        name='cache_stats',
    ),
    path(
        'admin/cache/cleanup/',
        views.CacheCleanupView.as_view(),
        name='cache_cleanup',
    ),
    path(
        'admin/analytics/',
        views.SystemAnalyticsView.as_view(),
        name='system_analytics',
    ),
    
    # Health check
    path('health/', views.health_check, name='health_check'),
    
    # debug face detection
    path(
        'debug/face-detection/',
        views.debug_face_detection,
        name='debug_face_detection',
    ),
    # path(
    #     'debug/resnet-features/',
    #     views.debug_resnet_features,
    #     name='debug_resnet_features',
    # ),
]
