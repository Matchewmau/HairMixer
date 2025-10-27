from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
import uuid

class CustomUser(AbstractUser):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    email = models.EmailField(unique=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'first_name', 'last_name']
    
    def __str__(self):
        return self.email

class UserProfile(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='profile')
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    phone_number = models.CharField(max_length=15, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.email}'s Profile"

# Enhanced Hair Analysis Models
class UploadedImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to="uploads/%Y/%m/")
    original_filename = models.CharField(max_length=255, blank=True)
    file_size = models.PositiveIntegerField(null=True, blank=True)  # in bytes
    image_width = models.PositiveIntegerField(null=True, blank=True)
    image_height = models.PositiveIntegerField(null=True, blank=True)
    face_detected = models.BooleanField(default=False)
    face_count = models.PositiveIntegerField(default=0)
    processing_status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed')
        ],
        default='pending'
    )
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['processing_status']),
        ]
    
    def __str__(self):
        return f"Image {self.id} - {self.processing_status}"


class UserPreference(models.Model):
    """
    Enhanced user preferences model for hairstyle recommendations.
    
    This model stores comprehensive user preferences including:
    - Face shape (auto-detected via ResNet50)
    - Hair characteristics (type, length, volume, thickness, texture)
    - Lifestyle and maintenance preferences
    - Styling preferences and occasions
    - Hairstyle family preferences
    
    The collected preferences are used by the ML recommendation engine
    (hairstyle_family_model.pkl) to provide personalized top 10 hairstyle
    recommendations.
    
    Attributes:
        faceshape (str): Auto-detected face shape from ResNet50 model
        faceshape_confidence (float): Confidence score of face shape detection
        gender (str): User's gender identity
        hair_type (str): Hair type (straight, wavy, curly, coily)
        hair_length (str): Current or desired hair length
        volume (str): Preferred hair volume level
        hair_thickness (str): Hair thickness/density
        hair_texture_detail (str): Detailed hair texture characteristics
        styling_preference (str): Preferred styling approach
        hair_condition (str): Current hair health condition
        wants_bangs (bool): Whether user wants bangs/fringe
        lifestyle (str): User's lifestyle category
        maintenance (str): Acceptable maintenance level
        styling_maintenance (str): Daily styling maintenance preference
        occasions (list): List of occasions for styling
        hairstyle_family (str): Preferred hairstyle family/category
        hairstyle_name (str): Specific hairstyle name if known
    """
    GENDER_CHOICES = [
        ("male", "Male"),
        ("female", "Female")
    ]
    OCCASION_CHOICES = [
        ("work", "Work"),
        ("casual", "Casual"),
        ("formal", "Formal"),
        ("party", "Party"),
        ("wedding", "Wedding"),
        ("birthday", "Birthday")
    ]
    HAIR_TYPE_CHOICES = [
        ("straight", "Straight"),
        ("wavy", "Wavy"),
        ("curly", "Curly"),
        ("coily", "Coily")
    ]
    LENGTH_CHOICES = [
        ("short", "Short"),
        ("medium", "Medium"),
        ("long", "Long")
    ]
    MAINTENANCE_CHOICES = [
        ("low", "Low (Wash & Go)"),
        ("medium", "Medium (Some Styling)"),
        ("high", "High (Daily Styling)")
    ]
    LIFESTYLE_CHOICES = [
        ("active", "Active"),
        ("professional", "Professional"),
        ("creative", "Creative"),
        ("casual", "Casual"),
        ("moderate", "Moderate"),
        ("relaxed", "Relaxed")
    ]
    VOLUME_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High")
    ]
    STYLING_PREFERENCE_CHOICES = [
        ("natural", "Natural"),
        ("casual", "Casual"),
        ("classic", "Classic"),
        ("polished", "Polished"),
        ("elegant", "Elegant"),
        ("glamorous", "Glamorous"),
        ("trendy", "Trendy"),
        ("edgy", "Edgy")
    ]
    HAIR_CONDITION_CHOICES = [
        ("none", "None/Healthy"),
        ("excellent", "Excellent"),
        ("good", "Good"),
        ("fair", "Fair"),
        ("damaged", "Damaged"),
        ("dry_ends", "Dry Ends"),
        ("oily_scalp", "Oily Scalp"),
        ("dandruff", "Dandruff"),
        ("frizzy", "Frizzy"),
        ("split_ends", "Split Ends"),
        ("thinning", "Thinning"),
        ("sensitive_scalp", "Sensitive Scalp")
    ]
    HAIR_THICKNESS_CHOICES = [
        ("thin", "Thin"),
        ("medium", "Medium"),
        ("thick", "Thick"),
        ("very_thick", "Very Thick")
    ]
    HAIR_TEXTURE_DETAIL_CHOICES = [
        ("fine", "Fine"),
        ("normal", "Normal"),
        ("thick", "Thick"),
        ("smooth", "Smooth"),
        ("coarse", "Coarse"),
        ("silky", "Silky"),
        ("frizzy", "Frizzy")
    ]
    HAIR_COLOR_CHOICES = [
        ("black", "Black"),
        ("brown", "Brown"),
        ("blonde", "Blonde"),
        ("red", "Red"),
        ("auburn", "Auburn"),
        ("gray", "Gray"),
        ("white", "White"),
        ("other", "Other")
    ]
    # Face shape choices - Updated to match ResNet50 model (5 classes)
    FACE_SHAPE_CHOICES = [
        ("heart", "Heart"),
        ("oblong", "Oblong"),
        ("oval", "Oval"),
        ("round", "Round"),
        ("square", "Square")
    ]

    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, null=True, blank=True
    )
    
    # Face shape (auto-detected from ResNet50)
    faceshape = models.CharField(
        max_length=20, choices=FACE_SHAPE_CHOICES, blank=True
    )
    faceshape_confidence = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Basic preferences
    gender = models.CharField(
        max_length=16, choices=GENDER_CHOICES, blank=True
    )
    occasions = models.JSONField(default=list, blank=True)
    hair_type = models.CharField(max_length=16, choices=HAIR_TYPE_CHOICES)
    hair_length = models.CharField(max_length=16, choices=LENGTH_CHOICES)
    hair_color = models.CharField(max_length=50, choices=HAIR_COLOR_CHOICES, blank=True)
    lifestyle = models.CharField(
        max_length=20, choices=LIFESTYLE_CHOICES, blank=True
    )
    maintenance = models.CharField(
        max_length=16, choices=MAINTENANCE_CHOICES
    )
    
    # New detailed hair characteristics
    volume = models.CharField(
        max_length=16, choices=VOLUME_CHOICES, blank=True
    )
    styling_maintenance = models.CharField(
        max_length=16, choices=MAINTENANCE_CHOICES, blank=True
    )
    hair_texture_detail = models.CharField(
        max_length=20, choices=HAIR_TEXTURE_DETAIL_CHOICES, blank=True
    )
    styling_preference = models.CharField(
        max_length=20, choices=STYLING_PREFERENCE_CHOICES, blank=True
    )
    # Hair condition - Changed to JSONField to support multiple conditions
    hair_condition = models.JSONField(default=list, blank=True)
    hair_thickness = models.CharField(
        max_length=16, choices=HAIR_THICKNESS_CHOICES, blank=True
    )
    wants_bangs = models.BooleanField(default=False, null=True, blank=True)
    
    # Hairstyle preferences
    hairstyle_family = models.CharField(max_length=100, blank=True)
    hairstyle_name = models.CharField(max_length=200, blank=True)
    
    # Advanced preferences
    budget_range = models.CharField(max_length=20, blank=True)
    color_preference = models.CharField(max_length=50, blank=True)
    avoid_styles = models.JSONField(default=list, blank=True)
    preferred_stylists = models.JSONField(default=list, blank=True)
    
    # Metadata
    version = models.PositiveIntegerField(default=1)  # For preference versioning
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"Preferences {self.id} - {self.hair_type} {self.hair_length}"

class HairstyleCategory(models.Model):
    """Categories for better hairstyle organization"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    sort_order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Hairstyle Categories"
        ordering = ['sort_order', 'name']
    
    def __str__(self):
        return self.name

class Hairstyle(models.Model):
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
        ('professional', 'Professional Only')
    ]
    
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('unisex', 'Unisex')
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    category = models.ForeignKey(HairstyleCategory, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Style attributes
    suitable_gender = models.CharField(
        max_length=10,
        choices=GENDER_CHOICES,
        default='unisex',
        db_index=True,
        help_text="Target gender for this hairstyle"
    )
    tags = models.JSONField(default=list, blank=True)
    face_shapes = models.JSONField(default=list, blank=True)
    hair_types = models.JSONField(default=list, blank=True)  # Compatible hair types
    hair_lengths = models.JSONField(default=list, blank=True)  # Compatible lengths
    occasions = models.JSONField(default=list, blank=True)
    
    # Images and media
    image = models.ImageField(upload_to="hairstyles/%Y/%m/", blank=True, null=True)
    image_url = models.URLField(blank=True)  # External CDN
    thumbnail = models.ImageField(upload_to="hairstyles/thumbs/%Y/%m/", blank=True, null=True)
    tutorial_video_url = models.URLField(blank=True)
    before_after_images = models.JSONField(default=list, blank=True)
    
    # Style metadata
    maintenance = models.CharField(max_length=16, blank=True)
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES, default='medium')
    estimated_time = models.PositiveIntegerField(null=True, blank=True, help_text="Time in minutes")
    trend_score = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(10.0)]
    )
    popularity_score = models.FloatField(default=0.0)
    
    # SEO and content
    seo_keywords = models.JSONField(default=list, blank=True)
    styling_tips = models.TextField(blank=True)
    products_needed = models.JSONField(default=list, blank=True)
    
    # Status and metadata
    is_active = models.BooleanField(default=True)
    is_featured = models.BooleanField(default=False)
    created_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-trend_score', '-popularity_score', 'name']
        indexes = [
            models.Index(fields=['is_active', 'trend_score']),
            models.Index(fields=['category', 'is_active']),
            models.Index(fields=['is_featured']),
        ]
    
    def __str__(self):
        return self.name
    
    def update_popularity(self):
        """Update popularity based on recent recommendations and feedback"""
        from django.db.models import Avg
        recent_recommendations = self.recommendationlog_set.filter(
            created_at__gte=timezone.now() - timezone.timedelta(days=30)
        ).count()
        
        avg_rating = self.feedback_set.aggregate(
            avg_rating=Avg('rating')
        )['avg_rating'] or 0
        
        # Simple popularity calculation
        self.popularity_score = (recent_recommendations * 0.3) + (avg_rating * 0.7)
        self.save(update_fields=['popularity_score'])

class RecommendationLog(models.Model):
    RECOMMENDATION_STATUS = [
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null=True, blank=True)
    uploaded = models.ForeignKey(UploadedImage, on_delete=models.CASCADE)
    preference = models.ForeignKey(UserPreference, on_delete=models.CASCADE)
    
    # Analysis results
    face_shape = models.CharField(max_length=50)
    face_shape_confidence = models.FloatField(default=0.0)
    detected_features = models.JSONField(default=dict, blank=True)  # Facial features
    
    # Recommendations
    selected_hairstyle = models.ForeignKey(Hairstyle, null=True, blank=True, on_delete=models.SET_NULL)
    candidates = models.JSONField(default=list)  # Hairstyle IDs with scores
    recommendation_scores = models.JSONField(default=dict, blank=True)
    
    # Processing metadata
    status = models.CharField(max_length=20, choices=RECOMMENDATION_STATUS, default='pending')
    processing_time = models.FloatField(null=True, blank=True)  # seconds
    model_version = models.CharField(max_length=20, default='v1.0')
    error_message = models.TextField(blank=True)
    
    # Analytics
    view_count = models.PositiveIntegerField(default=0)
    shared_count = models.PositiveIntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['status']),
            models.Index(fields=['face_shape']),
        ]
    
    def __str__(self):
        return f"Recommendation {self.id} - {self.face_shape}"

class Feedback(models.Model):
    RATING_CHOICES = [
        (1, '1 Star - Poor'),
        (2, '2 Stars - Fair'),
        (3, '3 Stars - Good'),
        (4, '4 Stars - Very Good'),
        (5, '5 Stars - Excellent')
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null=True, blank=True)
    recommendation = models.ForeignKey(RecommendationLog, on_delete=models.CASCADE)
    hairstyle = models.ForeignKey(Hairstyle, on_delete=models.CASCADE, null=True, blank=True)
    
    # Feedback data
    liked = models.BooleanField()
    rating = models.PositiveIntegerField(choices=RATING_CHOICES, null=True, blank=True)
    note = models.TextField(blank=True)
    
    # Detailed feedback
    accuracy_rating = models.PositiveIntegerField(choices=RATING_CHOICES, null=True, blank=True)
    style_satisfaction = models.PositiveIntegerField(choices=RATING_CHOICES, null=True, blank=True)
    would_recommend = models.BooleanField(null=True, blank=True)
    
    # Metadata
    is_public = models.BooleanField(default=False)  # For testimonials
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['hairstyle', 'rating']),
            models.Index(fields=['is_public']),
        ]
    
    def __str__(self):
        return f"Feedback {self.id} - {'Liked' if self.liked else 'Disliked'}"

# Analytics and Caching Models
class AnalyticsEvent(models.Model):
    """Track user interactions for analytics"""
    EVENT_TYPES = [
        ('page_view', 'Page View'),
        ('image_upload', 'Image Upload'),
        ('recommendation_generated', 'Recommendation Generated'),
        ('style_clicked', 'Style Clicked'),
        ('overlay_generated', 'Overlay Generated'),
        ('feedback_submitted', 'Feedback Submitted'),
        ('share_clicked', 'Share Clicked'),
    ]
    
    user = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True)
    event_type = models.CharField(max_length=30, choices=EVENT_TYPES)
    event_data = models.JSONField(default=dict, blank=True)
    session_id = models.CharField(max_length=100, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['event_type', 'created_at']),
            models.Index(fields=['user', 'created_at']),
        ]

class CachedRecommendation(models.Model):
    """Cache recommendations for better performance"""
    cache_key = models.CharField(max_length=255, unique=True, db_index=True)
    face_shape = models.CharField(max_length=50)
    preference_hash = models.CharField(max_length=64)  # Hash of preferences
    recommendations = models.JSONField()
    hit_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['expires_at']),
            models.Index(fields=['face_shape', 'preference_hash']),
        ]
    
    def is_expired(self):
        return timezone.now() > self.expires_at
