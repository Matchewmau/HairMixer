from rest_framework import serializers
from django.core.files.uploadedfile import InMemoryUploadedFile
from PIL import Image as PILImage
import hashlib
import json
from .models import (
    CustomUser, UserProfile, UploadedImage, UserPreference, 
    Hairstyle, HairstyleCategory, RecommendationLog, Feedback,
    AnalyticsEvent
)

class RecommendRequestSerializer(serializers.Serializer):
    image_id = serializers.UUIDField()
    preference_id = serializers.UUIDField()

class OverlayRequestSerializer(serializers.Serializer):
    image_id = serializers.UUIDField()
    hairstyle_id = serializers.UUIDField()
    overlay_type = serializers.ChoiceField(choices=[('basic', 'basic'), ('advanced', 'advanced')], default='basic')

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'email', 'first_name', 'last_name', 'date_joined']
        extra_kwargs = {
            'password': {'write_only': True}
        }

class UserRegistrationSerializer(serializers.Serializer):
    firstName = serializers.CharField(max_length=30)
    lastName = serializers.CharField(max_length=30)
    email = serializers.EmailField()
    password = serializers.CharField(min_length=8)
    
    def validate_email(self, value):
        if CustomUser.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value
    
    def validate_password(self, value):
        """Enhanced password validation"""
        if len(value) < 8:
            raise serializers.ValidationError("Password must be at least 8 characters long.")
        if not any(char.isdigit() for char in value):
            raise serializers.ValidationError("Password must contain at least one digit.")
        if not any(char.isupper() for char in value):
            raise serializers.ValidationError("Password must contain at least one uppercase letter.")
        return value

class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = UserProfile
        fields = ['user', 'avatar', 'phone_number', 'date_of_birth', 'created_at', 'updated_at']

class UploadedImageSerializer(serializers.ModelSerializer):
    file_size_mb = serializers.SerializerMethodField()
    
    class Meta:
        model = UploadedImage
        fields = [
            "id", "image", "original_filename", "file_size", "file_size_mb",
            "image_width", "image_height", "face_detected", "face_count",
            "processing_status", "created_at"
        ]
        read_only_fields = ["user", "file_size", "image_width", "image_height", "face_detected", "face_count"]
    
    def get_file_size_mb(self, obj):
        if obj.file_size:
            return round(obj.file_size / (1024 * 1024), 2)
        return None
    
    def validate_image(self, value):
        """Enhanced image validation"""
        if not isinstance(value, InMemoryUploadedFile):
            raise serializers.ValidationError("Invalid file format")
        
        # Check file size (10MB limit)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 10MB")
        
        # Check image format
        try:
            img = PILImage.open(value)
            img.verify()
            
            # Check minimum dimensions
            if img.width < 200 or img.height < 200:
                raise serializers.ValidationError("Image must be at least 200x200 pixels")
            
            # Check aspect ratio (not too wide or tall)
            aspect_ratio = img.width / img.height
            if aspect_ratio > 3 or aspect_ratio < 0.33:
                raise serializers.ValidationError("Image aspect ratio is too extreme")
                
        except Exception:
            raise serializers.ValidationError("Invalid image file")
        
        return value
    
    def create(self, validated_data):
        # Extract metadata from image
        if 'image' in validated_data:
            image_file = validated_data['image']
            validated_data['original_filename'] = image_file.name
            validated_data['file_size'] = image_file.size
            
            # Get image dimensions
            try:
                img = PILImage.open(image_file)
                validated_data['image_width'] = img.width
                validated_data['image_height'] = img.height
            except Exception:
                pass
        
        return super().create(validated_data)

class UserPreferenceSerializer(serializers.ModelSerializer):
    preference_hash = serializers.SerializerMethodField()
    
    class Meta:
        model = UserPreference
        fields = [
            "id", "gender", "occasions", "hair_type", "hair_length", 
            "hair_color", "lifestyle", "maintenance", "budget_range",
            "color_preference", "avoid_styles", "version", "preference_hash",
            "created_at", "updated_at"
        ]
        read_only_fields = ["user", "version", "preference_hash"]
    
    def get_preference_hash(self, obj):
        """Generate a hash of preferences for caching"""
        pref_data = {
            'occasions': sorted(obj.occasions or []),
            'hair_type': obj.hair_type,
            'hair_length': obj.hair_length,
            'lifestyle': obj.lifestyle,
            'maintenance': obj.maintenance,
        }
        return hashlib.md5(json.dumps(pref_data, sort_keys=True).encode()).hexdigest()
    
    def validate_occasions(self, value):
        if not value or len(value) == 0:
            raise serializers.ValidationError("At least one occasion must be selected")
        
        valid_occasions = [choice[0] for choice in UserPreference.OCCASION_CHOICES]
        for occasion in value:
            if occasion not in valid_occasions:
                raise serializers.ValidationError(f"'{occasion}' is not a valid occasion")
        return value

class HairstyleCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = HairstyleCategory
        fields = ['id', 'name', 'description', 'parent', 'sort_order']

class HairstyleSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    image_url_full = serializers.SerializerMethodField()
    estimated_time_formatted = serializers.SerializerMethodField()
    
    class Meta:
        model = Hairstyle
        fields = [
            "id", "name", "description", "category", "category_name",
            "tags", "face_shapes", "hair_types", "hair_lengths", "occasions",
            "image", "image_url", "image_url_full", "thumbnail",
            "maintenance", "difficulty", "estimated_time", "estimated_time_formatted",
            "trend_score", "popularity_score", "styling_tips", "products_needed",
            "is_featured", "created_at"
        ]
    
    def get_image_url_full(self, obj):
        if obj.image:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.image.url)
        elif obj.image_url:
            return obj.image_url
        return None
    
    def get_estimated_time_formatted(self, obj):
        if obj.estimated_time:
            hours = obj.estimated_time // 60
            minutes = obj.estimated_time % 60
            if hours > 0:
                return f"{hours}h {minutes}m"
            return f"{minutes}m"
        return None

class RecommendationLogSerializer(serializers.ModelSerializer):
    candidates_details = serializers.SerializerMethodField()
    face_shape_display = serializers.SerializerMethodField()
    processing_time_formatted = serializers.SerializerMethodField()
    
    class Meta:
        model = RecommendationLog
        fields = [
            "id", "face_shape", "face_shape_display", "face_shape_confidence",
            "detected_features", "candidates", "candidates_details", 
            "recommendation_scores", "status", "processing_time", 
            "processing_time_formatted", "model_version", "view_count",
            "created_at"
        ]
    
    def get_candidates_details(self, obj):
        if obj.candidates:
            hairstyles = Hairstyle.objects.filter(id__in=obj.candidates, is_active=True)
            return HairstyleSerializer(hairstyles, many=True, context=self.context).data
        return []
    
    def get_face_shape_display(self, obj):
        face_shape_map = {
            'oval': 'Oval',
            'round': 'Round', 
            'square': 'Square',
            'heart': 'Heart',
            'diamond': 'Diamond',
            'oblong': 'Oblong'
        }
        return face_shape_map.get(obj.face_shape, obj.face_shape.title())
    
    def get_processing_time_formatted(self, obj):
        if obj.processing_time:
            return f"{obj.processing_time:.2f}s"
        return None

class FeedbackSerializer(serializers.ModelSerializer):
    hairstyle_name = serializers.CharField(source='hairstyle.name', read_only=True)
    
    class Meta:
        model = Feedback
        fields = [
            "id", "liked", "rating", "note", "accuracy_rating", 
            "style_satisfaction", "would_recommend", "hairstyle",
            "hairstyle_name", "recommendation", "created_at"
        ]
        read_only_fields = ["user"]
    
    def validate_rating(self, value):
        if value and (value < 1 or value > 5):
            raise serializers.ValidationError("Rating must be between 1 and 5")
        return value

class AnalyticsEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalyticsEvent
        fields = ["event_type", "event_data", "session_id", "created_at"]
        read_only_fields = ["user", "ip_address", "user_agent"]