from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, UserProfile, SavedHairstyle, HairstyleLike, Hairstyle, HairstyleCategory

class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'
    fk_name = 'user'

class CustomUserAdmin(UserAdmin):
    # The fields to be used in displaying the User model.
    list_display = ('email', 'first_name', 'last_name', 'is_staff', 'is_active', 'date_joined')
    list_filter = ('is_staff', 'is_active', 'date_joined')
    
    # The fields to be used in the edit form
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    
    # The fields to be used in the add form
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'first_name', 'last_name', 'password1', 'password2'),
        }),
    )
    
    search_fields = ('email', 'first_name', 'last_name')
    ordering = ('email',)
    filter_horizontal = ('groups', 'user_permissions',)
    
    # Add the profile inline
    inlines = (UserProfileInline,)

class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone_number', 'date_of_birth', 'created_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('user__email', 'user__first_name', 'user__last_name', 'phone_number')
    
    fieldsets = (
        ('User Information', {'fields': ('user',)}),
        ('Profile Details', {'fields': ('avatar', 'phone_number', 'date_of_birth')}),
        ('Timestamps', {'fields': ('created_at', 'updated_at'), 'classes': ('collapse',)}),
    )
    
    readonly_fields = ('created_at', 'updated_at')


# Hairstyle Category Admin
class HairstyleCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'parent', 'sort_order', 'is_active', 'created_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('name', 'description')
    ordering = ('sort_order', 'name')


# Hairstyle Admin
class HairstyleAdmin(admin.ModelAdmin):
    list_display = (
        'name', 
        'category', 
        'gender_display',
        'difficulty', 
        'trend_score', 
        'popularity_score', 
        'is_active', 
        'is_featured'
    )
    list_filter = (
        'is_active', 
        'is_featured', 
        'category', 
        'difficulty', 
        'suitable_gender'
    )
    search_fields = ('name', 'description', 'seo_keywords')
    readonly_fields = ('created_at', 'updated_at', 'popularity_score')
    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'description', 'category', 'suitable_gender')
        }),
        ('Media', {
            'fields': ('image', 'thumbnail', 'image_url', 'tutorial_video_url', 'before_after_images')
        }),
        ('Matching Attributes', {
            'fields': ('face_shapes', 'hair_types', 'hair_lengths', 'occasions', 'tags'),
            'classes': ('collapse',)
        }),
        ('Details', {
            'fields': ('maintenance', 'difficulty', 'estimated_time', 'styling_tips', 'products_needed')
        }),
        ('Metrics & SEO', {
            'fields': ('trend_score', 'popularity_score', 'seo_keywords', 'is_active', 'is_featured')
        }),
        ('Meta', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def gender_display(self, obj):
        return obj.get_suitable_gender_display()
    gender_display.short_description = 'Gender'


# Saved Hairstyle Admin
class SavedHairstyleAdmin(admin.ModelAdmin):
    list_display = (
        'user',
        'hairstyle_name',
        'face_shape',
        'saved_at'
    )
    list_filter = ('saved_at', 'face_shape')
    search_fields = (
        'user__email',
        'hairstyle_name',
        'hairstyle__name'
    )
    readonly_fields = ('saved_at',)
    date_hierarchy = 'saved_at'
    ordering = ('-saved_at',)


# Hairstyle Like Admin
class HairstyleLikeAdmin(admin.ModelAdmin):
    list_display = (
        'user',
        'hairstyle',
        'reaction',
        'created_at',
        'updated_at'
    )
    list_filter = ('reaction', 'created_at')
    search_fields = (
        'user__email',
        'hairstyle__name'
    )
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)


# Register your models here
admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(UserProfile, UserProfileAdmin)
admin.site.register(HairstyleCategory, HairstyleCategoryAdmin)
admin.site.register(Hairstyle, HairstyleAdmin)
admin.site.register(SavedHairstyle, SavedHairstyleAdmin)
admin.site.register(HairstyleLike, HairstyleLikeAdmin)

# Customize admin site header and title
admin.site.site_header = "HairMixer Admin"
admin.site.site_title = "HairMixer Admin Portal"
admin.site.index_title = "Welcome to HairMixer Administration"
