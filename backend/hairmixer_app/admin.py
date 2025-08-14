from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, UserProfile

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

# Register your models here
admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(UserProfile, UserProfileAdmin)

# Customize admin site header and title
admin.site.site_header = "HairMixer Admin"
admin.site.site_title = "HairMixer Admin Portal"
admin.site.index_title = "Welcome to HairMixer Administration"
