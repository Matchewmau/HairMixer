from django.core.management.base import BaseCommand
from hairmixer_app.models import Hairstyle, HairstyleCategory

class Command(BaseCommand):
    help = 'Create sample hairstyles and categories with themes for testing'
    
    def handle(self, *args, **options):
        # Create categories with themes
        categories_data = [
            {'name': 'Classic Styles', 'description': 'Timeless and traditional hairstyles'},
            {'name': 'Trendy Styles', 'description': 'Modern and fashionable cuts'}, 
            {'name': 'Retro Styles', 'description': 'Vintage-inspired hairstyles'},
            {'name': 'Formal Styles', 'description': 'Elegant styles for special occasions'},
            {'name': 'Casual Styles', 'description': 'Relaxed, everyday hairstyles'},
        ]
        
        created_categories = []
        for cat_data in categories_data:
            cat, created = HairstyleCategory.objects.get_or_create(
                name=cat_data['name'],
                defaults={
                    'description': cat_data['description'],
                    'is_active': True,
                    'sort_order': 0
                }
            )
            created_categories.append(cat)
            if created:
                self.stdout.write(f"Created category: {cat.name}")
        
        # Sample hairstyle data with themes
        sample_styles = [
            {
                'name': 'Classic Bob',
                'description': 'A timeless bob cut that suits most face shapes',
                'category': created_categories[0],  # Classic
                'face_shapes': ['oval', 'square', 'heart'],
                'hair_types': ['straight', 'wavy'],
                'hair_lengths': ['short', 'medium'],
                'occasions': ['casual', 'formal', 'work'],
                'maintenance': 'medium',
                'difficulty': 'easy',
                'estimated_time': 30,
                'trend_score': 8.5,
                'popularity_score': 8.0,
                'tags': ['classic', 'versatile', 'professional'],
                'is_featured': True,
                'is_active': True
            },
            {
                'name': 'Trendy Pixie Cut',
                'description': 'Modern short pixie with edgy styling',
                'category': created_categories[1],  # Trendy
                'face_shapes': ['oval', 'heart'],
                'hair_types': ['straight', 'wavy'],
                'hair_lengths': ['pixie'],
                'occasions': ['casual', 'party', 'work'],
                'maintenance': 'low',
                'difficulty': 'professional',
                'estimated_time': 45,
                'trend_score': 9.2,
                'popularity_score': 8.5,
                'tags': ['trendy', 'modern', 'edgy', 'low-maintenance'],
                'is_featured': True,
                'is_active': True
            },
            {
                'name': '1970s Shag',
                'description': 'Retro-inspired layered shag with volume',
                'category': created_categories[2],  # Retro
                'face_shapes': ['oval', 'round', 'diamond'],
                'hair_types': ['wavy', 'curly'],
                'hair_lengths': ['medium', 'long'],
                'occasions': ['casual', 'party', 'date'],
                'maintenance': 'medium',
                'difficulty': 'medium',
                'estimated_time': 40,
                'trend_score': 7.8,
                'popularity_score': 7.5,
                'tags': ['retro', 'vintage', 'layered', 'volume'],
                'is_featured': True,
                'is_active': True
            },
            {
                'name': 'Elegant Updo',
                'description': 'Sophisticated updo perfect for formal events',
                'category': created_categories[3],  # Formal
                'face_shapes': ['oval', 'heart', 'square'],
                'hair_types': ['straight', 'wavy'],
                'hair_lengths': ['medium', 'long'],
                'occasions': ['formal', 'wedding', 'work'],
                'maintenance': 'high',
                'difficulty': 'professional',
                'estimated_time': 60,
                'trend_score': 8.0,
                'popularity_score': 7.8,
                'tags': ['formal', 'elegant', 'sophisticated', 'updo'],
                'is_featured': True,
                'is_active': True
            },
            {
                'name': 'Beach Waves',
                'description': 'Relaxed, casual waves with natural texture',
                'category': created_categories[4],  # Casual
                'face_shapes': ['oval', 'round', 'diamond'],
                'hair_types': ['wavy', 'straight'],
                'hair_lengths': ['medium', 'long'],
                'occasions': ['casual', 'date', 'travel'],
                'maintenance': 'low',
                'difficulty': 'easy',
                'estimated_time': 15,
                'trend_score': 8.8,
                'popularity_score': 9.0,
                'tags': ['casual', 'natural', 'beachy', 'effortless'],
                'is_featured': False,
                'is_active': True
            },
            {
                'name': 'Layered Medium Cut',
                'description': 'Versatile layered cut for medium-length hair',
                'category': created_categories[0],  # Classic
                'face_shapes': ['round', 'square', 'diamond'],
                'hair_types': ['straight', 'wavy'],
                'hair_lengths': ['medium'],
                'occasions': ['casual', 'work', 'date'],
                'maintenance': 'medium',
                'difficulty': 'medium',
                'estimated_time': 35,
                'trend_score': 8.1,
                'popularity_score': 8.3,
                'tags': ['classic', 'layered', 'versatile', 'flattering'],
                'is_featured': False,
                'is_active': True
            }
        ]
        
        # Create hairstyles
        for style_data in sample_styles:
            style, created = Hairstyle.objects.get_or_create(
                name=style_data['name'],
                defaults=style_data
            )
            if created:
                self.stdout.write(f"Created hairstyle: {style.name}")
        
        total_styles = Hairstyle.objects.filter(is_active=True).count()
        total_categories = HairstyleCategory.objects.filter(is_active=True).count()
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Sample data created successfully!\n'
                f'Categories: {total_categories}\n'
                f'Hairstyles: {total_styles}'
            )
        )