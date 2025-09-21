from django.core.management.base import BaseCommand
from hairmixer_app.services.cache_manager import CacheManager

class Command(BaseCommand):
    help = 'Clean up expired cache entries'
    
    def handle(self, *_args, **options):
        cache_manager = CacheManager()
        cleaned_count = cache_manager.cleanup_expired_cache()
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully cleaned up {cleaned_count} expired cache entries')
        )