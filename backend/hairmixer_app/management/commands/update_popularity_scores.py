from django.core.management.base import BaseCommand
from hairmixer_app.models import Hairstyle

class Command(BaseCommand):
    help = 'Update popularity scores for all hairstyles'
    
    def handle(self, *args, **options):
        hairstyles = Hairstyle.objects.filter(is_active=True)
        updated_count = 0
        
        for hairstyle in hairstyles:
            hairstyle.update_popularity()
            updated_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully updated popularity scores for {updated_count} hairstyles')
        )