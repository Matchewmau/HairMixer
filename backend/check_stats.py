import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.models import Hairstyle

print("="*60)
print("DATABASE STATISTICS")
print("="*60)

total = Hairstyle.objects.filter(is_active=True).count()
male = Hairstyle.objects.filter(is_active=True, suitable_gender='male').count()
female = Hairstyle.objects.filter(is_active=True, suitable_gender='female').count()
unisex = Hairstyle.objects.filter(is_active=True, suitable_gender='unisex').count()

with_images = Hairstyle.objects.filter(is_active=True).exclude(image='').count()
without_images = Hairstyle.objects.filter(is_active=True, image='').count()

print(f"\n📊 HAIRSTYLE COUNT:")
print(f"   Total active hairstyles: {total}")
print(f"\n👥 GENDER DISTRIBUTION:")
print(f"   Male: {male} ({male/total*100:.1f}%)")
print(f"   Female: {female} ({female/total*100:.1f}%)")
print(f"   Unisex: {unisex} ({unisex/total*100:.1f}%)")
print(f"\n🖼️  IMAGE STATUS:")
print(f"   With images: {with_images}")
print(f"   Without images: {without_images} ⚠️")
print(f"\n{'='*60}\n")
