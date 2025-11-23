"""
Extract all unique hairstyles from training data and populate database.

This script:
1. Reads merged_preferences_max_voting.csv (44,999 records)
2. Extracts ALL unique hairstyle names (~203 styles)
3. Groups them by hairstyle_family (26 families)
4. Populates the Hairstyle database table
5. Sets appropriate gender, category, and metadata

This implements the hierarchical recommendation system:
- Model predicts family (26 classes, high accuracy)
- System recommends specific styles from that family (203 total styles)
"""

import os
import sys
import django
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.models import Hairstyle, HairstyleCategory
from django.db import transaction


# Family to Category Mapping (same as in hairstyle_recommender.py)
FAMILY_TO_CATEGORY = {
    # Short styles
    'buzz_crew_family': 1,
    'short_crop_family': 1,
    'pixie_family': 1,
    'short_shag_family': 1,
    
    # Long styles
    'long_waves_family': 2,
    'long_curls_family': 2,
    'long_straight_family': 2,
    'long_shag_family': 2,
    'long_cuts_family': 2,
    'long_layered_family': 2,
    'long_bun_ponytail_family': 2,
    'long_slicked_family': 2,
    'braids_family': 2,
    
    # Medium styles
    'lob_family': 3,
    'bob_family': 3,
    'medium_shag_family': 3,
    'medium_waves_family': 3,
    'medium_mens_family': 3,
    'curtain_bangs_family': 3,
    
    # Curly/Textured styles
    'pompadour_quiff_family': 4,
    'faux_hawk_family': 4,
    'fade_undercut_family': 4,
}


def infer_gender_from_name(hairstyle_name, family_name):
    """Infer gender from hairstyle name and family."""
    name_lower = hairstyle_name.lower()
    family_lower = family_name.lower()
    
    # Male indicators
    male_keywords = ['man_', 'mens', 'male', 'beard', 'masculine', 'guy', 'boy']
    # Female indicators
    female_keywords = ['woman', 'womens', 'female', 'feminine', 'girl', 'ladies']
    
    # Check name first
    has_male = any(k in name_lower for k in male_keywords)
    has_female = any(k in name_lower for k in female_keywords)
    
    # Check family name
    if 'mens' in family_lower:
        has_male = True
    
    # Determine gender
    if has_male and not has_female:
        return 'male'
    elif has_female and not has_male:
        return 'female'
    else:
        return 'unisex'


def infer_hair_length(family_name):
    """Infer hair length from family name."""
    family_lower = family_name.lower()
    
    if 'long' in family_lower or 'braid' in family_lower:
        return ['long']
    elif 'short' in family_lower or 'buzz' in family_lower or 'crew' in family_lower or 'crop' in family_lower or 'pixie' in family_lower:
        return ['short']
    elif 'medium' in family_lower or 'lob' in family_lower or 'bob' in family_lower:
        return ['medium', 'short']
    else:
        return ['medium']  # Default


def generate_description(hairstyle_name, family_name):
    """Generate a description for the hairstyle."""
    name_parts = hairstyle_name.replace('_', ' ').title()
    family_parts = family_name.replace('_family', '').replace('_', ' ').title()
    
    return f"A stylish {name_parts} from the {family_parts} collection. Perfect for those looking for a versatile and fashionable look."


def main():
    print("="*80)
    print("POPULATE DATABASE WITH ALL 203 HAIRSTYLES")
    print("="*80)
    
    # Load training data
    csv_path = Path(__file__).parent / 'hairmixer_app' / 'ml' / 'models' / 'merged_preferences_max_voting.csv'
    print(f"\n📂 Loading data from: {csv_path}")
    
    if not csv_path.exists():
        print(f"❌ ERROR: File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} training records")
    
    # Extract unique hairstyles with their families
    print(f"\n📊 Analyzing unique hairstyles...")
    unique_styles = df[['hairstyle_family', 'hairstyle_name']].drop_duplicates()
    
    print(f"   Total unique hairstyles: {len(unique_styles)}")
    print(f"   Total unique families: {unique_styles['hairstyle_family'].nunique()}")
    
    # Group by family
    styles_by_family = defaultdict(list)
    for _, row in unique_styles.iterrows():
        styles_by_family[row['hairstyle_family']].append(row['hairstyle_name'])
    
    print(f"\n📁 Hairstyles by Family:")
    total_styles = 0
    for family, styles in sorted(styles_by_family.items()):
        print(f"   {family}: {len(styles)} styles")
        total_styles += len(styles)
    
    print(f"\n   TOTAL UNIQUE STYLES: {total_styles}")
    
    # Check current database
    current_count = Hairstyle.objects.filter(is_active=True).count()
    print(f"\n💾 Current database: {current_count} hairstyles")
    
    # Ask for confirmation
    print(f"\n⚠️  This will add {total_styles - current_count} new hairstyles to the database.")
    response = input("   Do you want to proceed? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("\n❌ Aborted by user")
        return
    
    # Ensure categories exist
    print(f"\n📂 Ensuring categories exist...")
    categories = {}
    category_names = {
        1: "Short Styles",
        2: "Long Styles",
        3: "Medium Styles",
        4: "Curly & Textured Styles",
        5: "Formal Styles",
        6: "Classic Styles",
        7: "Trendy Styles",
        8: "Retro Styles",
        9: "Casual Styles"
    }
    
    for cat_id, cat_name in category_names.items():
        category, created = HairstyleCategory.objects.get_or_create(
            id=cat_id,
            defaults={'name': cat_name, 'is_active': True}
        )
        categories[cat_id] = category
        if created:
            print(f"   ✅ Created category: {cat_name}")
        else:
            print(f"   ✓  Category exists: {cat_name}")
    
    # Populate database
    print(f"\n📝 Populating database with hairstyles...")
    
    added_count = 0
    skipped_count = 0
    
    with transaction.atomic():
        for family, styles in sorted(styles_by_family.items()):
            # Get category for this family
            category_id = FAMILY_TO_CATEGORY.get(family, 9)  # Default to Casual
            category = categories.get(category_id)
            
            print(f"\n   Processing {family} → Category {category_id}")
            
            for style_name in styles:
                # Check if already exists
                if Hairstyle.objects.filter(name__iexact=style_name).exists():
                    skipped_count += 1
                    continue
                
                # Infer attributes
                gender = infer_gender_from_name(style_name, family)
                hair_lengths = infer_hair_length(family)
                description = generate_description(style_name, family)
                
                # Create hairstyle
                hairstyle = Hairstyle.objects.create(
                    name=style_name.replace('_', ' ').title(),
                    description=description,
                    category=category,
                    suitable_gender=gender,
                    hair_lengths=hair_lengths,
                    tags=[family.replace('_family', ''), 'ai_generated'],
                    maintenance='medium',
                    difficulty='medium',
                    estimated_time=30,
                    trend_score=5.0,
                    popularity_score=5.0,
                    is_active=True
                )
                
                added_count += 1
                
                if added_count % 20 == 0:
                    print(f"      Added {added_count} hairstyles...")
    
    print(f"\n{'='*80}")
    print(f"✅ DATABASE POPULATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"   Added: {added_count} new hairstyles")
    print(f"   Skipped: {skipped_count} existing hairstyles")
    print(f"   Total in database: {Hairstyle.objects.filter(is_active=True).count()}")
    
    print(f"\n📈 Gender Distribution:")
    male_count = Hairstyle.objects.filter(is_active=True, suitable_gender='male').count()
    female_count = Hairstyle.objects.filter(is_active=True, suitable_gender='female').count()
    unisex_count = Hairstyle.objects.filter(is_active=True, suitable_gender='unisex').count()
    print(f"   Male: {male_count}")
    print(f"   Female: {female_count}")
    print(f"   Unisex: {unisex_count}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review generated hairstyles in Django admin")
    print(f"   2. Add images to hairstyles (currently no images)")
    print(f"   3. Refine descriptions and metadata")
    print(f"   4. Test recommendations with all 203 styles")
    print(f"   5. Adjust gender assignments if needed")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
