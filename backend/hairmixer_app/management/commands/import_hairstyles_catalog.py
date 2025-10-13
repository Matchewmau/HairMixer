"""
Management command to import hairstyles from the complete hairstyles catalog CSV.

This command populates the database with hairstyles from the 
complete_hairstyles_catalog.csv file, creating or updating entries
in the Hairstyle and HairstyleCategory models.

Usage:
    python manage.py import_hairstyles_catalog
    
    # With custom path:
    python manage.py import_hairstyles_catalog --catalog-path /path/to/catalog.csv
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from pathlib import Path
import pandas as pd
import ast
import logging

from hairmixer_app.models import Hairstyle, HairstyleCategory

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Import hairstyles from catalog CSV file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--catalog-path',
            type=str,
            help='Path to the catalog CSV file',
            default=None
        )
        parser.add_argument(
            '--skip-existing',
            action='store_true',
            help='Skip hairstyles that already exist in database'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be imported without actually importing'
        )

    def handle(self, *args, **options):
        catalog_path = options.get('catalog_path')
        skip_existing = options.get('skip_existing', False)
        dry_run = options.get('dry_run', False)
        
        # Determine catalog path
        if catalog_path:
            catalog_file = Path(catalog_path)
        else:
            # Try multiple default locations
            base_dir = Path(__file__).parent.parent.parent.parent.parent
            possible_paths = [
                # Location 1: data/catalog/ (original expected location)
                base_dir / 'data' / 'catalog' / 
                'complete_hairstyles_catalog.csv',
                # Location 2: In ML models directory (actual location)
                Path(__file__).parent.parent.parent / 'ml' / 'models' /
                'hairstyle_recommender' / 'complete_hairstyles_catalog.csv',
            ]
            
            catalog_file = None
            for path in possible_paths:
                if path.exists():
                    catalog_file = path
                    break
            
            if catalog_file is None:
                raise CommandError(
                    f'Catalog file not found in any of these locations:\n' +
                    '\n'.join(f'  - {p}' for p in possible_paths) +
                    '\nPlease specify the correct path using --catalog-path'
                )
        
        self.stdout.write(
            self.style.SUCCESS(f'Reading catalog from: {catalog_file}')
        )
        
        try:
            df = pd.read_csv(catalog_file)
        except Exception as e:
            raise CommandError(f'Failed to read CSV file: {e}')
        
        self.stdout.write(
            f'Found {len(df)} hairstyles in catalog'
        )
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING('DRY RUN MODE - No changes will be made')
            )
        
        created_count = 0
        updated_count = 0
        skipped_count = 0
        error_count = 0
        
        with transaction.atomic():
            for idx, row in df.iterrows():
                try:
                    result = self._import_hairstyle(
                        row, skip_existing, dry_run
                    )
                    if result == 'created':
                        created_count += 1
                    elif result == 'updated':
                        updated_count += 1
                    elif result == 'skipped':
                        skipped_count += 1
                        
                except Exception as e:
                    error_count += 1
                    self.stdout.write(
                        self.style.ERROR(
                            f'Error importing row {idx}: {e}'
                        )
                    )
                    logger.error(
                        f'Error importing hairstyle row {idx}: {e}', 
                        exc_info=True
                    )
            
            if dry_run:
                # Rollback transaction in dry run mode
                transaction.set_rollback(True)
        
        # Print summary
        self.stdout.write(
            self.style.SUCCESS('\n' + '='*60)
        )
        self.stdout.write(
            self.style.SUCCESS('Import Summary:')
        )
        self.stdout.write(f'  Created: {created_count}')
        self.stdout.write(f'  Updated: {updated_count}')
        self.stdout.write(f'  Skipped: {skipped_count}')
        if error_count > 0:
            self.stdout.write(
                self.style.ERROR(f'  Errors: {error_count}')
            )
        self.stdout.write(
            self.style.SUCCESS('='*60)
        )
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    '\nDRY RUN - No changes were saved to database'
                )
            )

    def _import_hairstyle(self, row, skip_existing, dry_run):
        """Import a single hairstyle from CSV row"""
        # Try multiple column names for hairstyle name
        hairstyle_name = (
            row.get('name') or 
            row.get('hairstyle_name') or 
            row.get('style_name')
        )
        
        if not hairstyle_name or pd.isna(hairstyle_name):
            raise ValueError('Hairstyle name is required')
        
        # Check if hairstyle already exists (handle duplicates)
        existing_qs = Hairstyle.objects.filter(name=hairstyle_name)
        
        if existing_qs.exists():
            if skip_existing:
                return 'skipped'
            # If duplicates exist, clean them up by keeping only first one
            if existing_qs.count() > 1:
                # Keep first, delete others
                to_keep = existing_qs.first()
                existing_qs.exclude(id=to_keep.id).delete()
                existing = to_keep
            else:
                existing = existing_qs.first()
        else:
            existing = None
        
        # Get or create category
        category_name = row.get('category', 'Uncategorized')
        if pd.isna(category_name):
            category_name = 'Uncategorized'
            
        category = None
        if not dry_run:
            category, _ = HairstyleCategory.objects.get_or_create(
                name=category_name,
                defaults={'is_active': True}
            )
        
        # Parse JSON fields
        def safe_parse_list(value, default=None):
            """Safely parse list fields from CSV"""
            if default is None:
                default = []
            if pd.isna(value) or value == '' or value is None:
                return default
            if isinstance(value, str):
                try:
                    # Try to parse as Python literal
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # If parsing fails, split by comma
                    return [v.strip() for v in value.split(',') if v.strip()]
            if isinstance(value, list):
                return value
            return default
        
        # Get description from available fields
        description = (
            row.get('description') or 
            row.get('major_rule') or 
            ''
        )
        if pd.isna(description):
            description = ''
        
        # Get face shapes
        face_shapes_list = safe_parse_list(row.get('face_shapes'))
        if not face_shapes_list and 'face_shape_name' in row:
            face_shape = row.get('face_shape_name')
            if face_shape and not pd.isna(face_shape):
                face_shapes_list = [str(face_shape).lower()]
        
        # Prepare hairstyle data
        defaults = {
            'description': str(description),
            'category': category,
            'tags': safe_parse_list(row.get('tags')),
            'face_shapes': face_shapes_list,
            'hair_types': safe_parse_list(row.get('hair_types')),
            'hair_lengths': safe_parse_list(row.get('hair_lengths')),
            'occasions': safe_parse_list(row.get('occasions')),
            'maintenance': str(row.get('maintenance', 'medium')).lower(),
            'difficulty': str(row.get('difficulty', 'medium')).lower(),
            'is_active': True,
        }
        
        # Handle numeric fields
        if 'estimated_time' in row and pd.notna(row.get('estimated_time')):
            try:
                defaults['estimated_time'] = int(row['estimated_time'])
            except (ValueError, TypeError):
                defaults['estimated_time'] = 30
        else:
            defaults['estimated_time'] = 30
        
        if 'trend_score' in row and pd.notna(row.get('trend_score')):
            try:
                defaults['trend_score'] = float(row['trend_score'])
            except (ValueError, TypeError):
                defaults['trend_score'] = 5.0
        else:
            defaults['trend_score'] = 5.0
        
        if 'popularity_score' in row and pd.notna(row.get('popularity_score')):
            try:
                defaults['popularity_score'] = float(row['popularity_score'])
            except (ValueError, TypeError):
                defaults['popularity_score'] = 5.0
        else:
            defaults['popularity_score'] = 5.0
        
        # Handle suitable_gender field
        gender_field = row.get('suitable_gender') or row.get('gender')
        if gender_field and pd.notna(gender_field):
            gender = str(gender_field).lower()
            if gender in ['male', 'female', 'unisex']:
                defaults['suitable_gender'] = gender
            else:
                defaults['suitable_gender'] = 'unisex'
        else:
            defaults['suitable_gender'] = 'unisex'
        
        # Handle styling tips and products
        if 'styling_tips' in row and pd.notna(row.get('styling_tips')):
            defaults['styling_tips'] = str(row['styling_tips'])
        
        if 'products_needed' in row:
            defaults['products_needed'] = safe_parse_list(
                row.get('products_needed')
            )
        
        if 'seo_keywords' in row:
            defaults['seo_keywords'] = safe_parse_list(
                row.get('seo_keywords')
            )
        
        # Handle image URLs
        if 'image_url' in row and pd.notna(row.get('image_url')):
            defaults['image_url'] = str(row['image_url'])
        
        if dry_run:
            if existing:
                self.stdout.write(
                    f'  Would update: {hairstyle_name}'
                )
                return 'updated'
            else:
                self.stdout.write(
                    f'  Would create: {hairstyle_name}'
                )
                return 'created'
        
        # Create or update hairstyle
        if existing:
            # Update existing hairstyle
            for key, value in defaults.items():
                setattr(existing, key, value)
            existing.save()
            self.stdout.write(
                self.style.WARNING(f'  ↻ Updated: {hairstyle_name}')
            )
            return 'updated'
        else:
            # Create new hairstyle
            hairstyle = Hairstyle.objects.create(
                name=hairstyle_name,
                **defaults
            )
            self.stdout.write(
                self.style.SUCCESS(f'  ✓ Created: {hairstyle_name}')
            )
            return 'created'
