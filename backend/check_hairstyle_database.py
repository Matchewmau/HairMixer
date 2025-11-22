"""
Check what fields exist in the Hairstyle database model and their values.
"""

import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.models import Hairstyle
from django.db import connection


def check_database_schema():
    """Check the actual database schema for Hairstyle table"""
    
    print("=" * 80)
    print("HAIRSTYLE DATABASE SCHEMA CHECK")
    print("=" * 80)
    print()
    
    # Get all field names from the model
    model_fields = [f.name for f in Hairstyle._meta.get_fields()]
    
    print("Fields defined in Hairstyle model:")
    print("-" * 80)
    for i, field in enumerate(sorted(model_fields), 1):
        field_obj = Hairstyle._meta.get_field(field)
        field_type = field_obj.__class__.__name__
        print(f"{i:2d}. {field:30s} ({field_type})")
    
    print()
    print(f"Total fields: {len(model_fields)}")
    print()
    
    # Check for hair-related fields
    print("=" * 80)
    print("HAIR-RELATED FIELDS")
    print("=" * 80)
    hair_fields = [f for f in model_fields if 'hair' in f.lower()]
    
    if hair_fields:
        print("Found hair-related fields:")
        for field in hair_fields:
            print(f"  ✅ {field}")
    else:
        print("  ❌ No hair-related fields found")
    
    print()
    
    # Check specifically for color and condition
    print("=" * 80)
    print("SPECIFIC CHECKS")
    print("=" * 80)
    
    color_fields = [f for f in model_fields if 'color' in f.lower()]
    condition_fields = [f for f in model_fields if 'condition' in f.lower()]
    
    print("\n🎨 Hair Color Fields:")
    if color_fields:
        for field in color_fields:
            print(f"  ✅ {field}")
    else:
        print("  ❌ No hair_color field found in database")
    
    print("\n🏥 Hair Condition Fields:")
    if condition_fields:
        for field in condition_fields:
            print(f"  ✅ {field}")
    else:
        print("  ❌ No hair_condition field found in database")
    
    # Get sample data
    print("\n" + "=" * 80)
    print("SAMPLE DATA (First 5 Hairstyles)")
    print("=" * 80)
    
    sample_styles = Hairstyle.objects.all()[:5]
    
    if sample_styles.exists():
        for style in sample_styles:
            print(f"\nHairstyle: {style.name}")
            print(f"  ID: {style.id}")
            
            # Display all field values
            for field in sorted(model_fields):
                if hasattr(style, field):
                    value = getattr(style, field)
                    # Handle different types
                    if hasattr(value, 'all'):  # Many-to-many or reverse relation
                        continue
                    if value is not None:
                        # Truncate long strings
                        if isinstance(value, str) and len(str(value)) > 50:
                            value = str(value)[:50] + "..."
                        print(f"  • {field}: {value}")
    else:
        print("❌ No hairstyles found in database")
        print("\nRun populate_all_hairstyles.py to add hairstyles.")
    
    # Check actual database columns
    print("\n" + "=" * 80)
    print("ACTUAL DATABASE COLUMNS")
    print("=" * 80)
    
    with connection.cursor() as cursor:
        cursor.execute(f"PRAGMA table_info({Hairstyle._meta.db_table})")
        columns = cursor.fetchall()
        
        print(f"\nColumns in '{Hairstyle._meta.db_table}' table:")
        print("-" * 80)
        for col in columns:
            col_id, name, col_type, not_null, default, pk = col
            print(f"{col_id:2d}. {name:30s} {col_type:15s} {'NOT NULL' if not_null else ''} {'PRIMARY KEY' if pk else ''}")
    
    print()
    
    # Check for hair_color and hair_condition in actual DB
    db_column_names = [col[1] for col in columns]
    
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()
    
    has_color = any('color' in col.lower() for col in db_column_names)
    has_condition = any('condition' in col.lower() for col in db_column_names)
    
    print("🎨 Hair Color in Database:", "✅ YES" if has_color else "❌ NO")
    print("🏥 Hair Condition in Database:", "✅ YES" if has_condition else "❌ NO")
    print()
    
    if not has_color and not has_condition:
        print("⚠️  CONCLUSION:")
        print("   Hair color and hair condition are NOT stored in the database.")
        print("   These attributes CANNOT affect hairstyle recommendations.")
        print()
        print("💡 TO ADD THESE FEATURES:")
        print("   1. Add 'hair_color' and 'hair_condition' fields to Hairstyle model")
        print("   2. Run: python manage.py makemigrations")
        print("   3. Run: python manage.py migrate")
        print("   4. Update hairstyle data with color/condition values")
        print("   5. Update recommendation_engine.py to use these fields")
    
    print("=" * 80)


if __name__ == '__main__':
    check_database_schema()