"""
Migration helper script to update existing face shape data
after ResNet50 model update (7 classes → 5 classes)

This script handles users who have 'diamond' or 'triangle' face shapes
which are no longer supported in the new model.
"""

import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.models import UserPreference


def analyze_existing_data():
    """Analyze existing face shape data"""
    print("="*80)
    print("ANALYZING EXISTING FACE SHAPE DATA")
    print("="*80)
    
    total = UserPreference.objects.count()
    print(f"\nTotal user preferences: {total}")
    
    # Count by face shape
    print("\nFace shape distribution:")
    print("-"*80)
    
    face_shapes = UserPreference.objects.values_list(
        'faceshape', flat=True
    ).distinct()
    
    for shape in sorted(face_shapes):
        if shape:
            count = UserPreference.objects.filter(faceshape=shape).count()
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {shape:10s}: {count:5d} ({percentage:5.2f}%)")
        else:
            count = UserPreference.objects.filter(faceshape='').count()
            count += UserPreference.objects.filter(faceshape__isnull=True).count()
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {'(empty)':10s}: {count:5d} ({percentage:5.2f}%)")
    
    # Check for deprecated shapes
    deprecated_shapes = ['diamond', 'triangle']
    deprecated_count = UserPreference.objects.filter(
        faceshape__in=deprecated_shapes
    ).count()
    
    print(f"\n{'='*80}")
    print(f"Deprecated face shapes (diamond, triangle): {deprecated_count}")
    
    if deprecated_count > 0:
        print(f"These need to be migrated to new face shapes.")
    else:
        print(f"No deprecated face shapes found.")
    
    return deprecated_count


def migrate_face_shapes(strategy='remap', dry_run=True):
    """
    Migrate face shapes from old model to new model
    
    Args:
        strategy: 'remap' or 'clear'
            - remap: Map diamond→heart, triangle→square
            - clear: Set to empty and reset confidence to 0
        dry_run: If True, show what would change without making changes
    """
    print("\n" + "="*80)
    print(f"MIGRATION STRATEGY: {strategy.upper()}")
    print(f"DRY RUN: {'YES' if dry_run else 'NO (MAKING CHANGES!)'}")
    print("="*80)
    
    deprecated_shapes = {
        'diamond': 'heart',    # Diamond has prominent cheekbones like heart
        'triangle': 'square'   # Triangle has strong jawline like square
    }
    
    changes = []
    
    for old_shape, new_shape in deprecated_shapes.items():
        users = UserPreference.objects.filter(faceshape=old_shape)
        count = users.count()
        
        if count > 0:
            if strategy == 'remap':
                msg = f"  {old_shape} → {new_shape}: {count} users"
                changes.append(msg)
                print(msg)
                
                if not dry_run:
                    users.update(faceshape=new_shape)
                    print(f"    ✓ Updated {count} records")
            
            elif strategy == 'clear':
                msg = f"  {old_shape} → (empty): {count} users"
                changes.append(msg)
                print(msg)
                
                if not dry_run:
                    users.update(faceshape='', faceshape_confidence=0.0)
                    print(f"    ✓ Cleared {count} records")
    
    if not changes:
        print("  No changes needed - no deprecated face shapes found.")
    
    if dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN COMPLETE - No changes were made.")
        print("Run with dry_run=False to apply changes.")
    else:
        print(f"\n{'='*80}")
        print("MIGRATION COMPLETE!")
    
    return len(changes) > 0


def verify_migration():
    """Verify that migration was successful"""
    print("\n" + "="*80)
    print("VERIFYING MIGRATION")
    print("="*80)
    
    deprecated_shapes = ['diamond', 'triangle']
    remaining = UserPreference.objects.filter(
        faceshape__in=deprecated_shapes
    ).count()
    
    if remaining == 0:
        print("✓ SUCCESS: No deprecated face shapes remain")
        
        # Show new distribution
        print("\nNew face shape distribution:")
        print("-"*80)
        
        valid_shapes = ['heart', 'oblong', 'oval', 'round', 'square']
        total = UserPreference.objects.count()
        
        for shape in valid_shapes:
            count = UserPreference.objects.filter(faceshape=shape).count()
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {shape:10s}: {count:5d} ({percentage:5.2f}%)")
        
        empty_count = UserPreference.objects.filter(faceshape='').count()
        empty_count += UserPreference.objects.filter(faceshape__isnull=True).count()
        percentage = (empty_count / total * 100) if total > 0 else 0
        print(f"  {'(empty)':10s}: {empty_count:5d} ({percentage:5.2f}%)")
        
        return True
    else:
        print(f"✗ FAILED: {remaining} deprecated face shapes still remain")
        
        for shape in deprecated_shapes:
            count = UserPreference.objects.filter(faceshape=shape).count()
            if count > 0:
                print(f"  - {shape}: {count} records")
        
        return False


def main():
    """Main migration workflow"""
    print("\n" + "#"*80)
    print("# FACE SHAPE DATA MIGRATION TOOL")
    print("# ResNet50 Model Update: 7 classes → 5 classes")
    print("#"*80)
    
    # Step 1: Analyze current data
    deprecated_count = analyze_existing_data()
    
    if deprecated_count == 0:
        print("\n✓ No migration needed - all face shapes are compatible!")
        return 0
    
    # Step 2: Show migration options
    print("\n" + "="*80)
    print("MIGRATION OPTIONS")
    print("="*80)
    print("""
Option 1: REMAP (Recommended)
  - Maps deprecated shapes to similar new shapes:
    • diamond → heart (both have prominent cheekbones)
    • triangle → square (both have strong jawlines)
  - Preserves face shape data for users
  - Maintains confidence scores

Option 2: CLEAR
  - Clears face shape and resets confidence to 0
  - Forces users to re-analyze their photos
  - Ensures 100% accuracy with new model
    """)
    
    # Step 3: Ask user to choose
    print("Choose migration strategy:")
    print("  1) Remap (default - keeps existing data)")
    print("  2) Clear (requires users to re-analyze)")
    print("  3) Cancel (exit without changes)")
    
    while True:
        choice = input("\nEnter choice [1-3] (default=1): ").strip() or "1"
        
        if choice == "1":
            strategy = "remap"
            break
        elif choice == "2":
            strategy = "clear"
            break
        elif choice == "3":
            print("Migration cancelled.")
            return 0
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Step 4: Show dry run
    print("\n" + "="*80)
    print("DRY RUN - Preview of changes")
    print("="*80)
    migrate_face_shapes(strategy=strategy, dry_run=True)
    
    # Step 5: Confirm before applying
    print("\n" + "="*80)
    confirm = input("Apply these changes? [y/N]: ").strip().lower()
    
    if confirm != 'y':
        print("Migration cancelled.")
        return 0
    
    # Step 6: Apply migration
    print("\nApplying migration...")
    migrate_face_shapes(strategy=strategy, dry_run=False)
    
    # Step 7: Verify
    success = verify_migration()
    
    if success:
        print("\n" + "="*80)
        print("✓ MIGRATION SUCCESSFUL!")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("✗ MIGRATION FAILED - Please review errors above")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
