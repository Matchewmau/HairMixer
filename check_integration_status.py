"""
System Status Check for RF No-Family Hairstyle Recommender

This script checks the status of all components required for the
RF No-Family hairstyle recommender integration.

Usage:
    python check_integration_status.py
"""

import os
import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_model_files():
    """Check if all model files exist"""
    print(f"\n{BLUE}{'='*60}")
    print("1. MODEL FILES CHECK")
    print(f"{'='*60}{RESET}\n")
    
    base_path = Path(__file__).parent / 'backend' / 'hairmixer_app' / 'ml' / 'models' / 'hairstyle_recommender'
    
    required_files = [
        'rf_name_no_family.pkl',
        'label_encoders_no_family.pkl',
        'feature_columns_no_family.pkl',
        'model_metadata_no_family.json'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = base_path / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"{GREEN}✓{RESET} {filename} ({size:.1f} MB)")
        else:
            print(f"{RED}✗{RESET} {filename} (NOT FOUND)")
            all_exist = False
    
    return all_exist


def check_code_files():
    """Check if all code files are created"""
    print(f"\n{BLUE}{'='*60}")
    print("2. CODE FILES CHECK")
    print(f"{'='*60}{RESET}\n")
    
    base_path = Path(__file__).parent
    
    files_to_check = [
        ('ML Module', 'backend/hairmixer_app/ml/hairstyle_recommender.py'),
        ('Management Command', 'backend/hairmixer_app/management/commands/import_hairstyles_catalog.py'),
        ('Test Script', 'backend/test_rf_recommender.py'),
        ('Integration Guide', 'RF_INTEGRATION_GUIDE.md'),
        ('Implementation Checklist', 'IMPLEMENTATION_CHECKLIST.md')
    ]
    
    all_exist = True
    for name, filepath in files_to_check:
        full_path = base_path / filepath
        if full_path.exists():
            print(f"{GREEN}✓{RESET} {name}")
        else:
            print(f"{RED}✗{RESET} {name} (NOT FOUND)")
            all_exist = False
    
    return all_exist


def check_catalog_file():
    """Check if hairstyles catalog exists"""
    print(f"\n{BLUE}{'='*60}")
    print("3. CATALOG FILE CHECK")
    print(f"{'='*60}{RESET}\n")
    
    base_path = Path(__file__).parent
    catalog_path = base_path / 'backend' / 'hairmixer_app' / 'ml' / 'models' / 'hairstyle_recommender' / 'complete_hairstyles_catalog.csv'
    
    if catalog_path.exists():
        size = catalog_path.stat().st_size / 1024  # KB
        print(f"{GREEN}✓{RESET} complete_hairstyles_catalog.csv ({size:.1f} KB)")
        return True
    else:
        print(f"{YELLOW}⚠{RESET} complete_hairstyles_catalog.csv (NOT FOUND)")
        print(f"   Expected location: {catalog_path}")
        return False


def check_django_setup():
    """Check if Django can be imported"""
    print(f"\n{BLUE}{'='*60}")
    print("4. DJANGO SETUP CHECK")
    print(f"{'='*60}{RESET}\n")
    
    try:
        # Try to setup Django
        backend_path = Path(__file__).parent / 'backend'
        sys.path.insert(0, str(backend_path))
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
        
        import django
        django.setup()
        
        print(f"{GREEN}✓{RESET} Django initialized successfully")
        print(f"   Version: {django.get_version()}")
        return True
        
    except Exception as e:
        print(f"{RED}✗{RESET} Django initialization failed")
        print(f"   Error: {e}")
        return False


def check_model_loading():
    """Check if model can be loaded"""
    print(f"\n{BLUE}{'='*60}")
    print("5. MODEL LOADING CHECK")
    print(f"{'='*60}{RESET}\n")
    
    try:
        from hairmixer_app.ml.hairstyle_recommender import get_hairstyle_recommender
        
        recommender = get_hairstyle_recommender()
        
        if recommender.is_available():
            info = recommender.get_model_info()
            print(f"{GREEN}✓{RESET} Model loaded successfully")
            print(f"   Model type: {info.get('model_type')}")
            print(f"   Features: {info.get('n_features')}")
            print(f"   Classes: {info.get('n_classes')}")
            return True
        else:
            print(f"{RED}✗{RESET} Model not available")
            return False
            
    except Exception as e:
        print(f"{RED}✗{RESET} Model loading failed")
        print(f"   Error: {e}")
        return False


def check_database():
    """Check database status"""
    print(f"\n{BLUE}{'='*60}")
    print("6. DATABASE CHECK")
    print(f"{'='*60}{RESET}\n")
    
    try:
        from hairmixer_app.models import Hairstyle, HairstyleCategory
        
        total = Hairstyle.objects.count()
        active = Hairstyle.objects.filter(is_active=True).count()
        categories = HairstyleCategory.objects.count()
        
        if total > 0:
            print(f"{GREEN}✓{RESET} Database populated")
            print(f"   Total hairstyles: {total}")
            print(f"   Active hairstyles: {active}")
            print(f"   Categories: {categories}")
            return True
        else:
            print(f"{YELLOW}⚠{RESET} No hairstyles in database")
            print(f"   Run: python manage.py import_hairstyles_catalog")
            return False
            
    except Exception as e:
        print(f"{RED}✗{RESET} Database check failed")
        print(f"   Error: {e}")
        return False


def print_summary(results):
    """Print overall summary"""
    print(f"\n{BLUE}{'='*60}")
    print("INTEGRATION STATUS SUMMARY")
    print(f"{'='*60}{RESET}\n")
    
    status_map = {
        'Model Files': results[0],
        'Code Files': results[1],
        'Catalog File': results[2],
        'Django Setup': results[3],
        'Model Loading': results[4],
        'Database': results[5]
    }
    
    for component, status in status_map.items():
        if status:
            print(f"{GREEN}✓{RESET} {component}: READY")
        else:
            print(f"{RED}✗{RESET} {component}: NOT READY")
    
    all_ready = all(results)
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    if all_ready:
        print(f"{GREEN}🎉 ALL COMPONENTS READY{RESET}")
        print(f"\nYou can now:")
        print(f"  1. Run tests: python backend/test_rf_recommender.py")
        print(f"  2. Start server: python backend/manage.py runserver")
        print(f"  3. Test API endpoints")
    else:
        print(f"{YELLOW}⚠ SOME COMPONENTS NOT READY{RESET}")
        print(f"\nNext steps:")
        if not results[0]:
            print(f"  - Copy model files to backend/hairmixer_app/ml/models/hairstyle_recommender/")
        if not results[2]:
            print(f"  - Place complete_hairstyles_catalog.csv in data/catalog/")
        if not results[5]:
            print(f"  - Run: python backend/manage.py import_hairstyles_catalog")
    print(f"{BLUE}{'='*60}{RESET}\n")


def main():
    """Run all checks"""
    print(f"\n{BLUE}{'='*60}")
    print("RF NO-FAMILY MODEL - INTEGRATION STATUS CHECK")
    print(f"{'='*60}{RESET}")
    
    # Run checks sequentially to handle dependencies
    model_files_ok = check_model_files()
    code_files_ok = check_code_files()
    catalog_ok = check_catalog_file()
    django_ok = check_django_setup()
    
    # Only check these if Django is set up
    model_loading_ok = check_model_loading() if django_ok else False
    database_ok = check_database() if django_ok else False
    
    results = [
        model_files_ok,
        code_files_ok,
        catalog_ok,
        django_ok,
        model_loading_ok,
        database_ok
    ]
    
    print_summary(results)


if __name__ == '__main__':
    main()
