"""
Simple Django management command to test face detection directly
"""
from django.core.management.base import BaseCommand
from hairmixer_app.ml.face_analyzer import analyze_face_comprehensive
import os
import glob

class Command(BaseCommand):
    help = 'Test face detection directly using Django environment'

    def handle(self, *args, **options):
        self.stdout.write("🔍 Testing face detection directly in Django environment...")
        
        # Find test image
        uploads_dir = 'uploads/2025/08'
        if not os.path.exists(uploads_dir):
            self.stdout.write(self.style.ERROR('Uploads directory not found'))
            return
        
        image_files = glob.glob(os.path.join(uploads_dir, '*'))
        if not image_files:
            self.stdout.write(self.style.ERROR('No test images found'))
            return
        
        test_image = image_files[0]
        self.stdout.write(f"📷 Using test image: {test_image}")
        
        try:
            result = analyze_face_comprehensive(test_image)
            
            self.stdout.write("📊 Analysis Result:")
            self.stdout.write(f"  Face Detected: {result.get('face_detected', False)}")
            self.stdout.write(f"  Face Shape: {result.get('face_shape', 'N/A')}")
            self.stdout.write(f"  Confidence: {result.get('confidence', 0.0)}")
            self.stdout.write(f"  Detection Method: {result.get('detection_method', 'N/A')}")
            
            if result.get('face_detected'):
                self.stdout.write(self.style.SUCCESS("✅ Face detection SUCCESSFUL!"))
            else:
                self.stdout.write(self.style.ERROR("❌ Face detection FAILED"))
                self.stdout.write(f"  Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"💥 Exception: {str(e)}"))
            import traceback
            traceback.print_exc()
