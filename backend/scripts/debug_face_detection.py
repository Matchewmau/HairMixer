"""
Debug script to visualize what MediaPipe is detecting and cropping
"""
import os
import sys
import django
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.ml.face_analyzer import FacialFeatureAnalyzer
from PIL import Image, ImageDraw
import numpy as np

def visualize_face_detection(image_path):
    """Visualize what the face detector is finding"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(image_path).name}")
    print('='*60)
    
    analyzer = FacialFeatureAnalyzer()
    
    # Load original image
    with Image.open(image_path) as img:
        img_rgb = np.array(img.convert('RGB'))
        print(f"Original image: {img_rgb.shape} ({img.size[0]}x{img.size[1]})")
    
    # Detect face
    result, error = analyzer.detect_and_analyze_face(image_path)
    
    if result and result.get('face_detected'):
        face_box = result.get('face_box', [])
        face_shape_info = result.get('face_shape', {})
        
        print(f"\n✅ Face detected!")
        print(f"Face box: {face_box}")
        
        if len(face_box) == 4:
            x1, y1, x2, y2 = face_box
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 1.0
            
            print(f"Crop dimensions: {width}x{height}")
            print(f"Aspect ratio: {aspect_ratio:.3f}")
            print(f"  {'(SQUARE)' if 0.95 < aspect_ratio < 1.05 else ''}")
            print(f"  {'(TALL/OBLONG)' if aspect_ratio < 0.9 else ''}")
            print(f"  {'(WIDE)' if aspect_ratio > 1.1 else ''}")
            
            print(f"\nPredicted shape: {face_shape_info.get('shape', 'N/A')}")
            print(f"Confidence: {face_shape_info.get('confidence', 0):.3f}")
            print(f"\nAll probabilities:")
            for shape, prob in sorted(
                face_shape_info.get('all_probabilities', {}).items(),
                key=lambda x: x[1],
                reverse=True
            ):
                bar = '█' * int(prob * 50)
                print(f"  {shape:8s}: {prob:.3f} {bar}")
            
            # Save visualization
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)
            draw.rectangle(face_box, outline='red', width=3)
            
            output_path = Path('debug_face_detection.jpg')
            img_pil.save(output_path)
            print(f"\n📸 Saved visualization to: {output_path.absolute()}")
            
            # Also save the cropped face
            face_crop = img_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face_crop)
            crop_path = Path('debug_face_crop.jpg')
            face_pil.save(crop_path)
            print(f"📸 Saved face crop to: {crop_path.absolute()}")
    else:
        print(f"\n❌ Face detection failed: {error}")

if __name__ == '__main__':
    # Find latest uploaded image
    uploads_dir = Path('media/uploads/2025/11')
    
    if uploads_dir.exists():
        images = sorted(
            list(uploads_dir.glob('*.jpg')) + 
            list(uploads_dir.glob('*.jpeg')) + 
            list(uploads_dir.glob('*.png')),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if images:
            # Analyze the 3 most recent images
            for img_path in images[:3]:
                visualize_face_detection(str(img_path))
        else:
            print("No images found in uploads directory")
    else:
        print(f"Uploads directory not found: {uploads_dir}")
