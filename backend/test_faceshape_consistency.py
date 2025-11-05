"""
Test script to check face shape classification consistency
"""
import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.ml.face_analyzer import FacialFeatureAnalyzer
from pathlib import Path
import torch

def test_consistency(image_path, num_runs=10):
    """Test if the same image gives consistent predictions"""
    print(f"Testing consistency with {num_runs} runs...")
    print(f"Image: {image_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("-" * 60)
    
    # Create analyzer
    analyzer = FacialFeatureAnalyzer()
    
    results = []
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}")
        result, error = analyzer.detect_and_analyze_face(image_path)
        
        if result and result.get('face_detected'):
            face_shape_info = result.get('face_shape', {})
            shape = face_shape_info.get('shape', 'unknown')
            confidence = face_shape_info.get('confidence', 0.0)
            method = face_shape_info.get('method', 'unknown')
            
            print(f"  Shape: {shape}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Method: {method}")
            
            results.append({
                'shape': shape,
                'confidence': confidence,
                'method': method,
                'all_probs': face_shape_info.get('all_probabilities', {})
            })
        else:
            print(f"  ERROR: {error}")
            results.append(None)
    
    print("\n" + "=" * 60)
    print("CONSISTENCY ANALYSIS")
    print("=" * 60)
    
    # Check if all predictions are the same
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("❌ No valid predictions!")
        return False
    
    shapes = [r['shape'] for r in valid_results]
    unique_shapes = set(shapes)
    
    print(f"\nTotal runs: {num_runs}")
    print(f"Valid predictions: {len(valid_results)}")
    print(f"Unique shapes predicted: {len(unique_shapes)}")
    print(f"Shapes: {unique_shapes}")
    
    # Count occurrences
    from collections import Counter
    shape_counts = Counter(shapes)
    print("\nShape distribution:")
    for shape, count in shape_counts.most_common():
        percentage = (count / len(valid_results)) * 100
        print(f"  {shape}: {count}/{len(valid_results)} ({percentage:.1f}%)")
    
    # Check confidence variance
    confidences = [r['confidence'] for r in valid_results]
    import numpy as np
    mean_conf = np.mean(confidences)
    std_conf = np.std(confidences)
    
    print(f"\nConfidence statistics:")
    print(f"  Mean: {mean_conf:.4f}")
    print(f"  Std Dev: {std_conf:.4f}")
    print(f"  Min: {min(confidences):.4f}")
    print(f"  Max: {max(confidences):.4f}")
    
    # Determine if consistent
    if len(unique_shapes) == 1 and std_conf < 0.001:
        print("\n✅ CONSISTENT: All predictions are identical!")
        return True
    elif len(unique_shapes) == 1:
        print(f"\n⚠️  MOSTLY CONSISTENT: Same shape but confidence varies (std={std_conf:.4f})")
        return True
    else:
        print(f"\n❌ INCONSISTENT: Multiple shapes predicted!")
        print("This indicates non-deterministic behavior in the model.")
        return False

if __name__ == '__main__':
    # You need to provide a test image path
    # Example: test_image = Path(r"D:\CODING\Python\HairMixer\backend\media\uploads\test.jpg")
    
    # Try to find an uploaded image
    media_dir = Path(__file__).parent / 'media' / 'uploads'
    
    if media_dir.exists():
        images = list(media_dir.rglob('*.jpg')) + list(media_dir.rglob('*.png'))
        if images:
            test_image = images[0]
            print(f"Found test image: {test_image}")
            test_consistency(str(test_image), num_runs=10)
        else:
            print("No images found in media/uploads directory")
            print("Please provide an image path manually")
    else:
        print(f"Media directory not found: {media_dir}")
        print("Please provide an image path manually")
