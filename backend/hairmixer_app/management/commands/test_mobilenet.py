from django.core.management.base import BaseCommand
from django.conf import settings
from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Test MobileNetV3 face shape classifier'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-path',
            type=str,
            help='Path to the MobileNetV3 .pth model file',
        )
        parser.add_argument(
            '--image-path',
            type=str,
            help='Path to test image',
        )

    def handle(self, *args, **options):
        try:
            from hairmixer_app.ml.mobilenet_classifier import mobile_net_loader
            from hairmixer_app.ml.face_analyzer import FacialFeatureAnalyzer
            
            self.stdout.write("Testing MobileNetV3 Face Shape Classifier...")
            
            # Test model loading
            model_path = options.get('model_path')
            if model_path:
                self.stdout.write(f"Loading model from: {model_path}")
                success = mobile_net_loader.load_model(model_path)
            else:
                self.stdout.write("Loading model from default location...")
                success = mobile_net_loader.load_model()
            
            if success:
                self.stdout.write(
                    self.style.SUCCESS("✅ MobileNetV3 model loaded successfully!")
                )
            else:
                self.stdout.write(
                    self.style.ERROR("❌ Failed to load MobileNetV3 model")
                )
                return
            
            # Test face analyzer integration
            self.stdout.write("Testing face analyzer integration...")
            analyzer = FacialFeatureAnalyzer()
            
            # Test with image if provided
            image_path = options.get('image_path', 'D:/CODING/Python/HairMixer/frontend/public/dashboard/formal.jpg')
            if image_path and Path(image_path).exists():
                self.stdout.write(f"Testing with image: {image_path}")
                
                result, error = analyzer.detect_and_analyze_face(image_path)
                
                if result:
                    self.stdout.write(
                        self.style.SUCCESS(f"✅ Face analysis successful!")
                    )
                    self.stdout.write(f"Face shape: {result.get('face_shape', 'N/A')}")
                    self.stdout.write(f"Confidence: {result.get('confidence', 'N/A')}")
                    self.stdout.write(f"Method: {result.get('method', 'N/A')}")
                    
                    if 'all_probabilities' in result:
                        self.stdout.write("All probabilities:")
                        for shape, prob in result['all_probabilities'].items():
                            self.stdout.write(f"  {shape}: {prob:.3f}")
                else:
                    self.stdout.write(
                        self.style.ERROR(f"❌ Face analysis failed: {error}")
                    )
            else:
                self.stdout.write("No test image provided. Use --image-path to test with an image.")
            
            # Display model info
            if mobile_net_loader.is_loaded():
                self.stdout.write("\nModel Information:")
                self.stdout.write(f"Device: {mobile_net_loader.device}")
                self.stdout.write(f"Number of classes: {mobile_net_loader.num_classes}")
                
                # Test face shapes mapping
                from hairmixer_app.ml.model import FACE_SHAPES
                self.stdout.write(f"Face shape classes: {FACE_SHAPES}")
            
        except ImportError as e:
            self.stdout.write(
                self.style.ERROR(f"Import error: {e}")
            )
            self.stdout.write("Make sure all required packages are installed.")
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Error: {e}")
            )
            import traceback
            self.stdout.write(traceback.format_exc())
