import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import logging
from pathlib import Path
from django.conf import settings
import os

logger = logging.getLogger(__name__)

class MobileNetV3FaceShapeClassifier(nn.Module):
    """MobileNetV3-Small model for face shape classification"""
    
    def __init__(self, num_classes=7, pretrained=False):
        super(MobileNetV3FaceShapeClassifier, self).__init__()
        
        # Load MobileNetV3-Small backbone
        self.backbone = mobilenet_v3_small(pretrained=pretrained)
        
        # Replace the classifier for face shape classification
        # MobileNetV3-Small has a classifier with in_features=576
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class DirectMobileNetV3FaceShapeClassifier(nn.Module):
    """Direct MobileNetV3-Small model matching the saved state_dict structure"""
    
    def __init__(self, num_classes=7, pretrained=False):
        super(DirectMobileNetV3FaceShapeClassifier, self).__init__()
        
        # Load MobileNetV3-Small directly without wrapper
        model = mobilenet_v3_small(pretrained=pretrained)
        
        # Copy the structure directly
        self.features = model.features
        
        # Create classifier to match your trained model structure: 576->1024->512->7
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),      # classifier.0
            nn.Hardswish(),            # classifier.1 
            nn.Dropout(0.2),           # classifier.2
            nn.Linear(1024, 512),      # classifier.3
            nn.Hardswish(),            # classifier.4
            nn.Dropout(0.2),           # classifier.5
            nn.Linear(512, num_classes) # classifier.6
        )
        
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetModelLoader:
    """Utility class to load and manage the MobileNetV3 model"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 7  # Update based on your training
        self.model_path = None
        
    def load_model(self, model_path=None):
        """Load the trained MobileNetV3 model"""
        try:
            if model_path is None:
                # Default model path - update this to your model location
                model_path = self._get_default_model_path()
            
            print(f"DEBUG: Attempting to load model from: {model_path}")
            print(f"DEBUG: File exists: {os.path.exists(model_path)}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            logger.info(f"Loading MobileNetV3 model from: {model_path}")
            
            # Create model instance - try direct structure first
            self.model = DirectMobileNetV3FaceShapeClassifier(
                num_classes=self.num_classes, 
                pretrained=False
            )
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the entire dict is the state_dict
                    self.model.load_state_dict(checkpoint)
            else:
                # Assume checkpoint is the state_dict directly
                self.model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("MobileNetV3 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading MobileNetV3 model: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _get_default_model_path(self):
        """Get the default path for the model file"""
        # You can update this path to match where you place your .pth file
        models_dir = Path(settings.BASE_DIR) / 'hairmixer_app' / 'ml' / 'models'
        return models_dir / 'mobilenetv3_small.pth'
    
    def predict(self, input_tensor):
        """Make prediction with the loaded model"""
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                
                return {
                    'predicted_class': predicted_class.cpu().numpy()[0],
                    'probabilities': probabilities.cpu().numpy()[0],
                    'confidence': float(torch.max(probabilities).cpu().numpy())
                }
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None


# Global model loader instance
mobile_net_loader = MobileNetModelLoader()
