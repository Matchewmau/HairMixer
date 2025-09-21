import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import logging
from pathlib import Path
from django.conf import settings
import os
import time

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
    """
    Direct MobileNetV3-Small model matching the saved
    state_dict structure
    """
    
    def __init__(self, num_classes=7, pretrained=False):
        super(DirectMobileNetV3FaceShapeClassifier, self).__init__()
        
        # Load MobileNetV3-Small directly without wrapper
        model = mobilenet_v3_small(pretrained=pretrained)
        
        # Copy the structure directly
        self.features = model.features
        
        # Create classifier to match your trained model structure:
        # 576->1024->512->7
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),      # classifier.0
            nn.Hardswish(),            # classifier.1
            nn.Dropout(0.2),           # classifier.2
            nn.Linear(1024, 512),      # classifier.3
            nn.Hardswish(),            # classifier.4
            nn.Dropout(0.2),           # classifier.5
            nn.Linear(512, num_classes)  # classifier.6
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
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        self.num_classes = 7  # Update based on your training
        self.model_path = None
        
    def load_model(self, model_path=None):
        """Load the trained MobileNetV3 model"""
        try:
            if model_path is None:
                # Default model path - update this to your model location
                model_path = self._get_default_model_path()
            model_path_str = str(model_path)

            logger.info(
                "Model load start",
                extra={
                    "event": "model_load_start",
                    "model": "MobileNetV3Small",
                    "path": model_path_str,
                    "device": str(self.device),
                    "torch": torch.__version__,
                    "cuda": bool(torch.cuda.is_available()),
                },
            )

            logger.debug(
                "Checking model path",
                extra={
                    "exists": os.path.exists(model_path_str),
                    "path": model_path_str,
                },
            )
            
            if not os.path.exists(model_path_str):
                logger.error(
                    "Model file not found",
                    extra={
                        "event": "model_load_error",
                        "path": model_path_str,
                    },
                )
                return False
            
            try:
                size_bytes = os.path.getsize(model_path_str)
            except OSError:
                size_bytes = None

            t0 = time.perf_counter()
            
            # Create model instance - try direct structure first
            self.model = DirectMobileNetV3FaceShapeClassifier(
                num_classes=self.num_classes,
                pretrained=False,
            )
            
            # Load trained weights
            checkpoint = torch.load(model_path_str, map_location=self.device)
            
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
            t1 = time.perf_counter()

            try:
                num_params = sum(p.numel() for p in self.model.parameters())
            except Exception:
                num_params = None

            logger.info(
                "Model load success",
                extra={
                    "event": "model_load_success",
                    "model": "MobileNetV3Small",
                    "path": model_path_str,
                    "size_bytes": size_bytes,
                    "device": str(self.device),
                    "load_time_ms": int((t1 - t0) * 1000),
                    "num_params": num_params,
                    "classes": self.num_classes,
                },
            )
            return True
            
        except Exception as e:
            logger.error(
                "Model load exception",
                extra={
                    "event": "model_load_error",
                    "model": "MobileNetV3Small",
                    "error": str(e),
                },
            )
            import traceback
            logger.debug(
                "Model load traceback",
                extra={"trace": traceback.format_exc()},
            )
            return False
    
    def _get_default_model_path(self):
        """Get the default path for the model file"""
        # You can update this path to match where you place your .pth file
        models_dir = (
            Path(settings.BASE_DIR) / 'hairmixer_app' / 'ml' / 'models'
        )
        return models_dir / 'mobilenetv3_small.pth'
    
    def predict(self, input_tensor):
        """Make prediction with the loaded model"""
        if self.model is None:
            logger.error(
                "Predict called without model",
                extra={"event": "model_predict_error", "reason": "not_loaded"},
            )
            return None
        
        try:
            t0 = time.perf_counter()
            try:
                shape = tuple(input_tensor.shape)  # type: ignore[attr-defined]
                batch_size = shape[0] if len(shape) > 0 else None
            except Exception:
                shape = None
                batch_size = None

            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence_tensor = torch.max(probabilities)
                t1 = time.perf_counter()
                
                pred_idx = int(predicted_class.cpu().numpy()[0])
                confidence = float(confidence_tensor.cpu().numpy())

                logger.info(
                    "Model prediction",
                    extra={
                        "event": "model_predict",
                        "model": "MobileNetV3Small",
                        "device": str(self.device),
                        "batch": batch_size,
                        "input_shape": shape,
                        "predicted_class": pred_idx,
                        "confidence": round(confidence, 6),
                        "latency_ms": int((t1 - t0) * 1000),
                    },
                )

                # Optional top-k details at debug level
                try:
                    topk_vals, topk_idx = probabilities.topk(3, dim=1)
                    logger.debug(
                        "TopK probabilities",
                        extra={
                            "topk_idx": topk_idx.cpu().numpy()[0].tolist(),
                            "topk_vals": [
                                float(v)
                                for v in topk_vals.cpu().numpy()[0].tolist()
                            ],
                        },
                    )
                except Exception:
                    pass

                return {
                    'predicted_class': pred_idx,
                    'probabilities': probabilities.cpu().numpy()[0],
                    'confidence': confidence,
                }
                
        except Exception as e:
            logger.error(
                "Prediction exception",
                extra={
                    "event": "model_predict_error",
                    "error": str(e),
                },
            )
            return None
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None


# Global model loader instance
mobile_net_loader = MobileNetModelLoader()
