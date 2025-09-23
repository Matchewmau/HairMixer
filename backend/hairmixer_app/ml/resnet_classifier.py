import torch
import torch.nn as nn
from torchvision.models import resnet50
import logging
from pathlib import Path
from django.conf import settings
import os
import time

logger = logging.getLogger(__name__)


class ResNet50FaceShapeClassifier(nn.Module):
    """ResNet50 model adapted for face shape classification"""

    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        super().__init__()
        # Avoid attempting to download weights by default
        model = resnet50(weights=None if not pretrained else None)
        in_features = model.fc.in_features  # 2048 for ResNet50
        # Match checkpoint structure (indices with weights: 1,5,9,12)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.0),                 # 0 (no weights)
            nn.Linear(in_features, 2048),      # 1
            nn.ReLU(inplace=True),             # 2
            nn.BatchNorm1d(2048),              # 3
            nn.Dropout(p=0.2),                 # 4
            nn.Linear(2048, 1024),             # 5
            nn.ReLU(inplace=True),             # 6
            nn.BatchNorm1d(1024),              # 7
            nn.Dropout(p=0.2),                 # 8
            nn.Linear(1024, 512),              # 9
            nn.ReLU(inplace=True),             # 10
            nn.Dropout(p=0.2),                 # 11
            nn.Linear(512, num_classes),       # 12
        )
        self.model = model

    def forward(self, x):
        return self.model(x)


class ResNetModelLoader:
    """Utility to load and serve predictions from a ResNet50 classifier"""

    def __init__(self):
        self.model: nn.Module | None = None
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        self.num_classes = 7
        self.model_path: str | None = None

    def _get_default_model_path(self) -> Path:
        models_dir = (
            Path(settings.BASE_DIR) / 'hairmixer_app' / 'ml' / 'models'
        )
        return models_dir / 'resnet50_80epoch.pth'

    def load_model(self, model_path: str | os.PathLike | None = None) -> bool:
        try:
            if model_path is None:
                model_path = self._get_default_model_path()
            model_path_str = str(model_path)
            self.model_path = model_path_str
            self.model_path = model_path_str

            logger.info(
                "Model load start",
                extra={
                    "event": "model_load_start",
                    "model": "ResNet50",
                    "path": model_path_str,
                    "device": str(self.device),
                    "torch": torch.__version__,
                    "cuda": bool(torch.cuda.is_available()),
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

            # Create model instance
            self.model = ResNet50FaceShapeClassifier(
                num_classes=self.num_classes, pretrained=False
            )

            # Load checkpoint
            checkpoint = torch.load(model_path_str, map_location=self.device)
            # Extract state dict
            if isinstance(checkpoint, dict):
                state = (
                    checkpoint.get('model_state_dict')
                    or checkpoint.get('state_dict')
                    or checkpoint
                )
            else:
                state = checkpoint

            # Remove optional DataParallel 'module.' prefix
            if any(k.startswith('module.') for k in state.keys()):
                state = {
                    (k[len('module.'):] if k.startswith('module.') else k): v
                    for k, v in state.items()
                }

            # Load with strict=False to allow minor head differences
            missing, unexpected = self.model.load_state_dict(
                state, strict=False
            )
            if missing or unexpected:
                logger.debug(
                    "State dict mismatches",
                    extra={
                        "missing": missing[:20],
                        "unexpected": unexpected[:20],
                    },
                )

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
                    "model": "ResNet50",
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
                    "model": "ResNet50",
                    "error": str(e),
                },
            )
            import traceback
            logger.debug(
                "Model load traceback",
                extra={"trace": traceback.format_exc()},
            )
            return False

    def predict(self, input_tensor: torch.Tensor):
        if self.model is None:
            logger.error(
                "Predict called without model",
                extra={"event": "model_predict_error", "reason": "not_loaded"},
            )
            return None

        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence_tensor = torch.max(probabilities)

                pred_idx = int(predicted_class.cpu().numpy()[0])
                confidence = float(confidence_tensor.cpu().numpy())

                t1 = time.perf_counter()
                logger.info(
                    "Model prediction",
                    extra={
                        "event": "model_predict",
                        "model": "ResNet50",
                        "device": str(self.device),
                        "predicted_class": pred_idx,
                        "confidence": round(confidence, 6),
                        "latency_ms": int((t1 - t0) * 1000),
                    },
                )

                return {
                    'predicted_class': pred_idx,
                    'probabilities': probabilities.cpu().numpy()[0],
                    'confidence': confidence,
                }
        except Exception as e:
            logger.error(
                "Prediction exception",
                extra={"event": "model_predict_error", "error": str(e)},
            )
            return None

    def is_loaded(self) -> bool:
        return self.model is not None


# Global loader instance
resnet_loader = ResNetModelLoader()
