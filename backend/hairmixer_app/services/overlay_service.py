import logging
from pathlib import Path
from typing import Union
from django.conf import settings
from ..overlay import AdvancedOverlayProcessor
from ..models import UploadedImage, Hairstyle

logger = logging.getLogger(__name__)


class OverlayService:
    def __init__(self):
        self.processor = AdvancedOverlayProcessor()

    def generate(self, uploaded: UploadedImage, style: Hairstyle, overlay_type: str = "basic") -> str:
        user_img_path = Path(settings.MEDIA_ROOT) / uploaded.image.name
        if style.image:
            style_img_path: Union[str, Path] = Path(settings.MEDIA_ROOT) / style.image.name
        elif style.image_url:
            style_img_path = self.processor.download_style_image(style.image_url, style.id)
        else:
            raise ValueError("Hairstyle image not available")

        out_rel = f"overlays/{uploaded.id}_{style.id}_{overlay_type}.png"
        out_abs = Path(settings.MEDIA_ROOT) / out_rel
        out_abs.parent.mkdir(parents=True, exist_ok=True)

        if overlay_type == "advanced":
            style_name = getattr(style, 'name', None)
            self.processor.create_advanced_overlay(user_img_path, style_img_path, out_abs, style_name=style_name)
        else:
            self.processor.create_basic_overlay(user_img_path, style_img_path, out_abs)

        return f"{settings.MEDIA_URL}{out_rel}"
