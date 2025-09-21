import logging
from pathlib import Path
from typing import Union
from django.conf import settings
from ..overlay import AdvancedOverlayProcessor, _GEMINI_AVAILABLE
from ..models import UploadedImage, Hairstyle
from urllib.parse import urlparse
import ipaddress
import socket

logger = logging.getLogger(__name__)


class OverlayService:
    def __init__(self):
        self.processor = AdvancedOverlayProcessor()

    def generate(
        self,
        uploaded: UploadedImage,
        style: Hairstyle,
        overlay_type: str = "basic",
    ) -> str:
        user_img_path = Path(settings.MEDIA_ROOT) / uploaded.image.name

    # Only require a hairstyle visual when we actually need to
    # composite locally
        style_img_path: Union[str, Path, None] = None
        if style.image:
            style_img_path = Path(settings.MEDIA_ROOT) / style.image.name
        elif style.image_url:
            url_ok = True
            try:
                if getattr(settings, 'ENABLE_SSRF_PROTECTION', False):
                    parsed = urlparse(style.image_url)
                    if (
                        parsed.scheme not in ('http', 'https')
                        or not parsed.hostname
                    ):
                        url_ok = False
                    else:
                        infos = socket.getaddrinfo(parsed.hostname, None)
                        for info in infos:
                            ip = ipaddress.ip_address(info[4][0])
                            if (
                                ip.is_private
                                or ip.is_loopback
                                or ip.is_reserved
                                or ip.is_link_local
                                or ip.is_multicast
                            ):
                                url_ok = False
                                break
            except Exception:
                url_ok = False

            if not url_ok:
                raise ValueError("Invalid or non-public hairstyle image URL")

            style_img_path = self.processor.download_style_image(
                style.image_url, style.id
            )

        out_rel = f"overlays/{uploaded.id}_{style.id}_{overlay_type}.png"
        out_abs = Path(settings.MEDIA_ROOT) / out_rel
        out_abs.parent.mkdir(parents=True, exist_ok=True)

        if overlay_type == "advanced":
            # If AI is viable, we can proceed without a style image.
            ai_viable = (
                self.processor.ai_enabled
                and _GEMINI_AVAILABLE
                and getattr(self.processor, 'gemini_sid', '')
                and getattr(self.processor, 'gemini_sidts', '')
            )
            # When AI isn't viable, we need a style image for basic fallback.
            if not ai_viable and style_img_path is None:
                raise ValueError(
                    "Hairstyle image not available for overlay"
                )
            style_name = getattr(style, 'name', None)
            self.processor.create_advanced_overlay(
                user_img_path, style_img_path, out_abs, style_name=style_name
            )
        else:
            if style_img_path is None:
                raise ValueError("Hairstyle image not available for overlay")
            self.processor.create_basic_overlay(
                user_img_path, style_img_path, out_abs
            )

        return f"{settings.MEDIA_URL}{out_rel}"
