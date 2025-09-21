from PIL import Image, ImageFilter
import logging
from pathlib import Path
import requests
import os
from typing import Optional

from django.conf import settings

try:
    # Optional dependency, only required for advanced overlay with AI
    from gemini_webapi import GeminiClient
    from gemini_webapi.constants import Model
    _GEMINI_AVAILABLE = True
except Exception:
    _GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedOverlayProcessor:
    """Advanced image overlay processing with facial landmark alignment"""
    
    def __init__(self):
        self.temp_dir = Path("temp_downloads")
        self.temp_dir.mkdir(exist_ok=True)
        self.ai_enabled = getattr(settings, 'OVERLAY_AI_ENABLED', False)
        self.gemini_sid = getattr(settings, 'GEMINI_SECURE_1PSID', '')
        self.gemini_sidts = getattr(settings, 'GEMINI_SECURE_1PSIDTS', '')
        self.gemini_model = getattr(settings, 'GEMINI_MODEL', 'G_2_5_FLASH')
        self.gemini_timeout = int(getattr(settings, 'GEMINI_TIMEOUT', 120))
        # Basic overlay parameters
        self.basic_width_ratio = float(getattr(settings, 'OVERLAY_BASIC_WIDTH_RATIO', 0.75))
        self.y_offset_ratio = float(getattr(settings, 'OVERLAY_Y_OFFSET_RATIO', 0.03))
        self.blur_radius = float(getattr(settings, 'OVERLAY_BLUR_RADIUS', 0.8))
    
    def create_basic_overlay(self, user_img_path, style_img_path, output_path):
        """Create a basic overlay (improved version of simple_overlay)"""
        try:
            # Load images
            base = Image.open(user_img_path).convert("RGBA")
            hairstyle = Image.open(style_img_path).convert("RGBA")
            
            # Get dimensions
            base_width, base_height = base.size
            
            # Resize hairstyle proportionally
            new_width = int(base_width * self.basic_width_ratio)
            ratio = new_width / hairstyle.size[0]
            new_height = int(hairstyle.size[1] * ratio)
            
            hairstyle_resized = hairstyle.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Calculate position (centered horizontally, positioned at top)
            x_offset = int((base_width - new_width) / 2)
            y_offset = max(0, int(base_height * self.y_offset_ratio))
            
            # Create result image
            result = base.copy()
            
            # Apply blend modes for more natural appearance
            hairstyle_blurred = hairstyle_resized.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
            
            hairstyle_with_opacity = Image.new("RGBA", hairstyle_blurred.size)
            hairstyle_with_opacity.paste(hairstyle_blurred, (0, 0))
            
            # Apply alpha composite
            result.alpha_composite(hairstyle_with_opacity, (x_offset, y_offset))
            
            # Save result
            result.save(output_path, "PNG")
            logger.info(f"Basic overlay created: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating basic overlay: {str(e)}")
            raise
    
    def create_advanced_overlay(self, user_img_path, style_img_path, output_path, style_name: Optional[str] = None):
        """Create advanced overlay via Gemini Web API if configured; otherwise fallback to basic.

        Contract:
        - Inputs: user_img_path (str/Path), style_img_path (unused here), output_path (Path)
        - Output: path string to saved PNG
        - Behavior: if AI disabled/missing creds/lib, uses basic overlay
        """
        try:
            # Guard: use AI only if enabled and library + creds are present
            use_ai = (
                self.ai_enabled and _GEMINI_AVAILABLE and self.gemini_sid and self.gemini_sidts
            )
            if not use_ai:
                reason = []
                if not self.ai_enabled:
                    reason.append('AI disabled')
                if not _GEMINI_AVAILABLE:
                    reason.append('gemini_webapi not installed')
                if not self.gemini_sid or not self.gemini_sidts:
                    reason.append('missing credentials')
                logger.warning(f"Advanced overlay falling back to basic overlay ({', '.join(reason)})")
                return self.create_basic_overlay(user_img_path, style_img_path, output_path)

            # Build a prompt similar to reference repo: "Edit the person's hair to {hairstyle}"
            # Use the provided style_name when available, otherwise keep generic wording.
            desired_style = style_name or "the selected hairstyle"
            prompt = f"Edit the person's hair to {desired_style}. Maintain natural look, lighting and proportions."

            # Initialize Gemini client
            client = GeminiClient(self.gemini_sid, self.gemini_sidts)
            # Longer session to reuse connection; mirrors reference repo defaults
            import asyncio
            async def _run():
                await client.init(timeout=self.gemini_timeout, auto_close=False, close_delay=60, auto_refresh=True, verbose=False)
                try:
                    # Choose model
                    model = getattr(Model, self.gemini_model, Model.G_2_5_FLASH)
                except Exception:
                    model = Model.G_2_5_FLASH

                # Generate edited images
                resp = await client.generate_content(prompt, files=[str(user_img_path)], model=model)
                images = getattr(resp, 'images', None)
                if not images:
                    raise RuntimeError('Gemini returned no edited images')

                # Save first image
                out_dir = Path(output_path).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                # gemini_webapi image objects support async save(path, filename)
                await images[0].save(path=str(out_dir), filename=Path(output_path).name)

            # Run the async flow
            # If already in an event loop (e.g., ASGI), use create_task style fallback
            try:
                asyncio.run(_run())
            except RuntimeError:
                # Likely running inside existing event loop; use nested loop policy
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                loop.run_until_complete(_run())

            logger.info(f"Advanced overlay created via Gemini: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error creating advanced overlay: {str(e)}")
            # Fallback to basic overlay on error, but only if we have a style
            # image available. If not, surface a clear input error.
            if not style_img_path:
                raise ValueError(
                    "Hairstyle image not available for overlay"
                )
            return self.create_basic_overlay(
                user_img_path, style_img_path, output_path
            )
    
    def download_style_image(self, image_url, style_id):
        """Download hairstyle image from URL"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Create temporary file
            temp_file = self.temp_dir / f"style_{style_id}.jpg"
            
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded style image: {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.error(
                f"Error downloading style image from {image_url}: {str(e)}"
            )
            raise
        