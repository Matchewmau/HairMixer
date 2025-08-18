from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2
import numpy as np
import logging
from pathlib import Path
import requests
from django.core.files.temp import NamedTemporaryFile

logger = logging.getLogger(__name__)

class AdvancedOverlayProcessor:
    """Advanced image overlay processing with facial landmark alignment"""
    
    def __init__(self):
        self.temp_dir = Path("temp_downloads")
        self.temp_dir.mkdir(exist_ok=True)
    
    def create_basic_overlay(self, user_img_path, style_img_path, output_path):
        """Create a basic overlay (improved version of simple_overlay)"""
        try:
            # Load images
            base = Image.open(user_img_path).convert("RGBA")
            hairstyle = Image.open(style_img_path).convert("RGBA")
            
            # Get dimensions
            base_width, base_height = base.size
            
            # Resize hairstyle proportionally
            new_width = int(base_width * 0.75)
            ratio = new_width / hairstyle.size[0]
            new_height = int(hairstyle.size[1] * ratio)
            
            hairstyle_resized = hairstyle.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Calculate position (centered horizontally, positioned at top)
            x_offset = int((base_width - new_width) / 2)
            y_offset = max(0, int(base_height * 0.03))
            
            # Create result image
            result = base.copy()
            
            # Apply blend modes for more natural appearance
            hairstyle_blurred = hairstyle_resized.filter(ImageFilter.GaussianBlur(radius=0.8))
            
            # Adjust opacity based on hairstyle characteristics
            opacity = 0.85
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
    
    def create_advanced_overlay(self, user_img_path, style_img_path, output_path):
        """Create advanced overlay - fallback to basic without face_recognition"""
        try:
            # For now, fallback to basic overlay since face_recognition is not available
            logger.warning("Advanced overlay falling back to basic overlay (face_recognition not available)")
            return self.create_basic_overlay(user_img_path, style_img_path, output_path)
            
        except Exception as e:
            logger.error(f"Error creating advanced overlay: {str(e)}")
            return self.create_basic_overlay(user_img_path, style_img_path, output_path)
    
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
            logger.error(f"Error downloading style image from {image_url}: {str(e)}")
            raise