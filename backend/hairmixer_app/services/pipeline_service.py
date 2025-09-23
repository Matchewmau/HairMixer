import logging
from typing import Dict, Any
from django.utils import timezone

from ..models import UploadedImage, UserPreference, Hairstyle
from .recommendation_service import RecommendationService
from .overlay_service import OverlayService

logger = logging.getLogger(__name__)


class RecommendationOverlayPipeline:
    """
    High-level pipeline that recommends a hairstyle then generates overlay.
    """

    def __init__(self):
        self.recommendation_service = RecommendationService()
        self.overlay_service = OverlayService()

    def run(
        self,
        uploaded: UploadedImage,
        prefs: UserPreference,
        overlay_type: str = 'advanced',
        user=None,
    ) -> Dict[str, Any]:
        start = timezone.now()
        # Step 1: get recommendations
        rec = self.recommendation_service.generate(uploaded, prefs, user=user)
        styles = rec.get('recommended_styles') or []
        if not styles:
            return {
                'error': 'No styles available for overlay',
                'recommendation': rec,
            }
        # choose top style
        top = styles[0]
        style_id = top.get('id')
        if not style_id:
            # cannot overlay without a concrete hairstyle record
            return {
                'error': 'Top style is not a concrete DB style',
                'recommendation': rec,
            }
        # fetch style
        try:
            style = Hairstyle.objects.get(id=style_id)
        except Hairstyle.DoesNotExist:
            return {
                'error': 'Selected hairstyle not found',
                'recommendation': rec,
            }
        # Step 2: generate overlay
        overlay_url = self.overlay_service.generate(
            uploaded, style, overlay_type
        )
        elapsed = (timezone.now() - start).total_seconds()
        return {
            'overlay_url': overlay_url,
            'overlay_type': overlay_type,
            'selected_style': top,
            'recommendation': rec,
            'processing_time': f"{elapsed:.2f}s",
        }
