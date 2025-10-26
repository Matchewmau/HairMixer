"""
Gemini API Service for generating personalized hairstyle descriptions and recommendations.

This service uses Google's Gemini API to generate:
- Personalized hairstyle descriptions
- How the style fits user preferences
- Product recommendations
- Maintenance instructions
- Styling tips
"""
import os
import logging
from typing import Dict, Optional, List
import google.generativeai as genai
from django.conf import settings

logger = logging.getLogger(__name__)


class GeminiHairstyleService:
    """Service for generating AI-powered hairstyle information using Gemini API"""
    
    def __init__(self):
        """Initialize Gemini API with credentials from settings"""
        api_key_env = os.environ.get('GEMINI_API_KEY')
        self.api_key = getattr(settings, 'GEMINI_API_KEY', api_key_env)
        default_model = 'gemini-1.5-flash'
        self.model_name = getattr(settings, 'GEMINI_MODEL_NAME', default_model)
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                logger.info(f"Gemini API initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {str(e)}")
                self.enabled = False
        else:
            logger.warning("Gemini API key not configured. AI features will be disabled.")
    
    def generate_hairstyle_details(
        self,
        hairstyle_name: str,
        hairstyle_description: str,
        user_preferences: Dict,
        face_shape: str,
        face_shape_confidence: float = 0.0,
        hairstyle_tags: Optional[List[str]] = None,
        hairstyle_occasions: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate detailed hairstyle information personalized to user preferences.
        
        Args:
            hairstyle_name: Name of the hairstyle
            hairstyle_description: Basic description of the hairstyle
            user_preferences: Dictionary containing user preferences
            face_shape: Detected face shape
            face_shape_confidence: Confidence score for face shape detection
            hairstyle_tags: Tags associated with the hairstyle
            hairstyle_occasions: Occasions suitable for the hairstyle
        
        Returns:
            Dictionary containing:
                - personalized_description: AI-generated personalized description
                - preference_match: How it matches user preferences
                - products: List of recommended products
                - maintenance_guide: Detailed maintenance instructions
                - styling_tips: Professional styling tips
                - success: Boolean indicating if generation was successful
        """
        if not self.enabled:
            return self._generate_fallback_details(
                hairstyle_name,
                hairstyle_description,
                user_preferences
            )
        
        try:
            # Build comprehensive prompt
            prompt = self._build_prompt(
                hairstyle_name,
                hairstyle_description,
                user_preferences,
                face_shape,
                face_shape_confidence,
                hairstyle_tags,
                hairstyle_occasions
            )
            
            # Generate content
            response = self.model.generate_content(prompt)
            
            # Parse response
            result = self._parse_response(response.text)
            result['success'] = True
            
            logger.info(f"Successfully generated details for hairstyle: {hairstyle_name}")
            return result
            
        except Exception as e:
            logger.error(
                f"Error generating hairstyle details: {str(e)}",
                exc_info=True
            )
            return self._generate_fallback_details(
                hairstyle_name,
                hairstyle_description,
                user_preferences
            )
    
    def _build_prompt(
        self,
        hairstyle_name: str,
        hairstyle_description: str,
        user_preferences: Dict,
        face_shape: str,
        face_shape_confidence: float,
        hairstyle_tags: Optional[List[str]],
        hairstyle_occasions: Optional[List[str]]
    ) -> str:
        """Build a comprehensive prompt for Gemini API"""
        
        # Extract user preferences
        hair_type = user_preferences.get('hair_type', 'wavy')
        hair_length = user_preferences.get('hair_length', 'medium')
        maintenance = user_preferences.get('maintenance', 'medium')
        lifestyle = user_preferences.get('lifestyle', 'casual')
        gender = user_preferences.get('gender', 'female')
        occasions = user_preferences.get('occasions', [])
        hair_thickness = user_preferences.get('hair_thickness', 'medium')
        hair_texture = user_preferences.get('hair_texture_detail', '')
        wants_bangs = user_preferences.get('wants_bangs', False)
        
        prompt = f"""You are a professional hairstylist and beauty consultant. Generate detailed, personalized information about a hairstyle recommendation.

**Hairstyle Information:**
- Name: {hairstyle_name}
- Base Description: {hairstyle_description}
- Tags: {', '.join(hairstyle_tags) if hairstyle_tags else 'N/A'}
- Suitable Occasions: {', '.join(hairstyle_occasions) if hairstyle_occasions else 'N/A'}

**User Profile:**
- Face Shape: {face_shape} (detected with {int(face_shape_confidence * 100)}% confidence)
- Hair Type: {hair_type}
- Current/Desired Hair Length: {hair_length}
- Hair Thickness: {hair_thickness}
- Hair Texture: {hair_texture if hair_texture else 'Not specified'}
- Prefers Bangs: {'Yes' if wants_bangs else 'No'}
- Maintenance Preference: {maintenance}
- Lifestyle: {lifestyle}
- Gender: {gender}
- Preferred Occasions: {', '.join(occasions) if occasions else 'Not specified'}

Please provide the following information in a clear, structured format:

**1. PERSONALIZED_DESCRIPTION:**
Write a 2-3 sentence personalized description of how this hairstyle will look on this specific person, considering their face shape, hair type, and preferences. Make it engaging and positive.

**2. PREFERENCE_MATCH:**
In 3-4 bullet points, explain specifically how this hairstyle matches the user's preferences and lifestyle. Be specific about face shape compatibility, maintenance level, and lifestyle fit.

**3. RECOMMENDED_PRODUCTS:**
List 5-7 specific hair products needed to achieve and maintain this style. Include:
- Product name and type (e.g., "Volumizing Mousse", "Texturizing Spray")
- Brief purpose (one sentence)
Format as: "• Product Name - Purpose"

**4. MAINTENANCE_GUIDE:**
Provide detailed maintenance instructions in 4-5 steps. Include:
- Daily routine
- Weekly care
- Touch-up frequency
- Expected time commitment
Format as numbered steps.

**5. STYLING_TIPS:**
Provide 4-5 professional styling tips specific to this hairstyle and hair type. Include:
- Techniques for best results
- Common mistakes to avoid
- Pro tricks for longevity
Format as bullet points starting with "• "

Format your response EXACTLY as shown above with clear section headers (use ** for headers). Be specific, practical, and professional. Keep the tone friendly and encouraging."""

        return prompt
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse Gemini API response into structured data"""
        
        result = {
            'personalized_description': '',
            'preference_match': [],
            'products': [],
            'maintenance_guide': [],
            'styling_tips': []
        }
        
        try:
            # Split response into sections
            sections = response_text.split('**')
            current_section = None
            current_content = []
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                # Check for section headers
                if 'PERSONALIZED_DESCRIPTION' in section.upper():
                    if current_section and current_content:
                        self._process_section(result, current_section, current_content)
                    current_section = 'personalized_description'
                    current_content = []
                elif 'PREFERENCE_MATCH' in section.upper():
                    if current_section and current_content:
                        self._process_section(result, current_section, current_content)
                    current_section = 'preference_match'
                    current_content = []
                elif 'RECOMMENDED_PRODUCTS' in section.upper() or 'PRODUCT' in section.upper():
                    if current_section and current_content:
                        self._process_section(result, current_section, current_content)
                    current_section = 'products'
                    current_content = []
                elif 'MAINTENANCE_GUIDE' in section.upper() or 'MAINTENANCE' in section.upper():
                    if current_section and current_content:
                        self._process_section(result, current_section, current_content)
                    current_section = 'maintenance_guide'
                    current_content = []
                elif 'STYLING_TIPS' in section.upper() or 'STYLING TIP' in section.upper():
                    if current_section and current_content:
                        self._process_section(result, current_section, current_content)
                    current_section = 'styling_tips'
                    current_content = []
                elif current_section:
                    current_content.append(section)
            
            # Process last section
            if current_section and current_content:
                self._process_section(result, current_section, current_content)
            
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
        
        return result
    
    def _process_section(self, result: Dict, section: str, content: List[str]) -> None:
        """Process a section of the response"""
        
        full_content = ' '.join(content).strip()
        
        if section == 'personalized_description':
            result['personalized_description'] = full_content
        
        elif section in ['preference_match', 'products', 'styling_tips']:
            # Parse bullet points
            items = []
            for line in full_content.split('\n'):
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    items.append(line.lstrip('•-* ').strip())
                elif line and not line.startswith('**'):
                    # Sometimes items don't have bullets
                    items.append(line)
            result[section] = items
        
        elif section == 'maintenance_guide':
            # Parse numbered steps
            items = []
            for line in full_content.split('\n'):
                line = line.strip()
                # Remove numbers at start (1., 2., etc.)
                if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                    clean_line = line.lstrip('0123456789.-•* ').strip()
                    if clean_line:
                        items.append(clean_line)
                elif line and not line.startswith('**'):
                    items.append(line)
            result[section] = items
    
    def _generate_fallback_details(
        self,
        hairstyle_name: str,
        hairstyle_description: str,
        user_preferences: Dict
    ) -> Dict:
        """Generate fallback details when Gemini API is not available"""
        
        hair_type = user_preferences.get('hair_type', 'wavy')
        maintenance = user_preferences.get('maintenance', 'medium')
        
        return {
            'success': False,
            'personalized_description': f"{hairstyle_name} is a versatile style that works well with {hair_type} hair. {hairstyle_description}",
            'preference_match': [
                f"Matches your {maintenance} maintenance preference",
                "Compatible with your face shape",
                "Suits your lifestyle needs",
                "Versatile for various occasions"
            ],
            'products': [
                "Shampoo and Conditioner - For daily cleansing and hydration",
                "Styling Product - To achieve the desired look",
                "Heat Protectant - Protect hair from styling tools",
                "Hair Oil or Serum - For shine and smoothness",
                "Finishing Spray - To hold the style in place"
            ],
            'maintenance_guide': [
                "Wash hair 2-3 times per week with appropriate shampoo",
                "Apply styling products to damp hair",
                "Style as desired using appropriate tools",
                "Touch up as needed throughout the week",
                "Schedule regular trims every 6-8 weeks"
            ],
            'styling_tips': [
                "Work with your natural hair texture for best results",
                "Use heat protection when using styling tools",
                "Don't over-wash to maintain natural oils",
                "Adjust products based on weather and humidity",
                "Practice makes perfect - give yourself time to master the style"
            ]
        }


# Singleton instance
_gemini_service = None

def get_gemini_service() -> GeminiHairstyleService:
    """Get or create the Gemini service singleton"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiHairstyleService()
    return _gemini_service
