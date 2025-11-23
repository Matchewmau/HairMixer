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
            
            # Ensure all required sections exist with fallback content
            result = self._ensure_complete_sections(
                result,
                hairstyle_name,
                hairstyle_description,
                user_preferences
            )
            
            result['success'] = True
            
            logger.info(
                f"Successfully generated details for hairstyle: "
                f"{hairstyle_name}"
            )
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
        
        # Extract user preferences with defaults
        hair_type = user_preferences.get('hair_type', 'wavy')
        hair_length = user_preferences.get('hair_length', 'medium')
        maintenance = user_preferences.get('maintenance', 'medium')
        lifestyle = user_preferences.get('lifestyle', 'casual')
        gender = user_preferences.get('gender', 'female')
        occasions = user_preferences.get('occasions', [])
        hair_thickness = user_preferences.get('hair_thickness', 'medium')
        hair_texture = user_preferences.get('hair_texture_detail', '')
        wants_bangs = user_preferences.get('wants_bangs', False)
        volume = user_preferences.get('volume', 'medium')
        styling_preference = user_preferences.get('styling_preference', 'natural')
        hair_color = user_preferences.get('hair_color', 'brown')
        hair_condition = user_preferences.get('hair_condition', [])
        
        # Log the actual values being used for debugging
        logger.info(
            f"Building prompt with face_shape='{face_shape}' "
            f"(confidence={face_shape_confidence:.2f}), "
            f"hair_type='{hair_type}', gender='{gender}'"
        )
        
        # Format lists and strings for display
        condition_str = (
            ', '.join(hair_condition) if hair_condition else 'Healthy'
        )
        tags_str = ', '.join(hairstyle_tags) if hairstyle_tags else 'N/A'
        occasions_hairstyle_str = (
            ', '.join(hairstyle_occasions) if hairstyle_occasions else 'N/A'
        )
        occasions_user_str = (
            ', '.join(occasions) if occasions else 'Not specified'
        )
        texture_str = hair_texture if hair_texture else 'Not specified'
        bangs_str = 'Yes' if wants_bangs else 'No'
        confidence_pct = int(face_shape_confidence * 100)
        
        prompt = f"""You are a professional hairstylist and beauty consultant. Generate detailed, personalized information about a hairstyle recommendation.

**Hairstyle Information:**
- Name: {hairstyle_name}
- Base Description: {hairstyle_description}
- Tags: {tags_str}
- Suitable Occasions: {occasions_hairstyle_str}

**User Profile:**
- Face Shape: {face_shape} (AI-detected with {confidence_pct}% confidence using ResNet50 model)
- Gender: {gender}
- Hair Type: {hair_type}
- Current/Desired Hair Length: {hair_length}
- Hair Color: {hair_color}
- Hair Thickness: {hair_thickness}
- Hair Volume: {volume}
- Hair Texture: {texture_str}
- Hair Condition: {condition_str}
- Prefers Bangs: {bangs_str}
- Styling Preference: {styling_preference}
- Maintenance Preference: {maintenance}
- Lifestyle: {lifestyle}
- Preferred Occasions: {occasions_user_str}

Please provide the following information in a clear, structured format:

**1. PERSONALIZED_DESCRIPTION:**
Write a 2-3 sentence personalized description of how this hairstyle will look on this specific person. You MUST explicitly reference their {face_shape} face shape and {hair_type} hair type in your explanation. Explain WHY it works for them specifically.

**2. PREFERENCE_MATCH:**
In 1-4 bullet points, explain specifically how this hairstyle matches the user's preferences.
- Mention if it fits their {maintenance} maintenance preference.
- Mention how it suits their lifestyle.

**3. RECOMMENDED_PRODUCTS:**
List 2-4 specific hair products.
IMPORTANT: For each product, you MUST include:
- Specific Product Type (e.g., "Argan Oil Serum", not just "Oil")
- Key Ingredients to look for (e.g., "Look for products with keratin or biotin")
- Specific Usage (e.g., "Apply dime-sized amount to damp ends")
Format EXACTLY as: "Product Type - Key Ingredients: [ingredients] - [usage]"

Example:
• Volumizing Mousse - Key Ingredients: Rice protein, polymers - Apply to roots of damp hair before blow-drying
• Heat Protectant Spray - Key Ingredients: Silicones or Argan oil - Mist all over dry hair before ironing

**4. MAINTENANCE_GUIDE:**
Provide 2-3 complete, actionable maintenance steps.
Include:
- Daily Styling: Exact time estimate and specific techniques.
- Wash Schedule: Exact frequency (e.g., "Every 2-3 days").
- Salon Visits: Exact frequency (e.g., "Every 6-8 weeks").
Format as: "1. Step Title: Detailed instructions..."

Example:
1. Daily Styling (15 min): Dampen hair, apply mousse, and scrunch. Diffuse on low heat.
2. Wash Schedule: Wash every 3 days using sulfate-free shampoo to prevent drying.
3. Salon Maintenance: Visit stylist every 8 weeks for a trim to keep layers fresh.

**5. STYLING_TIPS:**
Provide 2-4 professional styling tips specific to this hairstyle.
- Include specific techniques (e.g., "Use the cool shot button").
- Include specific tool settings (e.g., "Medium heat, low airflow").
Format as bullet points starting with "• " or "→ "

CRITICAL: Format your response EXACTLY as shown above with clear section 
headers marked with **. You MUST include ALL 5 SECTIONS:
1. PERSONALIZED_DESCRIPTION
2. PREFERENCE_MATCH
3. RECOMMENDED_PRODUCTS
4. MAINTENANCE_GUIDE
5. STYLING_TIPS

Do NOT skip any section. Be highly specific and practical. Avoid generic advice.
Keep the tone professional yet encouraging."""

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

            # Log which sections were found/missing
            expected_sections = [
                'personalized_description', 'preference_match',
                'products', 'maintenance_guide', 'styling_tips'
            ]
            missing_sections = [
                s for s in expected_sections
                if s not in result or not result[s]
            ]
            if missing_sections:
                logger.warning(
                    f"Missing or empty sections in Gemini response: "
                    f"{missing_sections}"
                )

        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")

        return result

    def _ensure_complete_sections(
        self,
        result: Dict,
        hairstyle_name: str,
        hairstyle_description: str,
        user_preferences: Dict
    ) -> Dict:
        """
        Ensure all required sections have content,
        filling missing sections with fallback content
        """

        hair_type = user_preferences.get('hair_type', 'wavy')
        maintenance = user_preferences.get('maintenance', 'medium')

        # Check and fill personalized_description
        if not result.get('personalized_description'):
            result['personalized_description'] = (
                f"{hairstyle_name} is a versatile style that works "
                f"beautifully with {hair_type} hair. {hairstyle_description}"
            )
            logger.info("Added fallback personalized_description")

        # Check and fill preference_match
        if not result.get('preference_match'):
            result['preference_match'] = [
                f"Matches your {maintenance} maintenance preference",
                "Compatible with your detected face shape",
                "Suits your lifestyle and daily routine",
                "Versatile for various occasions"
            ]
            logger.info("Added fallback preference_match")

        # Check and fill products
        if not result.get('products'):
            result['products'] = [
                "Shampoo and Conditioner - For daily cleansing",
                "Styling Cream or Mousse - To achieve the look",
                "Heat Protectant - Protects from styling tools",
                "Hair Serum or Oil - For shine and smoothness",
                "Finishing Spray - To hold the style"
            ]
            logger.info("Added fallback products")

        # Check and fill maintenance_guide
        if not result.get('maintenance_guide'):
            result['maintenance_guide'] = [
                "Daily Styling (10-15 min): Apply product to damp hair "
                "and style as desired",
                "Weekly Care: Deep condition once per week to maintain "
                "hair health",
                "Regular Trims: Visit salon every 6-8 weeks to maintain "
                "shape",
                "Touch-Ups: Adjust styling as needed throughout the day"
            ]
            logger.info("Added fallback maintenance_guide")

        # Check and fill styling_tips
        if not result.get('styling_tips'):
            result['styling_tips'] = [
                "Work with your natural hair texture",
                "Use heat protection when styling with hot tools",
                "Don't over-wash to maintain natural oils",
                "Adjust products based on weather and humidity",
                "Practice different techniques to find what works best"
            ]
            logger.info("Added fallback styling_tips")

        return result
    
    def _process_section(
        self, result: Dict, section: str, content: List[str]
    ) -> None:
        """Process a section of the response"""
        
        full_content = ' '.join(content).strip()
        
        if section == 'personalized_description':
            result['personalized_description'] = full_content
        
        elif section in ['preference_match', 'products', 'styling_tips']:
            # Parse bullet points
            items = []
            for line in full_content.split('\n'):
                line = line.strip()
                
                # Skip empty lines and section headers
                if not line or line.startswith('**'):
                    continue
                
                # Skip common formatting artifacts
                if any(skip in line.lower() for skip in [
                    'example:', 'format as:', 'include:', 
                    'important:', 'note:'
                ]):
                    continue
                
                # Extract bullet point content
                if line.startswith(('•', '-', '*', '→')):
                    clean_line = line.lstrip('•-*→ ').strip()
                    if clean_line and len(clean_line) > 10:
                        items.append(clean_line)
                elif line and len(line) > 10:
                    # Sometimes items don't have bullets
                    items.append(line)
            
            # Validate specific sections
            if section == 'products':
                # Filter out non-product items
                items = [
                    item for item in items 
                    if not any(skip in item.lower() for skip in [
                        'touch-up frequency', 'expected time',
                        'weekly care', 'daily routine'
                    ])
                ]
            
            result[section] = items
        
        elif section == 'maintenance_guide':
            # Parse numbered steps
            items = []
            for line in full_content.split('\n'):
                line = line.strip()
                
                # Skip empty lines and section headers
                if not line or line.startswith('**'):
                    continue
                
                # Skip formatting instructions
                if any(skip in line.lower() for skip in [
                    'example:', 'format as:', 'include:'
                ]):
                    continue
                
                # Extract numbered step content
                if line and (line[0].isdigit() or line.startswith(('•', '-'))):
                    # Remove numbering and bullets
                    clean_line = line.lstrip('0123456789.-•* ').strip()
                    if clean_line and len(clean_line) > 20:
                        items.append(clean_line)
                elif line and len(line) > 20:
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
