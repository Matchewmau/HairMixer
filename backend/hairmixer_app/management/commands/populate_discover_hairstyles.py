"""
Populate database with 24 hairstyles from the Discover page.
"""
import os
import sys
import django
from django.core.management.base import BaseCommand
from django.db import transaction

# Setup Django
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)

from hairmixer_app.models import Hairstyle, HairstyleCategory


class Command(BaseCommand):
    help = 'Populate database with 24 hairstyles from Discover page'

    def handle(self, *args, **options):
        self.stdout.write("=" * 80)
        self.stdout.write(self.style.SUCCESS("POPULATING DATABASE WITH 24 DISCOVER PAGE HAIRSTYLES"))
        self.stdout.write("=" * 80)

        # Ensure categories exist
        categories = self.ensure_categories()

        # Define the 24 hairstyles with all their details
        hairstyles_data = [
            # NATURAL CATEGORY (4 hairstyles)
            {
                "name": "Natural Afro",
                "category": "natural",
                "gender": "male",
                "length": "medium",
                "maintenance": "high",
                "description": "A hairstyle that embraces the natural texture and volume of coily or kinky hair, allowing it to grow outwards and upwards into a rounded shape. It's a statement of identity and requires significant moisture and care to prevent breakage.",
                "face_shapes": ["oval", "round", "square"],
                "tags": ["Afro-textured", "Voluminous", "Coily", "Natural Hair Movement"],
                "styling_tips": "Daily moisturizing (10-20 minutes). Regular deep conditioning weekly, trims every 4-6 weeks.",
                "estimated_time": 15,
                "difficulty": "medium"
            },
            {
                "name": "Surfer Hair (Beachy Waves)",
                "category": "natural",
                "gender": "male",
                "length": "medium",
                "maintenance": "low",
                "description": "A low-maintenance, tousled hairstyle that looks wind-swept and sun-kissed. It's defined by natural-looking waves and texture, often enhanced with sea salt spray to mimic the effect of a day at the beach.",
                "face_shapes": ["oval", "square", "heart"],
                "tags": ["Beachy", "Wavy", "Low-maintenance", "Tousled"],
                "styling_tips": "5-10 minutes air dry with spray. Trim every 8-12 weeks.",
                "estimated_time": 7,
                "difficulty": "easy"
            },
            {
                "name": "Wash-and-Go-Curls",
                "category": "natural",
                "gender": "female",
                "length": "medium",
                "maintenance": "medium",
                "description": "A styling method for naturally curly or coily hair that involves cleansing, conditioning, and applying styling products (like gel or cream) to wet hair to define the natural curl pattern without heat or manipulation. The hair is then air-dried or diffused.",
                "face_shapes": ["oval", "round", "heart"],
                "tags": ["Curly Girl Method", "Natural Curls", "Defined", "Heatless"],
                "styling_tips": "20-40 minutes plus drying time. Wash day routine every 3-7 days.",
                "estimated_time": 30,
                "difficulty": "medium"
            },
            {
                "name": "Long Layers",
                "category": "natural",
                "gender": "female",
                "length": "long",
                "maintenance": "low",
                "description": "A simple, classic cut for long hair that adds movement, removes weight, and enhances natural texture (whether straight or wavy). The layers are typically soft and blended, allowing the hair to fall naturally with shape.",
                "face_shapes": ["oval", "square", "round"],
                "tags": ["Low-maintenance", "Versatile", "Flowing", "Blended"],
                "styling_tips": "5-15 minutes air dry or quick blow-dry. Trim every 8-10 weeks.",
                "estimated_time": 10,
                "difficulty": "easy"
            },
            
            # CASUAL CATEGORY (4 hairstyles)
            {
                "name": "Messy Quiff",
                "category": "casual",
                "gender": "male",
                "length": "short",
                "maintenance": "medium",
                "description": "A relaxed version of the classic quiff, this style features volume and height at the front, but with a deliberately tousled and textured finish. The sides are typically shorter, and the top is styled loosely with fingers rather than a comb.",
                "face_shapes": ["oval", "round", "square"],
                "tags": ["Textured", "Voluminous", "Relaxed", "Modern"],
                "styling_tips": "5-10 minutes styling. Trim every 3-5 weeks.",
                "estimated_time": 7,
                "difficulty": "medium"
            },
            {
                "name": "Buzz Cut",
                "category": "casual",
                "gender": "male",
                "length": "short",
                "maintenance": "high",
                "description": "A very short hairstyle where the hair is clipped close to the head using clippers. It's a no-fuss, masculine style that is extremely easy to style (it requires none) but needs frequent trims to maintain the clean look.",
                "face_shapes": ["oval", "square", "oblong"],
                "tags": ["Minimalist", "Low-maintenance", "Military", "Sharp"],
                "styling_tips": "0 minutes styling. Trim every 2-3 weeks.",
                "estimated_time": 0,
                "difficulty": "easy"
            },
            {
                "name": "Messy Bun",
                "category": "casual",
                "gender": "female",
                "length": "medium",
                "maintenance": "low",
                "description": "A popular and quick updo where the hair is gathered into a bun, but with a deliberately loose, undone, and textured finish. Strands are often left out to frame the face, making it a go-to for a relaxed, everyday look.",
                "face_shapes": ["oval", "round", "heart", "square"],
                "tags": ["Updo", "Relaxed", "Quick-style", "Undone"],
                "styling_tips": "2-5 minutes. As needed maintenance.",
                "estimated_time": 3,
                "difficulty": "easy"
            },
            {
                "name": "Shoulder-Length Shag",
                "category": "casual",
                "gender": "female",
                "length": "medium",
                "maintenance": "low",
                "description": "A modern take on the '70s shag, this cut features heavy layers, lots of texture, and often a fringe (like curtain bangs). It's designed to enhance natural waves and create a rock-and-roll, lived-in vibe with minimal effort.",
                "face_shapes": ["oval", "heart", "square"],
                "tags": ["Layered", "Textured", "Retro", "Beachy"],
                "styling_tips": "5-15 minutes scrunch with spray. Trim every 6-8 weeks.",
                "estimated_time": 10,
                "difficulty": "easy"
            },
            
            # CLASSIC CATEGORY (4 hairstyles)
            {
                "name": "Side Part",
                "category": "classic",
                "gender": "male",
                "length": "short",
                "maintenance": "medium",
                "description": "A timeless men's hairstyle defined by a neat part on one side of the head. The hair on top has length and is combed over, while the sides are tapered or faded. It's a clean, polished look suitable for any occasion.",
                "face_shapes": ["oval", "square", "round"],
                "tags": ["Professional", "Timeless", "Vintage", "Groomed"],
                "styling_tips": "5-10 minutes styling. Trim every 3-4 weeks.",
                "estimated_time": 7,
                "difficulty": "medium"
            },
            {
                "name": "Pompadour",
                "category": "classic",
                "gender": "male",
                "length": "medium",
                "maintenance": "high",
                "description": "An iconic hairstyle featuring short sides and a long top that is swept upwards and back from the forehead, creating significant volume (the 'pomp'). It requires blow-drying and pomade to hold its dramatic shape.",
                "face_shapes": ["oval", "square", "round"],
                "tags": ["Vintage", "Rockabilly", "Voluminous", "High-maintenance"],
                "styling_tips": "10-15 minutes styling with blow-dryer. Trim every 3-4 weeks.",
                "estimated_time": 12,
                "difficulty": "hard"
            },
            {
                "name": "Classic Bob",
                "category": "classic",
                "gender": "female",
                "length": "short",
                "maintenance": "medium",
                "description": "A timeless cut where the hair is typically cut straight around the head at about jaw-level, often with a fringe. The 'classic' bob is precise, polished, and can be worn straight and sleek or with a slight bend.",
                "face_shapes": ["oval", "heart", "square"],
                "tags": ["Chic", "Polished", "Geometric", "Timeless"],
                "styling_tips": "10-15 minutes for sleek look. Trim every 4-6 weeks to maintain shape.",
                "estimated_time": 12,
                "difficulty": "medium"
            },
            {
                "name": "French Twist",
                "category": "classic",
                "gender": "female",
                "length": "medium",
                "maintenance": "medium",
                "description": "A sophisticated updo where hair is gathered, twisted vertically, and pinned neatly against the back of the head. It creates a sleek, polished 'roll' that is a go-to style for formal events and professional settings.",
                "face_shapes": ["oval", "round", "heart", "square"],
                "tags": ["Updo", "Formal", "Elegant", "Timeless"],
                "styling_tips": "10-15 minutes styling. Requires practice and pins.",
                "estimated_time": 12,
                "difficulty": "medium"
            },
            
            # ELEGANT CATEGORY (4 hairstyles)
            {
                "name": "Slick Back",
                "category": "elegant",
                "gender": "male",
                "length": "medium",
                "maintenance": "high",
                "description": "A sharp, polished hairstyle where the hair on top is combed straight back from the forehead, lying flat against the head. It typically features shorter sides (an undercut or fade) and requires a high-shine pomade for a sleek, wet look.",
                "face_shapes": ["oval", "square"],
                "tags": ["Formal", "Polished", "High-shine", "Sharp"],
                "styling_tips": "5-10 minutes styling with pomade. Trim every 3-4 weeks.",
                "estimated_time": 7,
                "difficulty": "medium"
            },
            {
                "name": "Taper Fade with Comb Over",
                "category": "elegant",
                "gender": "male",
                "length": "short",
                "maintenance": "medium",
                "description": "A modern and clean hairstyle that combines two classic elements. The taper fade provides a gradual, clean blend on the sides and back, while the longer top is neatly combed to one side, creating a defined part.",
                "face_shapes": ["oval", "round", "heart", "square"],
                "tags": ["Professional", "Sharp", "Modern-classic", "Faded"],
                "styling_tips": "5-10 minutes styling. Trim every 3-4 weeks.",
                "estimated_time": 7,
                "difficulty": "medium"
            },
            {
                "name": "Chignon",
                "category": "elegant",
                "gender": "female",
                "length": "medium",
                "maintenance": "medium",
                "description": "A classic and elegant updo, typically worn at the nape of the neck. The hair is gathered into a low ponytail, then looped, twisted, or tucked into a sleek, graceful knot. It's a popular choice for weddings and formal events.",
                "face_shapes": ["oval", "round", "heart", "square"],
                "tags": ["Formal", "Updo", "Bridal", "Sophisticated"],
                "styling_tips": "10-15 minutes styling. Requires pins and hairspray.",
                "estimated_time": 12,
                "difficulty": "medium"
            },
            {
                "name": "Classic Updo",
                "category": "elegant",
                "gender": "female",
                "length": "medium",
                "maintenance": "high",
                "description": "A formal hairstyle where the hair is swept up and secured away from the face and neck. This can range from intricate twists, braids, and curls to a voluminous, structured bun. It's designed for special occasions and black-tie events.",
                "face_shapes": ["oval", "round", "heart", "square"],
                "tags": ["Formal", "Black-tie", "Bridal", "Ornate"],
                "styling_tips": "30-60+ minutes, often professional. Special occasion style.",
                "estimated_time": 45,
                "difficulty": "professional"
            },
            
            # GLAMOROUS CATEGORY (4 hairstyles)
            {
                "name": "Quiff with High Shine",
                "category": "glamorous",
                "gender": "male",
                "length": "medium",
                "maintenance": "high",
                "description": "This is a statement-making quiff that focuses on both volume and a wet-look, high-shine finish. It's styled using a blow-dryer for maximum height and a strong-hold, glossy pomade to catch the light.",
                "face_shapes": ["oval", "square", "round"],
                "tags": ["High-shine", "Voluminous", "Statement", "Red carpet"],
                "styling_tips": "10-15 minutes with blow-dryer and pomade. Trim every 3-4 weeks.",
                "estimated_time": 12,
                "difficulty": "hard"
            },
            {
                "name": "Long Wavy Hair",
                "category": "glamorous",
                "gender": "male",
                "length": "long",
                "maintenance": "medium",
                "description": "Long, flowing hair on men, often with a natural wave or curl. When styled for a glamorous look, it's healthy, shiny, and intentionally styled (either defined waves or a 'hero' sweep back) rather than just unkempt. Think red-carpet movie star.",
                "face_shapes": ["oval", "square", "heart"],
                "tags": ["Flowing", "Wavy", "Rugged", "Romantic"],
                "styling_tips": "10-20 minutes for definition. Regular conditioning, trims every 10-12 weeks.",
                "estimated_time": 15,
                "difficulty": "medium"
            },
            {
                "name": "Hollywood Waves",
                "category": "glamorous",
                "gender": "female",
                "length": "medium",
                "maintenance": "high",
                "description": "A classic red-carpet hairstyle characterized by soft, uniform, and highly polished waves. The hair is typically deep-parted to one side and cascades over one shoulder, with a high-gloss finish.",
                "face_shapes": ["oval", "round", "heart", "square"],
                "tags": ["Red carpet", "Vintage", "Polished", "Wavy"],
                "styling_tips": "30-60 minutes styling. Special occasion style.",
                "estimated_time": 45,
                "difficulty": "hard"
            },
            {
                "name": "Voluminous Blowout",
                "category": "glamorous",
                "gender": "female",
                "length": "medium",
                "maintenance": "high",
                "description": "A salon-quality blowout designed to create maximum volume, body, and movement. It involves using a round brush and blow-dryer to lift the roots and create soft, bouncy, shiny hair that looks full and healthy.",
                "face_shapes": ["oval", "round", "heart", "square"],
                "tags": ["Bouncy", "Voluminous", "Luxe", "High-shine"],
                "styling_tips": "20-45 minutes with round brush. Requires heat styling.",
                "estimated_time": 30,
                "difficulty": "hard"
            },
            
            # TRENDY CATEGORY (4 hairstyles)
            {
                "name": "Textured Crop (French Crop)",
                "category": "trendy",
                "gender": "male",
                "length": "short",
                "maintenance": "low",
                "description": "A very popular modern cut featuring a short, textured top with a distinct fringe, contrasted by faded or undercut sides. The top is styled forward to create a messy, textured look. It's low-maintenance and stylish.",
                "face_shapes": ["oval", "square"],
                "tags": ["Faded", "Textured", "Fringe", "Contemporary"],
                "styling_tips": "2-5 minutes with matte clay or paste. Trim every 3-4 weeks.",
                "estimated_time": 3,
                "difficulty": "easy"
            },
            {
                "name": "Modern Mullet",
                "category": "trendy",
                "gender": "male",
                "length": "medium",
                "maintenance": "medium",
                "description": "A modern reinterpretation of the '80s classic. This version is more subtle, often featuring a taper fade on the sides, a textured top, and a less-dramatic, more blended length in the back. It's 'business in the front, party in the back' with a fashion-forward twist.",
                "face_shapes": ["oval", "square", "round"],
                "tags": ["Retro", "Edgy", "Faded", "Statement"],
                "styling_tips": "5-10 minutes styling. Trim every 4-6 weeks to maintain shape.",
                "estimated_time": 7,
                "difficulty": "medium"
            },
            {
                "name": "Wolf Cut",
                "category": "trendy",
                "gender": "female",
                "length": "medium",
                "maintenance": "medium",
                "description": "A viral hybrid of a shag and a mullet. It features short, choppy layers on top for volume and longer, thinned-out layers in the back. It's defined by its wild texture and is often paired with curtain bangs.",
                "face_shapes": ["oval", "heart", "square"],
                "tags": ["Viral", "Layered", "Edgy", "Textured"],
                "styling_tips": "10-15 minutes with texturizing spray. Trim every 6-8 weeks.",
                "estimated_time": 12,
                "difficulty": "medium"
            },
            {
                "name": "Bixie Cut",
                "category": "trendy",
                "gender": "female",
                "length": "short",
                "maintenance": "low",
                "description": "A hybrid cut that blends the length and shape of a short bob with the layers and texture of a pixie cut. It's longer than a pixie but shorter than a bob, offering a soft, versatile, and low-maintenance short style.",
                "face_shapes": ["oval", "heart", "round"],
                "tags": ["Short hair", "Hybrid", "Layered", "Versatile"],
                "styling_tips": "5-10 minutes styling. Trim every 4-6 weeks.",
                "estimated_time": 7,
                "difficulty": "easy"
            },
        ]

        # Populate database
        added_count = 0
        updated_count = 0
        skipped_count = 0

        with transaction.atomic():
            for data in hairstyles_data:
                category_name = data.pop('category')
                category = categories.get(category_name)
                
                # Map gender to model format
                gender_map = {'male': 'male', 'female': 'female', 'unisex': 'unisex'}
                suitable_gender = gender_map.get(data.pop('gender', 'unisex'), 'unisex')
                
                # Map length to array format
                length_str = data.pop('length', 'medium')
                hair_lengths = [length_str]
                
                # First, delete any duplicate hairstyles with the same name
                existing_hairstyles = Hairstyle.objects.filter(name=data['name'])
                if existing_hairstyles.count() > 1:
                    # Keep the first one, delete the rest
                    to_delete = list(existing_hairstyles[1:])
                    for duplicate in to_delete:
                        duplicate.delete()
                        self.stdout.write(self.style.WARNING(f"🗑️  Deleted duplicate: {data['name']}"))
                
                # Now use update_or_create safely
                hairstyle, created = Hairstyle.objects.update_or_create(
                    name=data['name'],
                    defaults={
                        'description': data.get('description', ''),
                        'category': category,
                        'suitable_gender': suitable_gender,
                        'hair_lengths': hair_lengths,
                        'face_shapes': data.get('face_shapes', []),
                        'tags': data.get('tags', []),
                        'maintenance': data.get('maintenance', 'medium'),
                        'difficulty': data.get('difficulty', 'medium'),
                        'estimated_time': data.get('estimated_time', 30),
                        'styling_tips': data.get('styling_tips', ''),
                        'trend_score': 7.0,
                        'popularity_score': 5.0,
                        'is_active': True,
                    }
                )
                
                if created:
                    added_count += 1
                    self.stdout.write(self.style.SUCCESS(f"✓ Added: {hairstyle.name}"))
                else:
                    updated_count += 1
                    self.stdout.write(self.style.WARNING(f"↻ Updated: {hairstyle.name}"))

        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(self.style.SUCCESS("✅ DATABASE POPULATION COMPLETE!"))
        self.stdout.write("=" * 80)
        self.stdout.write(f"\n📊 Summary:")
        self.stdout.write(f"   Added: {added_count} new hairstyles")
        self.stdout.write(f"   Updated: {updated_count} existing hairstyles")
        self.stdout.write(f"   Total in database: {Hairstyle.objects.filter(is_active=True).count()}")
        
        self.stdout.write(f"\n📈 Gender Distribution:")
        male_count = Hairstyle.objects.filter(is_active=True, suitable_gender='male').count()
        female_count = Hairstyle.objects.filter(is_active=True, suitable_gender='female').count()
        unisex_count = Hairstyle.objects.filter(is_active=True, suitable_gender='unisex').count()
        self.stdout.write(f"   Male: {male_count}")
        self.stdout.write(f"   Female: {female_count}")
        self.stdout.write(f"   Unisex: {unisex_count}")
        
        self.stdout.write("\n" + "=" * 80 + "\n")

    def ensure_categories(self):
        """Ensure all required categories exist"""
        categories_data = {
            'natural': {'name': 'Natural Styles', 'description': 'Embracing natural texture and low-maintenance beauty'},
            'casual': {'name': 'Casual Styles', 'description': 'Easy-going, everyday hairstyles'},
            'classic': {'name': 'Classic Styles', 'description': 'Timeless, sophisticated cuts'},
            'elegant': {'name': 'Elegant Styles', 'description': 'Refined styles for formal occasions'},
            'glamorous': {'name': 'Glamorous Styles', 'description': 'Bold, high-impact looks'},
            'trendy': {'name': 'Trendy Styles', 'description': 'Current viral and modern styles'},
        }
        
        categories = {}
        for key, data in categories_data.items():
            category, created = HairstyleCategory.objects.get_or_create(
                name=data['name'],
                defaults={'description': data['description'], 'is_active': True}
            )
            categories[key] = category
            if created:
                self.stdout.write(self.style.SUCCESS(f"✓ Created category: {category.name}"))
        
        return categories
