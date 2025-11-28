from django.core.management.base import BaseCommand
from hairmixer_app.models import Hairstyle, HairstyleCategory

class Command(BaseCommand):
    help = 'Adds missing natural/coily hairstyles to the database'

    def handle(self, *args, **options):
        self.stdout.write("Adding missing hairstyles...")

        # Ensure 'Natural Styles' category exists
        category, _ = HairstyleCategory.objects.get_or_create(
            name='Natural Styles',
            defaults={'description': 'Embracing natural texture and low-maintenance beauty'}
        )
        
        # Also get 'Trendy Styles' for some modern cuts
        trendy_category, _ = HairstyleCategory.objects.get_or_create(
            name='Trendy Styles',
            defaults={'description': 'Current viral and modern styles'}
        )

        missing_styles = [
            {
                "name": "Coily Crew Cut",
                "model_id": "coily_crew_cut",
                "category": category,
                "gender": "male",
                "length": "short",
                "description": "A low-maintenance, sharp look for coily hair. The hair is cut short and uniform, emphasizing the natural texture close to the scalp.",
                "face_shapes": ["oval", "round", "square", "oblong"],
                "tags": ["Low-maintenance", "Short", "Natural", "Clean"],
                "styling_tips": "Keep scalp moisturized. Regular trims every 2-3 weeks.",
                "maintenance": "low",
                "difficulty": "easy",
                "estimated_time": 5
            },
            {
                "name": "Coily Man Bun",
                "model_id": "coily_man_bun",
                "category": trendy_category,
                "gender": "male",
                "length": "long",
                "description": "A stylish way to manage long coily hair. The hair is pulled back and secured into a bun, protecting the ends and keeping hair off the face.",
                "face_shapes": ["oval", "square", "heart"],
                "tags": ["Protective", "Long hair", "Trendy", "Practical"],
                "styling_tips": "Use a leave-in conditioner before securing. Don't pull too tight to avoid tension.",
                "maintenance": "medium",
                "difficulty": "medium",
                "estimated_time": 10
            },
            {
                "name": "Coily Twists",
                "model_id": "coily_twists",
                "category": category,
                "gender": "unisex",
                "length": "medium",
                "description": "Two-strand twists created with natural coily hair. This versatile style protects the hair, retains moisture, and can be worn as-is or unraveled for a twist-out.",
                "face_shapes": ["oval", "round", "heart", "square"],
                "tags": ["Protective", "Versatile", "Natural", "Twists"],
                "styling_tips": "Apply twisting cream or butter. Wear a satin bonnet at night.",
                "maintenance": "medium",
                "difficulty": "medium",
                "estimated_time": 60
            },
            {
                "name": "Frohawk",
                "model_id": "frohawk",
                "category": trendy_category,
                "gender": "unisex",
                "length": "medium",
                "description": "An edgy style where the sides are faded, tapered, or pinned up, while the center strip of coily hair is left voluminous, mimicking a mohawk.",
                "face_shapes": ["oval", "round", "square"],
                "tags": ["Edgy", "Voluminous", "Statement", "Modern"],
                "styling_tips": "Pick out the center for volume. Use gel or pins to sleek down the sides.",
                "maintenance": "medium",
                "difficulty": "medium",
                "estimated_time": 20
            },
            {
                "name": "High Puff",
                "model_id": "high_puff",
                "category": category,
                "gender": "female",
                "length": "medium",
                "description": "A simple yet elegant style where coily hair is gathered at the crown of the head, allowing the natural volume to create a beautiful puff.",
                "face_shapes": ["oval", "round", "heart"],
                "tags": ["Quick", "Volume", "Updo", "Natural"],
                "styling_tips": "Use a wide headband or shoelace method to gather hair without tension. Smooth edges with gel.",
                "maintenance": "low",
                "difficulty": "easy",
                "estimated_time": 10
            },
            {
                "name": "Long Coils with Headband",
                "model_id": "long_coils_with_headband",
                "category": category,
                "gender": "female",
                "length": "long",
                "description": "Long, flowing coily hair accessorized with a headband. This style keeps hair off the face while showing off length and texture.",
                "face_shapes": ["oval", "heart", "square", "oblong"],
                "tags": ["Accessorized", "Long hair", "Casual", "Flowing"],
                "styling_tips": "Define coils with gel or cream. Place headband to frame the face.",
                "maintenance": "medium",
                "difficulty": "easy",
                "estimated_time": 15
            },
            {
                "name": "Long Coily Layers",
                "model_id": "long_coily_layers",
                "category": category,
                "gender": "female",
                "length": "long",
                "description": "A shape-enhancing cut for long coily hair. Layers are added to reduce bulk, prevent a triangle shape, and allow the curls to move freely.",
                "face_shapes": ["oval", "square", "round"],
                "tags": ["Layered", "Voluminous", "Shape", "Long hair"],
                "styling_tips": "Diffuse or air dry to maintain definition. Regular trims to keep layers fresh.",
                "maintenance": "medium",
                "difficulty": "medium",
                "estimated_time": 30
            },
            {
                "name": "Mid-Length Freeform Coils",
                "model_id": "mid-length_freeform_coils",
                "category": category,
                "gender": "unisex",
                "length": "medium",
                "description": "A style that embraces the organic, undefined nature of coily hair. It allows the hair to clump and form unique shapes naturally without manipulation.",
                "face_shapes": ["oval", "round", "square", "diamond"],
                "tags": ["Freeform", "Organic", "Natural", "Bold"],
                "styling_tips": "Keep clean and moisturized. Separate large clumps only if desired.",
                "maintenance": "low",
                "difficulty": "easy",
                "estimated_time": 10
            },
            {
                "name": "Short Coils with Fade",
                "model_id": "short_coils_with_fade",
                "category": trendy_category,
                "gender": "male",
                "length": "short",
                "description": "A modern cut featuring short, defined coils on top with faded sides. It offers a clean, sharp look with a touch of texture.",
                "face_shapes": ["oval", "square", "round"],
                "tags": ["Faded", "Modern", "Sharp", "Textured"],
                "styling_tips": "Use a sponge brush for coil definition on top. Keep fade fresh.",
                "maintenance": "medium",
                "difficulty": "easy",
                "estimated_time": 10
            },
            {
                "name": "Tapered Fro",
                "model_id": "tapered_fro",
                "category": category,
                "gender": "female",
                "length": "short",
                "description": "A chic, shaped afro where the sides and back are tapered short, leaving more length and volume on top. It gives the afro a structured, modern silhouette.",
                "face_shapes": ["oval", "heart", "diamond", "round"],
                "tags": ["Chic", "Shaped", "Volume", "Natural"],
                "styling_tips": "Pick the top for height. Define curls with cream if desired.",
                "maintenance": "medium",
                "difficulty": "medium",
                "estimated_time": 15
            },
            {
                "name": "Twist Out",
                "model_id": "twist_out",
                "category": category,
                "gender": "female",
                "length": "medium",
                "description": "The result of unraveling two-strand twists. This style creates defined, wavy/curly texture with plenty of volume and body.",
                "face_shapes": ["oval", "round", "heart", "square"],
                "tags": ["Defined", "Voluminous", "Textured", "Wavy"],
                "styling_tips": "Unravel carefully with oil to reduce frizz. Fluff roots for volume.",
                "maintenance": "medium",
                "difficulty": "medium",
                "estimated_time": 20
            },
            {
                "name": "Wash and Go",
                "model_id": "wash_and_go",
                "category": category,
                "gender": "female",
                "length": "medium",
                "description": "A styling method that defines the natural curl pattern using water and gel/cream, without heat or manipulation tools. It showcases the true texture of the hair.",
                "face_shapes": ["oval", "round", "heart", "diamond"],
                "tags": ["Natural Pattern", "Defined", "Curly", "Fresh"],
                "styling_tips": "Apply product to soaking wet hair. Air dry or diffuse.",
                "maintenance": "medium",
                "difficulty": "medium",
                "estimated_time": 30
            }
        ]

        added_count = 0
        for style_data in missing_styles:
            # Remove model_id as it's not a model field, just for our reference
            style_data.pop('model_id')
            
            # Use update_or_create to avoid duplicates
            obj, created = Hairstyle.objects.update_or_create(
                name=style_data['name'],
                defaults={
                    'category': style_data['category'],
                    'suitable_gender': style_data['gender'],
                    'hair_lengths': [style_data['length']],
                    'description': style_data['description'],
                    'face_shapes': style_data['face_shapes'],
                    'tags': style_data['tags'],
                    'styling_tips': style_data['styling_tips'],
                    'maintenance': style_data['maintenance'],
                    'difficulty': style_data['difficulty'],
                    'estimated_time': style_data['estimated_time'],
                    'is_active': True
                }
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Created: {obj.name}"))
                added_count += 1
            else:
                self.stdout.write(f"Updated: {obj.name}")

        self.stdout.write(self.style.SUCCESS(f"\nSuccessfully added/updated {added_count} hairstyles."))
