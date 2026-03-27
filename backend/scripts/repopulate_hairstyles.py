import os
import sys
import django
import joblib

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from hairmixer_app.models import Hairstyle, HairstyleCategory

# Hairstyle Data Mapping
HAIRSTYLE_DATA = {
    "afro": {"gender": "unisex", "desc": "A natural, voluminous style that celebrates texture and fullness."},
    "beach_waves": {"gender": "female", "desc": "Effortless, tousled waves that give a relaxed and summery vibe."},
    "blunt_bangs": {"gender": "female", "desc": "A bold fringe cut straight across the forehead for a striking look."},
    "blunt_bob": {"gender": "female", "desc": "A chic, sharp bob cut that ends at the jawline for a modern silhouette."},
    "bouncy_curls": {"gender": "female", "desc": "A voluminous, playful style with defined curls that adds movement and life."},
    "caesar_cut": {"gender": "male", "desc": "A short, horizontally straight cut with a fringe, perfect for a low-maintenance look."},
    "chin-length_bob": {"gender": "female", "desc": "A classic bob that frames the face beautifully at the chin level."},
    "classic_taper": {"gender": "male", "desc": "A timeless cut that gets gradually shorter down the back and sides."},
    "coily_crew_cut": {"gender": "male", "desc": "A short, practical cut that keeps coils neat and manageable."},
    "coily_man_bun": {"gender": "male", "desc": "A stylish way to manage long coils by gathering them into a bun."},
    "coily_twists": {"gender": "unisex", "desc": "Defined twists that protect and showcase natural coily texture."},
    "crew_cut": {"gender": "male", "desc": "A classic military-style cut that is short all over and easy to maintain."},
    "curtain_bangs": {"gender": "female", "desc": "Long, sweeping bangs that part in the middle to frame the face like curtains."},
    "curtain_bangs_layers": {"gender": "female", "desc": "Soft curtain bangs blended with layers for a romantic, flowing look."},
    "curtains_style": {"gender": "male", "desc": "A 90s-inspired look with a middle part and long fringe framing the face."},
    "deep_side_part": {"gender": "female", "desc": "A dramatic side part that adds volume and elegance to any length."},
    "feathered_layers": {"gender": "female", "desc": "Soft, wispy layers that flip back, reminiscent of 70s style icons."},
    "french_crop": {"gender": "male", "desc": "A short, textured crop with a blunt fringe, offering a modern edge."},
    "frohawk": {"gender": "unisex", "desc": "A faux hawk style created with natural texture, edgy and bold."},
    "high_fade_with_volume": {"gender": "male", "desc": "Short sides with a high fade, leaving volume on top for styling."},
    "high_pompadour": {"gender": "male", "desc": "A high-volume style swept upwards and back for a retro, rockabilly vibe."},
    "high_ponytail": {"gender": "female", "desc": "A sleek or messy ponytail worn high on the head for a lifted look."},
    "high_puff": {"gender": "female", "desc": "A voluminous puff gathered high, perfect for showcasing natural curls."},
    "layered_cut": {"gender": "female", "desc": "Versatile layers that remove weight and add movement to the hair."},
    "lob_with_waves": {"gender": "female", "desc": "A long bob paired with soft waves for a trendy, versatile style."},
    "long_coils_with_headband": {"gender": "female", "desc": "Long, flowing coils accessorized with a headband for a sweet look."},
    "long_coily_layers": {"gender": "female", "desc": "Layered cut for long coils to add shape and reduce bulk."},
    "long_layers": {"gender": "female", "desc": "Long hair with layers to add texture and bounce without losing length."},
    "long_soft_waves": {"gender": "female", "desc": "Elegant, flowing waves that look polished and sophisticated."},
    "medium_waves": {"gender": "female", "desc": "Mid-length hair with natural-looking waves for everyday elegance."},
    "messy_fringe": {"gender": "male", "desc": "A textured, tousled fringe that adds a relaxed, youthful vibe."},
    "mid-length_freeform_coils": {"gender": "unisex", "desc": "Natural, free-flowing coils at a medium length."},
    "pixie_cut": {"gender": "female", "desc": "A short, cropped style that highlights facial features and is easy to style."},
    "pixie_with_volume": {"gender": "female", "desc": "A pixie cut with added length on top for volume and styling versatility."},
    "short_coils_with_fade": {"gender": "male", "desc": "Short coils on top with faded sides for a sharp, clean look."},
    "shoulder-length_flow": {"gender": "male", "desc": "Medium length hair pushed back for a relaxed, surfer-style flow."},
    "shoulder-length_waves": {"gender": "female", "desc": "Waves that hit right at the shoulder, balancing volume and length."},
    "shoulder_waves": {"gender": "female", "desc": "Soft waves resting on the shoulders for a classic, feminine look."},
    "side-swept_bangs": {"gender": "female", "desc": "Bangs swept to one side to soften the forehead and frame the face."},
    "side_part": {"gender": "male", "desc": "A classic, professional look with a clean side part."},
    "side_swept_fringe": {"gender": "male", "desc": "A longer fringe swept to the side for a casual yet styled appearance."},
    "sleek_ponytail": {"gender": "female", "desc": "A smooth, pulled-back ponytail for a polished and high-fashion look."},
    "slick_back_tapered": {"gender": "male", "desc": "Hair combed straight back with tapered sides for a gentleman's cut."},
    "slick_back_volume": {"gender": "male", "desc": "Slicked back hair with added volume at the roots for a modern twist."},
    "spiky_hair": {"gender": "male", "desc": "Short hair styled upwards for a fun, energetic, and textured look."},
    "tapered_fro": {"gender": "male", "desc": "An afro shape that tapers down at the neck and ears for a clean profile."},
    "textured_crop": {"gender": "male", "desc": "A short crop with plenty of texture on top for a rugged, modern look."},
    "textured_quiff": {"gender": "male", "desc": "A quiff with added texture, offering a relaxed take on the classic style."},
    "textured_quiff_medium": {"gender": "male", "desc": "A medium-length quiff with texture, balancing volume and messiness."},
    "twist_out": {"gender": "female", "desc": "A defined curly style created by untwisting set hair."},
    "undercut": {"gender": "male", "desc": "Short or shaved sides with longer hair on top, high contrast and bold."},
    "wash_and_go": {"gender": "female", "desc": "A natural style that embraces the hair's natural texture with minimal styling."},
    "wavy_lob": {"gender": "female", "desc": "A wavy long bob that is chic, modern, and low maintenance."},
    "wavy_mid-length": {"gender": "female", "desc": "Medium length hair with waves, a versatile and popular choice."},
    "wavy_shoulder-length": {"gender": "female", "desc": "Shoulder-grazing hair with waves for a soft and flattering look."},
}

def get_category(name):
    name_lower = name.lower()
    if any(k in name_lower for k in ['short', 'pixie', 'crop', 'crew', 'buzz', 'fade', 'taper', 'caesar', 'spiky', 'undercut']):
        return "Short Styles"
    elif any(k in name_lower for k in ['long']):
        return "Long Styles"
    elif any(k in name_lower for k in ['medium', 'mid-length', 'shoulder', 'bob', 'lob']):
        return "Medium Length"
    elif any(k in name_lower for k in ['coil', 'afro', 'twist', 'puff', 'textured', 'curly', 'curls']):
        return "Curly & Textured"
    elif any(k in name_lower for k in ['wave', 'wavy']):
        return "Wavy Styles"
    elif any(k in name_lower for k in ['bun', 'ponytail', 'updo']):
        return "Updos & Buns"
    return "Uncategorized"

def repopulate():
    print("Starting hairstyle repopulation...")
    
    # Ensure categories exist
    categories = [
        "Short Styles", "Long Styles", "Medium Length", 
        "Curly & Textured", "Wavy Styles", "Updos & Buns", "Uncategorized"
    ]
    cat_objs = {}
    for cat_name in categories:
        c, _ = HairstyleCategory.objects.get_or_create(name=cat_name)
        cat_objs[cat_name] = c

    # Load encoder to get the official list of classes
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, 'hairmixer_app', 'ml', 'models', 'hairstyle_model')
    ENCODER_PATH = os.path.join(MODEL_DIR, 'hairstyle_family_label_encoder.joblib')
    
    if not os.path.exists(ENCODER_PATH):
        print("Encoder not found! Cannot verify against model classes.")
        return

    encoder = joblib.load(ENCODER_PATH)
    model_classes = encoder.classes_
    
    print(f"Found {len(model_classes)} classes in the model.")

    for hairstyle_name in model_classes:
        # Normalize name for display if needed, but keep the key matching the model class
        # The model class is the 'name' in DB usually, or we map it.
        # Let's assume the DB name should match the model class for simplicity in lookup,
        # OR we use a display name.
        # The user wants "High Pompadour" (spaces) but model has "high_pompadour" (underscores).
        # I will convert underscores to spaces for the DB name, but ensure we can map back if needed.
        # Actually, for the recommendation system to work, the names might need to match or be mapped.
        # If the recommendation system uses the DB name to find the model class, we need to be careful.
        # Let's check if the previous system used exact matches.
        # The previous script used `display_name = name.replace('_', ' ').title()`.
        # I will do the same.
        
        display_name = hairstyle_name.replace('_', ' ').title()
        
        # Get data from our map
        data = HAIRSTYLE_DATA.get(hairstyle_name)
        if not data:
            print(f"Warning: No data found for {hairstyle_name}, using defaults.")
            data = {
                "gender": "unisex",
                "desc": f"A stylish {display_name}."
            }
        
        category_name = get_category(display_name)
        category = cat_objs[category_name]

        # Update or Create
        # We search by name (display_name)
        obj, created = Hairstyle.objects.update_or_create(
            name__iexact=display_name,
            defaults={
                'name': display_name,
                'description': data['desc'],
                'suitable_gender': data['gender'],
                'category': category,
                'is_active': True
            }
        )
        
        action = "Created" if created else "Updated"
        print(f"{action}: {display_name} ({data['gender']}) - {category_name}")

    print("Repopulation complete.")

if __name__ == '__main__':
    repopulate()
