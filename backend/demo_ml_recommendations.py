"""
Complete API demonstration: Upload image → Detect face → Get 10 ML recommendations

This script demonstrates the complete flow:
1. Upload an image with face
2. ResNet50 detects face shape automatically
3. Create user preferences (including detected face shape)
4. Random Forest model generates 10 hairstyle recommendations
"""
import requests
import json
import os
import sys

# Configuration
API_BASE_URL = "http://localhost:8000/api"
TEST_IMAGE_PATH = "test_face.jpg"  # Replace with your test image

# Color codes for terminal output
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(80)}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 80}{RESET}\n")


def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")


def print_info(text):
    """Print info message"""
    print(f"{YELLOW}ℹ {text}{RESET}")


def step_1_upload_image():
    """Step 1: Upload image and detect face shape"""
    print_header("STEP 1: Upload Image & Detect Face Shape")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print_error(f"Test image not found: {TEST_IMAGE_PATH}")
        print_info("Please provide a test image with a clear face")
        return None
    
    print_info(f"Uploading image: {TEST_IMAGE_PATH}")
    
    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{API_BASE_URL}/upload/",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Image uploaded successfully!")
            
            # Extract face detection results
            image_id = data.get('id')
            face_detected = data.get('face_detected', False)
            faceshape = data.get('faceshape')
            confidence = data.get('faceshape_confidence', 0)
            
            print(f"\n  Image ID: {image_id}")
            print(f"  Face Detected: {face_detected}")
            
            if faceshape:
                print(f"  {GREEN}Face Shape: {faceshape.upper()}{RESET}")
                print(f"  Confidence: {confidence:.1f}%")
                print(f"\n  {YELLOW}ResNet50 Model: ✓ Working{RESET}")
            else:
                print_error("Face shape not detected")
            
            return {
                'image_id': image_id,
                'faceshape': faceshape,
                'confidence': confidence
            }
        else:
            print_error(f"Upload failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
            
    except Exception as e:
        print_error(f"Error uploading image: {e}")
        return None


def step_2_create_preferences(face_data):
    """Step 2: Create user preferences with detected face shape"""
    print_header("STEP 2: Create User Preferences")
    
    # Sample preferences with detected face shape
    preferences = {
        "image_id": face_data['image_id'],
        "faceshape": face_data['faceshape'],
        "faceshape_confidence": face_data['confidence'],
        "gender": "female",
        "hair_type": "wavy",
        "hair_length": "medium",
        "volume": "medium",
        "lifestyle": "professional",
        "maintenance": "medium",
        "styling_maintenance": "medium",
        "occasions": ["work", "casual", "formal"],
        "hair_color": "brown",
        "hair_texture_detail": "normal",
        "styling_preference": "elegant",
        "hair_condition": "none",
        "hair_thickness": "medium",
        "wants_bangs": False
    }
    
    print_info("Creating preferences with 21 features:")
    print(f"  Face Shape: {preferences['faceshape']} (auto-detected)")
    print(f"  Gender: {preferences['gender']}")
    print(f"  Hair Type: {preferences['hair_type']}")
    print(f"  Hair Length: {preferences['hair_length']}")
    print(f"  Occasions: {', '.join(preferences['occasions'])}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/preferences/",
            json=preferences,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 201:
            data = response.json()
            preference_id = data.get('id')
            print_success(f"Preferences created: {preference_id}")
            return preference_id
        else:
            print_error(f"Failed to create preferences: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
            
    except Exception as e:
        print_error(f"Error creating preferences: {e}")
        return None


def step_3_get_ml_recommendations(preference_id):
    """Step 3: Get 10 ML-powered hairstyle recommendations"""
    print_header("STEP 3: Generate ML Recommendations")
    
    print_info("Requesting ML recommendations...")
    print(f"  Using Random Forest model: hairstyle_family_model.pkl")
    print(f"  Model features: 15 features")
    print(f"  Model classes: 26 hairstyle families")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend/ml/",
            json={"preference_id": preference_id},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            recommendation_count = data.get('recommendation_count', 0)
            model_used = data.get('model_used')
            faceshape = data.get('faceshape')
            recommendations = data.get('recommendations', [])
            
            print_success(f"Received {recommendation_count} recommendations!")
            print(f"\n  Model Used: {GREEN}{model_used}{RESET}")
            print(f"  Face Shape: {faceshape}")
            
            if recommendations:
                print(f"\n{BOLD}  TOP 10 HAIRSTYLE RECOMMENDATIONS:{RESET}")
                print(f"  {'-' * 76}")
                
                for i, rec in enumerate(recommendations[:10], 1):
                    name = rec['name']
                    score = rec['match_score']
                    confidence = rec['confidence']
                    family = rec['hairstyle_family']
                    category = rec['category']
                    
                    # Color code by confidence
                    if confidence >= 70:
                        color = GREEN
                    elif confidence >= 50:
                        color = YELLOW
                    else:
                        color = RESET
                    
                    print(f"  {i:2d}. {color}{name:<35}{RESET} | "
                          f"Score: {score:.3f} ({confidence:.1f}%)")
                    print(f"      Category: {category} | Family: {family}")
                    
                    if i == 1:
                        print(f"      Maintenance: {rec['maintenance']} | "
                              f"Difficulty: {rec['difficulty']} | "
                              f"Time: {rec['estimated_time']}min")
                    print()
            
            return recommendations
        else:
            print_error(f"Failed to get recommendations: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
            
    except Exception as e:
        print_error(f"Error getting recommendations: {e}")
        return None


def main():
    """Run the complete demonstration"""
    print(f"\n{BOLD}{GREEN}{'╔' + '═' * 78 + '╗'}{RESET}")
    print(f"{BOLD}{GREEN}║{' ' * 15}HAIRSTYLE RECOMMENDER API DEMONSTRATION{' ' * 20}║{RESET}")
    print(f"{BOLD}{GREEN}╚{'═' * 78}╝{RESET}")
    
    print(f"\n{YELLOW}This demonstration shows the complete ML recommendation flow:{RESET}")
    print(f"  1. Upload image → ResNet50 detects face shape")
    print(f"  2. Create preferences → 21 features collected")
    print(f"  3. ML model → Random Forest generates 10 recommendations")
    
    # Check API is running
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        print_success(f"API is running: {API_BASE_URL}")
    except:
        print_error(f"API is not running at {API_BASE_URL}")
        print_info("Please start the Django server: python manage.py runserver")
        return
    
    # Step 1: Upload image and detect face
    face_data = step_1_upload_image()
    if not face_data or not face_data['faceshape']:
        print_error("Cannot proceed without face detection")
        return
    
    # Step 2: Create preferences
    preference_id = step_2_create_preferences(face_data)
    if not preference_id:
        print_error("Cannot proceed without preferences")
        return
    
    # Step 3: Get ML recommendations
    recommendations = step_3_get_ml_recommendations(preference_id)
    
    # Summary
    print_header("DEMONSTRATION SUMMARY")
    if recommendations and len(recommendations) == 10:
        print_success("✅ All steps completed successfully!")
        print(f"\n  {GREEN}✓{RESET} Face shape detected: {face_data['faceshape']}")
        print(f"  {GREEN}✓{RESET} Preferences created: {preference_id}")
        print(f"  {GREEN}✓{RESET} Recommendations generated: {len(recommendations)}/10")
        print(f"\n  {BOLD}The hairstyle recommendation system is working perfectly!{RESET}")
    else:
        print_error("Some steps failed. Please check the errors above.")
    
    print("\n")


if __name__ == "__main__":
    main()
