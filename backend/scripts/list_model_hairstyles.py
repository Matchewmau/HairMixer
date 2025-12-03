import os
import sys
import joblib

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'hairmixer_app', 'ml', 'models', 'hairstyle_model')
ENCODER_PATH = os.path.join(MODEL_DIR, 'hairstyle_family_label_encoder.joblib')

def list_hairstyles():
    if not os.path.exists(ENCODER_PATH):
        print(f"Error: Encoder not found at {ENCODER_PATH}")
        return

    try:
        encoder = joblib.load(ENCODER_PATH)
        print("Hairstyle Classes found in model:")
        for i, label in enumerate(encoder.classes_):
            print(f"{i}: {label}")
    except Exception as e:
        print(f"Error loading encoder: {e}")

if __name__ == '__main__':
    list_hairstyles()
