"""
Create label encoder mapping from training data CSV.

This recreates the label encoding used during hairmixer_model.pkl training.
The model predicts numeric labels (0-202), we need to map them back to
hairstyle names.
"""

import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("CREATE LABEL ENCODER FOR HAIRMIXER_MODEL.PKL")
print("="*80)

# Load CSV
csv_path = Path(__file__).parent / 'hairmixer_app' / 'ml' / 'models' / 'merged_preferences_max_voting.csv'
print(f"\n📂 Loading data from: {csv_path}")

df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df)} records")

# Extract hairstyle names
hairstyle_names = df['hairstyle_name'].values
print(f"\n📊 Found {len(hairstyle_names)} hairstyle entries")

# Create label encoder (same way as during training)
le = LabelEncoder()
encoded_labels = le.fit_transform(hairstyle_names)

print(f"✅ Encoded into {len(le.classes_)} unique classes")

# Create mapping dictionary
label_to_name = {i: name for i, name in enumerate(le.classes_)}
name_to_label = {name: i for i, name in enumerate(le.classes_)}

print(f"\n📋 Sample Mappings (first 10):")
for i in range(min(10, len(label_to_name))):
    print(f"   {i} → {label_to_name[i]}")

# Save to JSON
output_path = Path(__file__).parent / 'hairmixer_app' / 'ml' / 'models' / 'label_encoder.json'

mapping_data = {
    'label_to_name': label_to_name,
    'name_to_label': name_to_label,
    'n_classes': len(le.classes_),
    'classes': list(le.classes_)
}

with open(output_path, 'w') as f:
    json.dump(mapping_data, f, indent=2)

print(f"\n💾 Saved label encoder to: {output_path}")
print(f"   Total classes: {len(le.classes_)}")
print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

print(f"\n{'='*80}")
print("✅ LABEL ENCODER CREATED SUCCESSFULLY!")
print("="*80)
print("\nYou can now use hairmixer_model.pkl with this label encoder to")
print("convert numeric predictions (0-202) to hairstyle names.")
print(f"\n{'='*80}\n")
