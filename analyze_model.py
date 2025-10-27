import joblib
import pandas as pd

# Load model components
model = joblib.load('backend/hairmixer_app/ml/models/hairstyle_model/hairstyle_model.joblib')
columns = joblib.load('backend/hairmixer_app/ml/models/hairstyle_model/model_columns.joblib')
label_enc = joblib.load('backend/hairmixer_app/ml/models/hairstyle_model/hairstyle_family_label_encoder.joblib')

print("=" * 80)
print("HAIRSTYLE MODEL ANALYSIS")
print("=" * 80)

print("\n=== Model Type ===")
print(f"Type: {type(model).__name__}")
print(f"Number of estimators: {model.n_estimators}")

print("\n=== Target Classes (Hairstyle Names) ===")
print(f"Total classes: {len(label_enc.classes_)}")
print("Classes:")
for i, cls in enumerate(label_enc.classes_, 1):
    print(f"  {i:2d}. {cls}")

print("\n=== Features (Total: {}) ===".format(len(columns)))

# Group features by category
feature_groups = {}
for col in columns:
    parts = col.split('_', 1)
    if len(parts) == 2:
        category = parts[0]
        value = parts[1]
        if category not in feature_groups:
            feature_groups[category] = []
        feature_groups[category].append(value)

print("\nFeature Categories and Values:")
for category in sorted(feature_groups.keys()):
    values = feature_groups[category]
    print(f"\n{category.upper()} ({len(values)} values):")
    for val in sorted(values):
        print(f"  - {val}")

print("\n=== Required Input Features ===")
print("To use this model, you need to one-hot encode the following features:")
for category in sorted(feature_groups.keys()):
    print(f"  • {category}")
