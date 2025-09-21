# MobileNetV3 Face Shape Classifier Integration Guide

## Step 1: Place Your Model File

1. Copy your trained MobileNetV3 `.pth` file to the models directory:
   ```
   backend/hairmixer_app/ml/models/mobilenetv3_small.pth
   ```

2. Alternative: If you want to use a different filename or location, set the path in your environment or update the loader in `mobilenet_classifier.py`.

## Step 2: Update Face Shape Mapping (if needed)

The system is currently configured for 6–7 face shape classes. Update the FACE_SHAPES mapping in `backend/hairmixer_app/ml/model.py` to match your training labels:

```python
FACE_SHAPES = {
    0: "oval",
    1: "round", 
    2: "square",
    3: "heart",
    4: "diamond",
    5: "oblong",
    6: "triangular"  # Update this to match your 7th class
}
```

## Step 3: Verify Model Architecture Compatibility

Your MobileNetV3-Small model should have:
- Input size: 224x224x3 (RGB images)
- Output: 7 classes for face shapes
- Standard ImageNet preprocessing (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

If your model uses different preprocessing, update the `transform` in `face_analyzer.py` (inside `FacialFeatureAnalyzer.__init__`).

## Step 4: Test the Integration

Use one of the following quick checks on Windows PowerShell:

1) Verify the analyzer loads MobileNetV3

```powershell
# From repo root
D:/CODING/Python/HairMixer/venv/Scripts/python.exe - <<'PY'
from backend.hairmixer_app.ml.face_analyzer import FacialFeatureAnalyzer
a = FacialFeatureAnalyzer()
print('shape_classifier:', a.shape_classifier)
print('feature_extractor:', type(a.feature_extractor).__name__ if a.feature_extractor else None)
PY
```

2) Run a one-off prediction on an image

```powershell
# Replace with your test image path
$img = 'D:/path/to/face.jpg'
D:/CODING/Python/HairMixer/venv/Scripts/python.exe - <<'PY'
from backend.hairmixer_app.ml.face_analyzer import analyze_face_comprehensive
import pathlib
result = analyze_face_comprehensive(pathlib.Path(r'''$env:IMG''') if False else r'''REPLACE''')
print(result)
PY
```
Note: The API also analyzes during `/api/upload/`.

## Step 5: Model File Format Requirements

Your `.pth` file should be saved in one of these formats:

### Option 1: State Dict Only
```python
torch.save(model.state_dict(), 'model.pth')
```

### Option 2: Complete Checkpoint
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  # optional
    'epoch': epoch,  # optional
    'loss': loss,  # optional
}, 'model.pth')
```

## Step 6: Troubleshooting

### Model Loading Issues
- Ensure PyTorch version compatibility
- Check if CUDA is available if you trained on GPU
- Verify file path permissions

### Prediction Issues
- Check image preprocessing matches training
- Verify input image dimensions
- Check class mapping indices

### Integration Issues
- Restart Django server after changes
- Set Django logging to DEBUG to see detailed messages (see Backend README)
- Use the shell snippets above to isolate issues

## Step 7: Production Deployment

1. Place your model file in the correct directory
2. Update Django settings if needed
3. Test thoroughly with various images
4. Monitor performance and accuracy

## Custom Preprocessing (if needed)

If your model uses different preprocessing than ImageNet standard, update the transform in `FacialFeatureAnalyzer.__init__()`:

```python
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Your input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[your_mean], std=[your_std])  # Your values
])
```

## Model Performance Tips

1. The system uses MobileNetV3-only for face shape classification.
2. Face detection uses MediaPipe by default (MTCNN as optional fallback if installed).
3. Monitor logs (DEBUG) to see which detector was used and the confidence.

## File Structure After Setup

```
backend/
├── hairmixer_app/
│   ├── ml/
│   │   ├── models/
│   │   │   └── mobilenetv3_face_shape_classifier.pth  # Your model here
│   │   ├── face_analyzer.py              # Uses MediaPipe for detection, MobileNetV3 for shape
│   │   ├── mobilenet_classifier.py      # MobileNetV3 model loader
│   │   └── model.py  # Updated face shapes mapping
│   └── management/
│       └── commands/               # (optional custom commands)
```
