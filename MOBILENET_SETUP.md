# MobileNetV3 Face Shape Classifier Integration Guide

## Step 1: Place Your Model File

1. Copy your trained MobileNetV3 `.pth` file to the models directory:
   ```
   backend/hairmixer_app/ml/models/mobilenetv3_small.pth
   ```

2. Alternative: If you want to use a different filename or location, you can specify the path when loading.

## Step 2: Update Face Shape Mapping (if needed)

The system is currently configured for 7 face shape classes. Update the FACE_SHAPES mapping in `backend/hairmixer_app/ml/model.py` to match your training labels:

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

If your model uses different preprocessing, update the `transform` in `face_analyzer.py`.

## Step 4: Test the Integration

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Test model loading:
   ```bash
   python manage.py test_mobilenet
   ```

3. Test with a specific model file:
   ```bash
   python manage.py test_mobilenet --model-path path/to/your/model.pth
   ```

4. Test with an image:
   ```bash
   python manage.py test_mobilenet --image-path path/to/test/image.jpg
   ```

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
- Check logs for detailed error messages
- Use the test command to isolate issues

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

1. The system will automatically use MobileNetV3 if available and loaded
2. Falls back to ResNet50 features if MobileNetV3 fails
3. Falls back to geometric analysis if both fail
4. Monitor logs to see which method is being used

## File Structure After Setup

```
backend/
├── hairmixer_app/
│   ├── ml/
│   │   ├── models/
│   │   │   └── mobilenetv3_face_shape_classifier.pth  # Your model here
│   │   ├── face_analyzer.py  # Updated with MobileNetV3 support
│   │   ├── mobilenet_classifier.py  # New model loader
│   │   └── model.py  # Updated face shapes mapping
│   └── management/
│       └── commands/
│           └── test_mobilenet.py  # Test command
```
