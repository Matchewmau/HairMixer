"""
Quick Start ResNet50 Training Script
Simple script to train ResNet50 for face shape classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Face shape classes
FACE_SHAPES = ["oval", "round", "square", "heart", "diamond", "oblong"]


class SimpleResNet50(nn.Module):
    """Simple ResNet50 for face shape classification"""
    
    def __init__(self, num_classes=6, pretrained=True):
        super(SimpleResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        # Replace the final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


def get_data_transforms():
    """Get data transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_model(data_dir, num_epochs=10, batch_size=16, lr=0.001):
    """Train the model"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_transform, val_transform = get_data_transforms()
    
    # Load dataset (expects ImageFolder structure)
    try:
        full_dataset = ImageFolder(data_dir, transform=train_transform)
        logger.info(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} classes")
        logger.info(f"Classes: {full_dataset.classes}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None
    
    # Split dataset
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = SimpleResNet50(num_classes=len(full_dataset.classes), pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Print results
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {train_loss/len(train_loader):.4f}, '
                   f'Train Acc: {train_acc:.2f}%')
        logger.info(f'  Val Loss: {val_loss/len(val_loader):.4f}, '
                   f'Val Acc: {val_acc:.2f}%')
    
    # Save model
    save_path = Path('models')
    save_path.mkdir(exist_ok=True)
    model_path = save_path / 'resnet50_faceshape.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': full_dataset.classes,
        'num_classes': len(full_dataset.classes)
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Final validation accuracy: {val_acc:.2f}%")
    
    return model


def create_dataset_structure(base_dir='face_dataset'):
    """Create dataset directory structure"""
    base_path = Path(base_dir)
    
    for class_name in FACE_SHAPES:
        class_dir = base_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create instruction file
        with open(class_dir / 'README.txt', 'w') as f:
            f.write(f"Add {class_name} face images here\n")
            f.write("Supported: .jpg, .jpeg, .png\n")
    
    logger.info(f"Dataset structure created at: {base_path}")
    logger.info("Add your images and run: python quick_train.py")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'face_dataset'
    
    # Check if dataset exists
    if not Path(data_dir).exists():
        logger.info(f"Dataset directory '{data_dir}' not found.")
        logger.info("Creating sample dataset structure...")
        create_dataset_structure(data_dir)
        logger.info("\nDataset structure created!")
        logger.info("Steps to use:")
        logger.info("1. Add images to each class folder")
        logger.info("2. Run: python quick_train.py")
    else:
        # Start training
        logger.info(f"Training with dataset: {data_dir}")
        model = train_model(data_dir, num_epochs=20, batch_size=16, lr=0.001)
        
        if model:
            logger.info("Training completed successfully!")
        else:
            logger.error("Training failed!")
