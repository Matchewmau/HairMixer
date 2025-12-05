"""
Utility to download ML models from Google Drive on first run.
This is used when models are too large to store in Git.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Google Drive folder ID from the shared link
GDRIVE_FOLDER_ID = "1TFjWz9Yq-ZZBFaJNr25ZeJUTaqf1TcX8"

# Model file IDs (you need to get these from Google Drive)
# To get file ID: Right-click file -> Get link -> Extract ID from URL
MODEL_FILES = {
    # ResNet50 face classifier model
    "resnet50_best_model.pth": {
        "id": None,  # Will be set from environment or discovered
        "path": "resnet50_best_model.pth",
    },
    # Hairstyle model files (in hairstyle_model folder)
    "hairstyle_model.joblib": {
        "id": None,
        "path": "hairstyle_model/hairstyle_model.joblib",
    },
    "hairstyle_family_label_encoder.joblib": {
        "id": None,
        "path": "hairstyle_model/hairstyle_family_label_encoder.joblib",
    },
    "model_columns.joblib": {
        "id": None,
        "path": "hairstyle_model/model_columns.joblib",
    },
}


def get_models_dir() -> Path:
    """Get the models directory path."""
    return Path(__file__).parent / "models"


def download_from_gdrive_folder(folder_id: str, output_dir: Path) -> bool:
    """
    Download all files from a Google Drive folder.
    
    Args:
        folder_id: Google Drive folder ID
        output_dir: Local directory to save files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import gdown
    except ImportError:
        logger.error("gdown not installed. Run: pip install gdown")
        return False
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download entire folder
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        logger.info(f"Downloading models from Google Drive folder: {folder_id}")
        
        gdown.download_folder(
            url=url,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
        )
        
        logger.info(f"Models downloaded successfully to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download models from Google Drive: {e}")
        return False


def download_file_from_gdrive(file_id: str, output_path: Path) -> bool:
    """
    Download a single file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        output_path: Local path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import gdown
    except ImportError:
        logger.error("gdown not installed. Run: pip install gdown")
        return False
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        url = f"https://drive.google.com/uc?id={file_id}"
        logger.info(f"Downloading {output_path.name} from Google Drive...")
        
        gdown.download(url, str(output_path), quiet=False)
        
        logger.info(f"Downloaded: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {output_path.name}: {e}")
        return False


def ensure_models_exist() -> bool:
    """
    Ensure all required models exist, downloading if necessary.
    
    Returns:
        True if all models are available, False otherwise
    """
    models_dir = get_models_dir()
    
    # Check if key model files exist
    resnet_path = models_dir / "resnet50_best_model.pth"
    hairstyle_model_path = models_dir / "hairstyle_model" / "hairstyle_model.joblib"
    
    if resnet_path.exists() and hairstyle_model_path.exists():
        logger.info("All model files already exist.")
        return True
    
    logger.info("Model files not found. Attempting to download from Google Drive...")
    
    # Try to download from folder
    folder_id = os.environ.get("GDRIVE_MODELS_FOLDER_ID", GDRIVE_FOLDER_ID)
    
    if download_from_gdrive_folder(folder_id, models_dir):
        # Verify download
        if resnet_path.exists() or hairstyle_model_path.exists():
            return True
    
    logger.warning("Could not download models. Some features may not work.")
    return False


def download_models_on_startup():
    """
    Called during Django app initialization to ensure models are available.
    Only downloads if models don't exist (to avoid slowing down every restart).
    """
    try:
        # Skip in development if models exist locally
        if os.environ.get("DJANGO_DEBUG", "true").lower() == "true":
            models_dir = get_models_dir()
            if (models_dir / "resnet50_best_model.pth").exists():
                return
        
        ensure_models_exist()
    except Exception as e:
        logger.error(f"Error during model download check: {e}")
