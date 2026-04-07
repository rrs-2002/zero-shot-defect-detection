import os
import torch

class Config:
    # Paths
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'Datasets', 'archive')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # ML Hyperparameters
    MODEL_NAME = "openai/clip-vit-base-patch32" # optimized for CPU speed
    DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
    BATCH_SIZE = 1
    
    # WinCLIP specific
    WINDOW_SIZE = (224, 224) # Standard CLIP input size
    STRIDE = 56 # 75% overlap. Dramatically increases localization AU-ROC.
    
    # Classes
    OBJECTS = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
        'leather', 'metal_nut', 'pill', 'screw', 'tile',
        'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    # Advanced Prompt Engineering (Boosts Semantic Understanding)
    NORMAL_STATES = ["normal", "perfect", "flawless", "damage-free", "pristine", "clean", "good condition"]
    DEFECT_STATES = ["damaged", "broken", "cracked", "scratched", "faulty", "defective", "cut", "hole", "anomaly", "blemish"]
    TEMPLATES = [
        "a photo of a {}",
        "a close-up photo of a {}",
        "a cropped photo of a {}",
        "a good photo of a {}",
        "a bright photo of a {}",
        "a dark photo of a {}",
        "a bad photo of a {}",
        "a blurry photo of a {}"
    ]
