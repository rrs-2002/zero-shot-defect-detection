import os
import random
from PIL import Image
from transformers import CLIPProcessor
from ..config import Config

class DataLoader:
    def __init__(self):
        # We don't strictly need the processor here if we use it in the model, 
        # but good for standalone testing.
        pass
        
    def load_image(self, image_path):
        """Loads an image for CLIP."""
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
            
    def get_sample_image(self, category):
        """Returns a random test image path for a given category."""
        # Fix path construction to handle "dataset/archive/category/test/..."
        category_path = os.path.join(Config.DATA_DIR, category, 'test')
        if not os.path.exists(category_path):
            print(f"Category path not found: {category_path}")
            return None
            
        # Get all subfolders (defect types + good)
        subfolders = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
        
        # Prioritize defect images for demonstration
        defect_folders = [s for s in subfolders if s != 'good']
        if defect_folders:
            target_folder = random.choice(defect_folders)
        elif 'good' in subfolders:
            target_folder = 'good'
        else:
            return None
            
        image_dir = os.path.join(category_path, target_folder)
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if images:
            return os.path.join(image_dir, random.choice(images))
        return None
