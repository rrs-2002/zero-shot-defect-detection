import sys
import os
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.winclip import WinCLIP
from src.data.loader import DataLoader

def test_inference():
    print("Initializing components...")
    loader = DataLoader()
    model = WinCLIP()
    
    # Get a sample image
    # Try getting a bottle image
    image_path = loader.get_sample_image("bottle")
    if not image_path:
        print("No sample image found for 'bottle'. Trying to generate dummy...")
        image = Image.new('RGB', (224, 224), color = 'red')
    else:
        print(f"Testing on image: {image_path}")
        image = loader.load_image(image_path)
    
    if image:
        print("Running prediction...")
        heatmap, score = model.predict(image, "bottle")
        print(f"Prediction Complete.")
        print(f"Max Anomaly Score: {score}")
        print(f"Heatmap Shape: {heatmap.shape}")
    else:
        print("Failed to load image.")

if __name__ == "__main__":
    test_inference()
