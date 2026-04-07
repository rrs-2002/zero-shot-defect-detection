import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from ..config import Config

class Visualizer:
    @staticmethod
    def save_results(image, heatmap, score, category, filename="result.jpg"):
        """Overlays heatmap on image and saves to static folder."""
        # Convert PIL to OpenCv
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        
        # Apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
        
        # Save path
        save_dir = os.path.join(Config.BASE_DIR, 'app', 'static', 'results')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, overlay)
        
        return f"results/{filename}"
