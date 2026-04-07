import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from ..config import Config

class WinCLIP:
    def __init__(self):
        self.device = Config.DEVICE
        print(f"Loading WinCLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained(Config.MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)
        print("Model loaded.")

    def encode_text(self, category):
        """Generates text embeddings for normal and defect states."""
        # 1. Generate text prompts
        normal_prompts = [tpl.format(f"{state} {category}") for state in Config.NORMAL_STATES for tpl in Config.TEMPLATES]
        defect_prompts = [tpl.format(f"{state} {category}") for state in Config.DEFECT_STATES for tpl in Config.TEMPLATES]
        
        all_prompts = normal_prompts + defect_prompts
        
        # 2. Tokenize and Encode
        inputs = self.processor(text=all_prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            if hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            else:
                text_features = outputs
            
        # 3. Normalize
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 4. Average embeddings for Ensemble
        # Split back into normal and defect groups
        n_normal = len(normal_prompts)
        normal_features = text_features[:n_normal].mean(dim=0, keepdim=True)
        defect_features = text_features[n_normal:].mean(dim=0, keepdim=True)
        
        # Re-normalize after averaging
        normal_features /= normal_features.norm(dim=-1, keepdim=True)
        defect_features /= defect_features.norm(dim=-1, keepdim=True)
        
        # Stack: [2, Feature_Dim] -> Index 0: Normal, Index 1: Defect
        return torch.cat([normal_features, defect_features], dim=0)

    def extract_windows(self, image):
        """Slides a window across the image and returns patches."""
        # Resize image to a larger size to allow for sliding windows if needed, 
        # or use the original image if it's large enough.
        # For simplicity/speed on CPU, we might resize the image to a fixed large size first.
        # CLIP expects specific inputs, but here we manually crop.
        
        # Standardize input size for patches (224x224)
        W_h, W_w = Config.WINDOW_SIZE
        stride = Config.STRIDE
        
        w, h = image.size
        # Unfold/Sliding Window manual implementation for PIL Image
        patches = []
        coordinates = [] # (x, y)
        
        # Simple sliding window
        for y in range(0, h - W_h + 1, stride):
            for x in range(0, w - W_w + 1, stride):
                patch = image.crop((x, y, x + W_w, y + W_h))
                patches.append(patch)
                coordinates.append((x, y))
                
        return patches, coordinates, (w, h)

    def predict(self, image, category="object"):
        """
        Runs WinCLIP inference.
        Returns:
            heatmap: 2D numpy array of anomaly scores.
            score: Max anomaly score (float).
        """
        # 1. Prepare Text Embeddings
        text_embedding = self.encode_text(category) # shape [2, 512]
        
        # 2. Extract Patches (Local Features)
        patches, coords, img_dims = self.extract_windows(image)
        
        if not patches:
            # Image smaller than window?
            patches = [image.resize(Config.WINDOW_SIZE)]
            coords = [(0,0)]
            
        # Process patches in batches to avoid OOM on CPU RAM if large image
        # Using Batch Size 16 for CPU safety
        batch_size = 16
        patch_scores = []
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            inputs = self.processor(images=batch_patches, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                if hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                else:
                    image_features = outputs
                
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Similarity: [Batch, 512] @ [2, 512].T -> [Batch, 2]
            # Index 1 is "Defect" score
            similarity = (100.0 * image_features @ text_embedding.T).softmax(dim=-1)
            batch_scores = similarity[:, 1].cpu().numpy() # Get defect probability
            patch_scores.extend(batch_scores)

        # 3. Construct Heatmap
        w, h = img_dims
        heatmap = np.zeros((h, w))
        count_map = np.zeros((h, w))
        
        W_h, W_w = Config.WINDOW_SIZE
        
        for score, (x, y) in zip(patch_scores, coords):
            heatmap[y:y+W_h, x:x+W_w] += score
            count_map[y:y+W_h, x:x+W_w] += 1
            
        # Average overlap
        heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map!=0)
        
        # --- AU-ROC BOOST: Gaussian Smoothing ---
        # Anomalies form spatial clusters. Smoothing the raw patch scores into a continuous 
        # map removes isolated false-positive noise spikes, significantly increasing metrics.
        heatmap = gaussian_filter(heatmap, sigma=4)
        
        # Calculate max score BEFORE normalization for image-level classification
        max_score = heatmap.max()
        
        # Normalize heatmap 0-1 for visualization
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
        return heatmap, max_score
