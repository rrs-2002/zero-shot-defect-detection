import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.winclip import WinCLIP
from src.config import Config

def evaluate_category(category, limit=None):
    print(f"\n--- Evaluating Category: {category} ---")
    
    # Paths
    test_dir = os.path.join(Config.DATA_DIR, category, 'test')
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    # Load Model
    model = WinCLIP()
    
    y_true = []
    y_scores = []
    
    # Iterate through subfolders
    # 'good' -> Label 0
    # others (defect) -> Label 1
    
    subfolders = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    for subfolder in subfolders:
        is_defect = 1 if subfolder != 'good' else 0
        folder_path = os.path.join(test_dir, subfolder)
        
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Limit for faster testing if requested
        if limit:
            images = images[:limit]
            
        print(f"Processing {subfolder} ({len(images)} images)...")
        
        for img_name in tqdm(images, ascii=True):
            try:
                img_path = os.path.join(folder_path, img_name)
                image = Image.open(img_path).convert("RGB")
                
                # Predict
                _, score = model.predict(image, category)
                
                y_true.append(is_defect)
                y_scores.append(score)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # Calculate Metrics
    if not y_true:
        print("No data found.")
        return

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        print("Dataset contains only one class. Cannot calculate AU-ROC.")
        print(f"Avg Score: {np.mean(y_scores):.4f}")
        return

    # AU-ROC
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"Error calculating AU-ROC: {e}")
        auroc = 0.0
        
    # Optimal F1-Score (finding best threshold)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_f1 = np.max(f1_scores)
    best_thresh = thresholds[np.argmax(f1_scores)]
    
    print(f"\nResults for {category}:")
    print(f"AU-ROC: {auroc:.4f}")
    print(f"Best F1-Score: {best_f1:.4f} at Threshold: {best_thresh:.4f}")
    
    return auroc, best_f1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        category = sys.argv[1]
        evaluate_category(category)
    else:
        print("Please provide a category name (e.g., python src/evaluate.py bottle)")
