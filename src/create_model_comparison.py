import json
import os
import sys
import numpy as np

# Add project root to path to import src modules
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.evaluate import evaluate_category
from src.config import Config

def create_model_comparison():
    print("--- Starting Dynamic Model Benchmarking (WinCLIP on local dataset) ---")

    dataset_dir = Config.DATA_DIR
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        local_auroc = 0.0
        local_f1 = 0.0
    else:
        categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        print(f"Found {len(categories)} categories: {categories}")

        aurocs = []
        f1s = []

        for category in categories:
            try:
                # Limit to 10 images per sub-folder for a fast but representative sample
                result = evaluate_category(category, limit=10)
                if result:
                    auc, f1 = result
                    aurocs.append(auc)
                    f1s.append(f1)
            except Exception as e:
                print(f"  [WARN] Failed to evaluate '{category}': {e}")

        local_auroc = float(np.mean(aurocs)) if aurocs else 0.0
        local_f1    = float(np.mean(f1s))    if f1s    else 0.0

    print(f"\n=== WinCLIP Averaged Results across {len(aurocs)} categories ===")
    print(f"  Average AU-ROC   : {local_auroc:.4f}")
    print(f"  Average F1-Score : {local_f1:.4f}")

    # Comparative analysis: WinCLIP local + literature benchmarks for others
    models_data = [
        {
            "Model_Name": "WinCLIP (this project)",
            "Core_Architecture": "Window-based CLIP with compositional prompt ensemble",
            "Key_Strengths": "Sliding window analysis excels at finding fine structural anomalies and missing components (screws, transistors).",
            "Limitations": "Lower global classification AUROC compared to newer methods; slower on CPU.",
            "Best_Use_Case": "Structural defects, logical anomalies, and small missing components on rigid objects.",
            "Zero_Shot_Performance_Notes": (
                f"Locally validated on MVTec AD ({len(aurocs)} categories, 10 samples/sub-folder): "
                f"avg AU-ROC = {local_auroc:.4f}, avg F1-Score = {local_f1:.4f}."
            )
        },
        {
            "Model_Name": "AnomalyCLIP",
            "Core_Architecture": "Object-agnostic text prompt learning with CLIP backbone",
            "Key_Strengths": "Learns generalizable, object-agnostic prompts — excels on texture-heavy categories (carpet, tile).",
            "Limitations": "Less specialized for fine structural / logical anomalies.",
            "Best_Use_Case": "Texture-based defect detection across a wide range of object types.",
            "Zero_Shot_Performance_Notes": "Benchmark (MVTec AD): ~0.916 Classification AUROC, ~0.907 Segmentation AUROC. Significantly outperforms WinCLIP on global metrics."
        },
        {
            "Model_Name": "AdaptCLIP",
            "Core_Architecture": "Hybrid static + dynamic learnable prompts with alternating visual/textual optimization",
            "Key_Strengths": "Adapts to each test image dynamically without any training data; jointly optimizes both modalities.",
            "Limitations": "Computationally heavier at inference due to dynamic adaptation steps.",
            "Best_Use_Case": "High-variability appearance settings where training is not available.",
            "Zero_Shot_Performance_Notes": "Benchmark: ~89.6% pixel-AUROC in zero-shot settings on MVTec AD."
        },
        {
            "Model_Name": "AnomalyGPT",
            "Core_Architecture": "LLM integrated with a Vision-Language Model (VLM)",
            "Key_Strengths": "No manual threshold tuning required; supports multi-turn textual reasoning and natural language defect explanations.",
            "Limitations": "High computational cost from LLM inference; latency unsuitable for real-time industrial lines.",
            "Best_Use_Case": "Interactive, explainable inspection pipelines where human-readable reasoning is required.",
            "Zero_Shot_Performance_Notes": "Benchmark: Leverages large-scale LLM world knowledge for superior reasoning without manual calibration."
        },
        {
            "Model_Name": "FiLo",
            "Core_Architecture": "LLM + Grounding DINO for fine-grained visual grounding",
            "Key_Strengths": "Targets micro-defects by grounding detailed textual descriptions to precise image regions.",
            "Limitations": "Requires multiple heavy models (LLM + Grounding DINO), increasing deployment complexity.",
            "Best_Use_Case": "Micro-defect detection where exact text-to-region grounding is critical.",
            "Zero_Shot_Performance_Notes": "Benchmark: State-of-the-art on fine-grained anomaly detection benchmarks."
        }
    ]

    # Save output relative to project root (no hardcoded old path)
    output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_comparison.json")

    with open(output_path, 'w') as f:
        json.dump(models_data, f, indent=4)

    print(f"\nSaved comparison JSON to: {output_path}")

if __name__ == "__main__":
    create_model_comparison()
