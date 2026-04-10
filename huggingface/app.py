"""
Zero-Shot Industrial Anomaly Detection using WinCLIP
Hugging Face Spaces — Gradio Application

This is a self-contained Gradio app that wraps the WinCLIP model.
Deploy this to Hugging Face Spaces for a public API + UI.
"""

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import io
import json
import traceback

# ============================
# CONFIG
# ============================
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = (224, 224)
STRIDE = 56  # 75% overlap

OBJECTS = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]

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


# ============================
# MODEL (Loaded once at startup)
# ============================
print(f"Loading WinCLIP model on {DEVICE}...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully!")


# ============================
# CORE FUNCTIONS
# ============================

def encode_text(category):
    """Generates averaged text embeddings for normal and defect states."""
    normal_prompts = [tpl.format(f"{state} {category}") for state in NORMAL_STATES for tpl in TEMPLATES]
    defect_prompts = [tpl.format(f"{state} {category}") for state in DEFECT_STATES for tpl in TEMPLATES]

    all_prompts = normal_prompts + defect_prompts

    inputs = processor(text=all_prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        if hasattr(outputs, 'pooler_output'):
            text_features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            text_features = outputs.last_hidden_state[:, 0, :]
        else:
            text_features = outputs

    text_features /= text_features.norm(dim=-1, keepdim=True)

    n_normal = len(normal_prompts)
    normal_features = text_features[:n_normal].mean(dim=0, keepdim=True)
    defect_features = text_features[n_normal:].mean(dim=0, keepdim=True)

    normal_features /= normal_features.norm(dim=-1, keepdim=True)
    defect_features /= defect_features.norm(dim=-1, keepdim=True)

    return torch.cat([normal_features, defect_features], dim=0)


def extract_windows(image):
    """Slides a window across the image and returns patches."""
    W_h, W_w = WINDOW_SIZE
    stride = STRIDE

    w, h = image.size
    patches = []
    coordinates = []

    for y in range(0, h - W_h + 1, stride):
        for x in range(0, w - W_w + 1, stride):
            patch = image.crop((x, y, x + W_w, y + W_h))
            patches.append(patch)
            coordinates.append((x, y))

    return patches, coordinates, (w, h)


def predict(image, category="object"):
    """
    Runs WinCLIP inference on a single image.
    Returns: heatmap (2D numpy array), max_score (float)
    """
    text_embedding = encode_text(category)

    patches, coords, img_dims = extract_windows(image)

    if not patches:
        patches = [image.resize(WINDOW_SIZE)]
        coords = [(0, 0)]

    batch_size = 32 if DEVICE == "cuda" else 16
    patch_scores = []

    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i + batch_size]
        inputs = processor(images=batch_patches, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            img_outputs = model.get_image_features(**inputs)
            if hasattr(img_outputs, 'pooler_output'):
                image_features = img_outputs.pooler_output
            elif hasattr(img_outputs, 'last_hidden_state'):
                image_features = img_outputs.last_hidden_state[:, 0, :]
            else:
                image_features = img_outputs

        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_embedding.T).softmax(dim=-1)
        batch_scores = similarity[:, 1].cpu().numpy()
        patch_scores.extend(batch_scores)

    # Construct Heatmap
    w, h = img_dims
    heatmap = np.zeros((h, w))
    count_map = np.zeros((h, w))

    W_h, W_w = WINDOW_SIZE

    for score, (x, y) in zip(patch_scores, coords):
        heatmap[y:y + W_h, x:x + W_w] += score
        count_map[y:y + W_h, x:x + W_w] += 1

    heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map != 0)

    # Gaussian smoothing for noise reduction
    heatmap = gaussian_filter(heatmap, sigma=4)

    max_score = float(heatmap.max())

    # Normalize for visualization
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Ensure float32 for OpenCV compatibility
    heatmap = heatmap.astype(np.float32)

    return heatmap, max_score


def create_overlay(image, heatmap):
    """Creates a heatmap overlay on the original image."""
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return Image.fromarray(overlay_rgb)


# ============================
# GRADIO INTERFACE FUNCTION
# ============================

def analyze_image(input_image, category):
    """
    Main Gradio function: Takes an image and category, returns the overlay + results.
    """
    if input_image is None:
        return None, "⚠️ Please upload an image first."

    try:
        # Convert to PIL if needed (Gradio sends numpy)
        if isinstance(input_image, np.ndarray):
            image = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")
        else:
            image = input_image.convert("RGB")

        # Resize for speed (same as Flask dashboard)
        image.thumbnail((400, 400), Image.Resampling.LANCZOS)

        # Run inference
        heatmap, max_score = predict(image, category)

        # Create overlay visualization
        overlay = create_overlay(image, heatmap)

        # Determine status
        threshold = 0.75
        status = "🔴 Defect Detected" if max_score > threshold else "🟢 Normal"

        result_text = (
            f"**Status:** {status}\n\n"
            f"**Anomaly Score:** {max_score:.4f}\n\n"
            f"**Category:** {category}\n\n"
            f"**Threshold:** {threshold}\n\n"
            f"**Device:** {DEVICE.upper()}"
        )

        return overlay, result_text

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error during analysis: {error_msg}")
        return None, f"❌ **Error during analysis:**\n```\n{str(e)}\n```"


# ============================
# GRADIO APP UI
# ============================

css = """
.gradio-container {
    font-family: 'Inter', sans-serif !important;
}
#title {
    text-align: center;
    margin-bottom: 0.5em;
}
#subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 1.5em;
}
"""

with gr.Blocks(css=css, title="WinCLIP — Zero-Shot Defect Detection", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🔍 Zero-Shot Industrial Anomaly Detection
        ### WinCLIP Architecture — No Training Data Required
        Upload a product image, select its category, and the AI will detect defects using only natural language understanding.
        """,
        elem_id="title"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📷 Upload Product Image",
                type="numpy",
                height=350
            )
            category_dropdown = gr.Dropdown(
                choices=OBJECTS,
                value="bottle",
                label="🏭 Product Category",
                info="Select the type of product being inspected"
            )
            analyze_btn = gr.Button(
                "🚀 Analyze for Defects",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="🔥 Defect Heatmap Overlay",
                type="pil",
                height=350
            )
            result_text = gr.Markdown(
                label="📊 Analysis Results",
                value="*Upload an image and click Analyze to see results.*"
            )

    analyze_btn.click(
        fn=analyze_image,
        inputs=[input_image, category_dropdown],
        outputs=[output_image, result_text],
        api_name="predict"
    )

    gr.Markdown(
        """
        ---
        **How it works:** The WinCLIP model scans your image using a sliding window (224×224px, 75% overlap).
        Each patch is compared against text descriptions of "normal" and "defective" states using CLIP's
        cosine similarity. Red regions in the heatmap indicate high defect probability.

        **Built by:** Ritu Raj Singh | KIIT University | M.Tech CSE
        """
    )


# ============================
# LAUNCH
# ============================
if __name__ == "__main__":
    demo.launch()
