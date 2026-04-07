# Zero-Shot Defect Detection

## Project Overview
This project implements a **Zero-Shot Industrial Anomaly Detection** system using **WinCLIP (Window-based CLIP)**. The core objective is to detect and precisely localize defects on manufactured components without requiring *any* prior training on defect data. It includes a Flask-based interactive web dashboard for real-time visual analysis.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Dataset Selection: MVTec AD](#1-dataset-selection-mvtec-ad)
- [Methodology & Architectural Choices](#2-methodology--architectural-choices)
- [Hardware Considerations & Optimizations](#3-hardware-considerations--optimizations)
- [Benchmarking and Evaluation](#4-benchmarking-and-evaluation)
- [Directory Structure](#directory-structure)
- [File-Level Description](#file-level-description)
- [Setup & Installation](#setup--installation)
- [Dashboard Usage](#dashboard-usage)
- [Testing](#testing)

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.x | Core programming language |
| **Deep Learning** | PyTorch, Torchvision | Tensor operations, model inference |
| **VLM Backbone** | HuggingFace Transformers (`openai/clip-vit-base-patch32`) | Pre-trained CLIP model for vision-language alignment |
| **Image Processing** | OpenCV (`opencv-python`), Pillow (PIL) | Image I/O, resizing, heatmap overlay generation |
| **Background Removal** | rembg, ONNX Runtime | AI-based foreground extraction for live camera analysis |
| **Scientific Computing** | NumPy, SciPy | Array operations, Gaussian smoothing of heatmaps |
| **ML Metrics** | scikit-learn | AU-ROC, F1-Score, Precision-Recall curve computation |
| **Web Framework** | Flask | Backend API and server-side rendering |
| **Frontend** | Bootstrap 5, Jinja2 Templates | Responsive dashboard UI |
| **Data Analysis** | Pandas, Matplotlib, Seaborn | Data manipulation and visualization |
| **Notebooks** | Jupyter | Interactive exploration and prototyping |
| **Utilities** | tqdm, ftfy, regex | Progress bars, text fixing for CLIP tokenization |

---

## 1. Dataset Selection: MVTec AD
**What it is:** The MVTec Anomaly Detection (MVTec AD) dataset is the industry standard benchmark for unsupervised anomaly detection. It contains high-resolution images of 15 different industrial object and texture categories.

**Why we used it:** 
- **Comprehensive:** It provides both normal (defect-free) images and images with various, highly-detailed anomalies (scratches, pokes, missing components, contaminations).
- **Ground Truth:** It provides pixel-precise ground truth masks for defects, making it perfect for evaluating both global classification and local segmentation performance.
- **Zero-Shot Evaluation:** Because our model does not train on this data, we use the `test/` splits of the MVTec dataset purely to validate the zero-shot capabilities of the VLM.

**Supported Categories (15):**
`bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`

---

## 2. Methodology & Architectural Choices

### Why WinCLIP?
Standard Vision-Language Models (VLMs) like OpenAI's CLIP are designed to classify an *entire* image. However, industrial defects are usually tiny, localized aberrations on an otherwise normal object. WinCLIP solves this by adapting CLIP for dense, patch-level prediction.

### Step-by-Step Pipeline
1. **Prompt Ensembling:** We generate multiple textual descriptions of the object in both its normal state ("a flawless photo of a [object]") and defective state ("a damaged [object] with a scratch"). These text prompts are fed into CLIP's text encoder and averaged to create robust, generalized embeddings ($T_{normal}$ and $T_{defect}$).

2. **Sliding Window Visual Extraction:** Instead of passing the whole image to CLIP's visual encoder, we slide a window (patch) across the image. For every patch, we extract a visual feature vector.

3. **Cosine Similarity Scoring:** Each localized visual patch is compared against the pre-calculated text embeddings. If a patch is mathematically closer to the "$T_{defect}$" embedding than the "$T_{normal}$" embedding in the latent space, it is assigned a high anomaly score.

4. **Heatmap Aggregation:** The scores from all the overlapping patches are aggregated back into a dense 2D spatial map, forming our defect heatmap.

---

## 3. Hardware Considerations & Optimizations
**Constraint:** This implementation runs entirely on a **CPU**. Standard ViT (Vision Transformer) backbones are highly compute-intensive.

**Optimizations Applied:**
- We utilize `openai/clip-vit-base-patch32` instead of the larger `patch16` or `patch14` variants. The larger stride significantly reduces the number of patches the model has to process, allowing for acceptable latency on standard CPUs.
- Text embeddings are calculated *once* during model initialization and cached, rather than dynamically computing them for every inference request.
- Images are downsampled to a maximum of 400×400 pixels before analysis to limit the number of sliding windows.
- Patches are processed in batches of 16 to manage CPU RAM usage.
- Gaussian smoothing (`sigma=4`) is applied to heatmaps to suppress isolated false-positive noise.

---

## 4. Benchmarking and Evaluation
To validate our implementation natively against literature, we built a dynamic evaluation script (`src/evaluate.py`).

- **Metrics:** The system calculates the threshold-independent **AU-ROC** (Area Under the Receiver Operating Characteristic curve) and the optimal **F1-Score**.
- **Comparative Analysis:** The results (average AU-ROC of ~0.77 and F1 of ~0.92 across the dataset with CPU optimizations) are statically compared against state-of-the-art models (like AnomalyCLIP, AnomalyGPT, FiLo) in `data/processed/model_comparison.json`.

---

## Directory Structure

```
Implementation/
│
├── run.py                      # Application entry point (starts Flask server)
├── run_dashboard.bat           # Windows batch script to launch the dashboard
├── run_evaluation.bat          # Windows batch script to run metric evaluation
├── run_tests.bat               # Windows batch script to run automated tests
├── requirements.txt            # Python dependency list
├── README.md                   # This documentation file
│
├── app/                        # Flask web application (frontend + API)
│   ├── __init__.py             # App factory — creates and configures the Flask app
│   ├── routes.py               # Route definitions (/, /dashboard, /analyze, /analyze_live)
│   ├── templates/              # Jinja2 HTML templates
│   │   ├── base.html           # Base layout with navbar and Bootstrap 5 CDN
│   │   ├── index.html          # Landing/home page with redirect to dashboard
│   │   └── dashboard.html      # Main interactive dashboard (upload + live camera)
│   └── static/                 # Static web assets
│       ├── css/
│       │   └── style.css       # Custom stylesheet
│       ├── js/
│       │   └── script.js       # Custom JavaScript
│       ├── uploads/            # Temporary storage for user-uploaded images
│       └── results/            # Generated heatmap overlay images
│
├── src/                        # Core ML pipeline source code
│   ├── __init__.py             # Package initializer
│   ├── config.py               # Central configuration (paths, hyperparameters, prompts)
│   ├── evaluate.py             # Benchmark evaluation script (AU-ROC, F1-Score)
│   ├── create_model_comparison.py  # Generates comparative analysis JSON/CSV
│   ├── models/                 # ML model implementations
│   │   ├── __init__.py         # Package initializer
│   │   ├── winclip.py          # WinCLIP zero-shot anomaly detection model
│   │   ├── architecture.py     # Model architecture definitions
│   │   ├── predict.py          # Prediction utilities
│   │   └── train.py            # Training utilities (reserved for future use)
│   ├── data/                   # Data handling modules
│   │   ├── __init__.py         # Package initializer
│   │   ├── loader.py           # Dataset loading and sampling utilities
│   │   └── preprocessing.py    # Image preprocessing utilities
│   └── utils/                  # Shared utility modules
│       ├── __init__.py         # Package initializer
│       ├── visualization.py    # Heatmap overlay generation using OpenCV
│       └── logger.py           # Logging utilities
│
├── data/                       # Data storage
│   ├── raw/                    # Raw, unprocessed data
│   │   └── Datasets/
│   │       ├── archive/        # Extracted MVTec AD dataset (15 categories)
│   │       └── archive.zip     # Original MVTec AD download (~5 GB)
│   └── processed/              # Processed outputs and analysis results
│       ├── model_comparison.json   # Comparative model benchmark data (JSON)
│       └── Comparison_Matrix.csv   # Evaluation comparison matrix (CSV)
│
├── models/                     # Model artifacts directory (downloaded weights cache)
│
├── notebooks/                  # Jupyter notebooks for exploration and prototyping
│
├── docs/                       # Documentation and presentations
│   └── presentations/          # LaTeX presentation source and compiled PDFs
│       ├── presentation.tex    # Initial presentation slide deck (LaTeX source)
│       ├── presentation.pdf    # Compiled initial presentation
│       ├── final_review.tex    # Final review presentation (LaTeX source)
│       ├── final_review.pdf    # Compiled final review presentation
│       └── *.aux, *.log, ...   # LaTeX compilation artifacts
│
├── tests/                      # Automated test suite
│   ├── test_model.py           # Unit tests for model loading and text encoding
│   └── test_inference.py       # End-to-end inference test with sample images
│
└── .venv/                      # Python virtual environment (not tracked in version control)
```

---

## File-Level Description

### Root Files

| File | Purpose |
|---|---|
| `run.py` | Application entry point. Imports the Flask app factory from `app/__init__.py` and starts the development server on `http://127.0.0.1:5000`. |
| `run_dashboard.bat` | Convenience Windows batch script that sets `FLASK_APP` and `FLASK_ENV` environment variables and launches the Flask dashboard. |
| `run_evaluation.bat` | Interactive batch script that prompts the user for a category name (e.g., `bottle`) and runs `src/evaluate.py` to benchmark that category. |
| `run_tests.bat` | Batch script that sequentially executes `test_model.py` and `test_inference.py` from the `tests/` directory. |
| `requirements.txt` | Lists all 18 Python dependencies required by the project (Flask, PyTorch, Transformers, OpenCV, rembg, etc.). |

---

### `app/` — Flask Web Application

| File | Purpose |
|---|---|
| `__init__.py` | **App factory function** (`create_app()`). Creates a Flask instance, loads configuration from `src.config.Config`, and registers the `main` Blueprint from `routes.py`. |
| `routes.py` | Defines 4 Flask routes: `GET /` (home), `GET /dashboard` (main UI), `POST /analyze` (image upload inference), `POST /analyze_live` (webcam snapshot inference with `rembg` background removal). Lazy-loads the WinCLIP model on first request. |
| `templates/base.html` | Jinja2 base template providing the HTML skeleton, Bootstrap 5.3 CDN, dark-themed navbar, and block placeholders for child templates. |
| `templates/index.html` | Landing page extending `base.html`. Displays a hero banner with a call-to-action button redirecting to the dashboard. |
| `templates/dashboard.html` | Primary interactive UI with a tabbed interface (Upload / Live Camera), a category dropdown selector for all 15 MVTec AD object types, analysis result cards (status + anomaly score), and the heatmap visualization area. Contains embedded JavaScript for async form submissions and WebRTC camera access. |
| `static/css/style.css` | Custom CSS stylesheet for additional styling beyond Bootstrap defaults. |
| `static/js/script.js` | Custom JavaScript file for any additional frontend logic. |
| `static/uploads/` | Temporary directory where user-uploaded and live-captured images are saved before inference. |
| `static/results/` | Directory where generated heatmap overlay result images are saved and served to the frontend. |

---

### `src/` — Core ML Pipeline

| File | Purpose |
|---|---|
| `config.py` | **Central configuration class** defining: project paths (`BASE_DIR`, `DATA_DIR`, `MODEL_DIR`), model hyperparameters (`MODEL_NAME`, `WINDOW_SIZE=224×224`, `STRIDE=56`), the 15 MVTec AD object categories, and prompt engineering templates (7 normal states × 8 templates + 10 defect states × 8 templates = 136 prompts per category). |
| `models/winclip.py` | **Core model implementation**. The `WinCLIP` class loads the CLIP ViT-B/32 model from HuggingFace, provides `encode_text()` for prompt ensemble generation, `extract_windows()` for sliding-window patch extraction, and `predict()` that orchestrates the full pipeline: text encoding → patch extraction → batch inference → heatmap aggregation → Gaussian smoothing → score normalization. |
| `models/architecture.py` | Model architecture definitions (supplementary to the main WinCLIP implementation). |
| `models/predict.py` | Standalone prediction utility functions. |
| `models/train.py` | Training utilities (reserved; the zero-shot approach does not require training). |
| `data/loader.py` | The `DataLoader` class provides `load_image()` for robust image loading with RGB conversion and `get_sample_image()` to randomly select test images from the MVTec dataset (prioritizing defect subfolders over "good" for demonstration purposes). |
| `data/preprocessing.py` | Image preprocessing utilities for data pipeline operations. |
| `utils/visualization.py` | The `Visualizer` class provides `save_results()` which takes a PIL image and heatmap array, applies a JET colormap via OpenCV, blends it as a 40% overlay on the original image, and saves the result to `app/static/results/`. |
| `utils/logger.py` | Logging utility for structured application logging. |
| `evaluate.py` | **Benchmark evaluation script**. Iterates through all test images in a given MVTec category (both "good" and defect subfolders), runs WinCLIP inference on each, collects true labels and predicted anomaly scores, then computes AU-ROC and optimal F1-Score using scikit-learn. Runnable from CLI: `python src/evaluate.py <category>`. |
| `create_model_comparison.py` | Runs `evaluate.py` across all available MVTec categories (sampling 10 images per subfolder for speed), computes averaged metrics, and saves a comprehensive JSON comparison against 4 state-of-the-art models (AnomalyCLIP, AdaptCLIP, AnomalyGPT, FiLo) to `data/processed/model_comparison.json`. |

---

### `data/` — Data Storage

| Path | Purpose |
|---|---|
| `raw/Datasets/archive/` | Extracted MVTec AD dataset. Contains 15 category folders, each with `train/` (normal images) and `test/` (normal + various defect subfolders) splits, plus `ground_truth/` pixel-level masks. |
| `raw/Datasets/archive.zip` | Original MVTec AD dataset archive (~5 GB compressed). |
| `processed/model_comparison.json` | Auto-generated JSON file containing detailed comparative analysis of WinCLIP against 4 research models, including architecture, strengths, limitations, and benchmark scores. |
| `processed/Comparison_Matrix.csv` | CSV evaluation comparison matrix covering 10 criteria (hardware, training data, deployment time, cost, AU-ROC, etc.) across all 5 models. |

---

### `tests/` — Automated Test Suite

| File | Purpose |
|---|---|
| `test_model.py` | **Unit tests** (using `unittest`): validates that the WinCLIP model loads successfully (`model` and `processor` are not `None`) and that text encoding produces the expected shape `[2, feature_dim]`. |
| `test_inference.py` | **Integration test**: loads a sample image from the MVTec dataset (falls back to a synthetic 224×224 red image), runs the full `predict()` pipeline, and verifies that a valid heatmap and anomaly score are produced. |

---

### `docs/presentations/` — Documentation

| File | Purpose |
|---|---|
| `presentation.tex` | Initial project presentation slide deck written in LaTeX (Beamer). |
| `presentation.pdf` | Compiled PDF of the initial presentation (~480 KB). |
| `final_review.tex` | Final review presentation slide deck in LaTeX (Beamer). |
| `final_review.pdf` | Compiled PDF of the final review presentation (~1.2 MB). |

---

## Setup & Installation
1. **Clone / Download the project** and navigate to the `Implementation/` directory.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Linux/macOS
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: The first run downloads the weights for the CLIP model (~300MB) from HuggingFace.*

4. **Download the MVTec AD dataset** (if not already present):
   - Place the extracted dataset at `data/raw/Datasets/archive/`
   - Each category folder should contain `train/`, `test/`, and `ground_truth/` subdirectories.

5. **Run the Interactive Dashboard**:
   Double-click `run_dashboard.bat` OR run:
   ```bash
   python run.py
   ```
   Open [http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard) in your browser.

---

## Dashboard Usage
1. Select the **Object Category** (e.g., Bottle, Cable) from the dropdown.
2. Choose an input method:
   - **Upload Image**: Select an image file from the `data/raw/Datasets/archive/.../test/` directory.
   - **Live Camera**: Grant webcam access, then capture a snapshot. The system automatically removes the background using `rembg` and places the object on a solid black background for MVTec-compatible analysis.
3. Click **Analyze** (or **Capture & Analyze Snapshot** for camera mode).
4. View the resulting:
   - **Status** — "Normal" or "Defect Detected" (threshold: anomaly score > 0.4)
   - **Anomaly Score** — Raw maximum anomaly score
   - **Heatmap** — JET colormap overlay highlighting suspected defect regions

---

## Testing
To run the automated unit tests, double-click `run_tests.bat` OR run:
```bash
python tests/test_model.py
python tests/test_inference.py
```

---

## Evaluation & Benchmarking
To evaluate WinCLIP on a specific MVTec AD category:
```bash
python src/evaluate.py bottle
```
Or double-click `run_evaluation.bat` and enter the category name when prompted.

To regenerate the full comparative model benchmark:
```bash
python src/create_model_comparison.py
```
This runs evaluation across all 15 categories and outputs results to `data/processed/model_comparison.json`.
