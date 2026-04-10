# Zero-Shot Defect Detection

## Project Overview
This project implements a **Zero-Shot Industrial Anomaly Detection** system using **WinCLIP (Window-based CLIP)**. The core objective is to detect and precisely localize defects on manufactured components without requiring *any* prior training on defect data. It includes a Flask-based interactive web dashboard for real-time visual analysis, as well as a cloud-ready deployment architecture (Gradio on Hugging Face Spaces + GitHub Pages Frontend).

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
- [Setup & Installation (Local)](#setup--installation-local)
- [Cloud Deployment (Hugging Face + GitHub Pages)](#cloud-deployment-hugging-face--github-pages)
- [Dashboard Usage](#dashboard-usage)
- [Testing](#testing)
- [Evaluation & Benchmarking](#evaluation--benchmarking)

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
| **Local Web Framework**| Flask | Local backend API and server-side rendering for the web dashboard |
| **Local Frontend**| Bootstrap 5, Jinja2 Templates | Responsive local dashboard UI |
| **Cloud Backend API**| Gradio, Hugging Face Spaces | Cloud hosting API for the ML inference pipeline |
| **Cloud Frontend** | GitHub Pages, `@gradio/client`, Vanilla CSS/JS | Static decoupled frontend for robust production inference calls |
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
├── app/                        # Local Flask web application (frontend + API)
│   ├── __init__.py             # App factory
│   ├── routes.py               # Route definitions
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # Static web assets
│
├── frontend/                   # Cloud-ready Standalone HTML/JS Frontend (for GitHub Pages)
│   └── index.html              # Frontend utilizing @gradio/client for Hugging Face API interactions
│
├── huggingface/                # Gradio-based Headless Backend (for Hugging Face Spaces)
│   ├── app.py                  # Gradio API application wrapping WinCLIP
│   ├── requirements.txt        # Hugging Face deployment dependencies
│   └── README.md               # HF Space configuration file
│
├── src/                        # Core ML pipeline source code
│   ├── config.py               # Central configuration
│   ├── evaluate.py             # Benchmark evaluation script
│   ├── create_model_comparison.py  
│   ├── models/                 # ML model implementations (WinCLIP, architecture, predict)
│   ├── data/                   # Data handling and preprocessing routines
│   └── utils/                  # Shared utilities (heatmap generation, logging)
│
├── data/                       # Data storage (MVTec AD)
│   ├── raw/                    # Raw datasets
│   └── processed/              # Processed model benchmark data JSON/CSV
│
├── models/                     # Model artifacts directory (downloaded weights cache)
│
├── notebooks/                  # Jupyter notebooks for exploration and prototyping
│
├── docs/                       # Documentation and presentations
│
└── tests/                      # Automated test suite (unit and integration)
```

---

## File-Level Description

### Root Files

| File | Purpose |
|---|---|
| `run.py` | Application entry point. Imports the Flask app factory from `app/__init__.py` and starts the development server on `http://127.0.0.1:5000`. |
| `run_dashboard.bat` | Convenience Windows batch script that sets `FLASK_APP` and `FLASK_ENV` environment variables and launches the local Flask dashboard. |
| `run_evaluation.bat` | Interactive batch script that prompts the user for a category name (e.g., `bottle`) and runs `src/evaluate.py` to benchmark that category. |
| `run_tests.bat` | Batch script that sequentially executes `test_model.py` and `test_inference.py` from the `tests/` directory. |
| `requirements.txt` | Lists all Python dependencies required by the project (Flask, PyTorch, Transformers, OpenCV, rembg, etc.). |

---

### `frontend/` — Standalone Cloud Frontend

| File | Purpose |
|---|---|
| `index.html` | A static HTML/JS frontend engineered for GitHub Pages deployment. It uses the `@gradio/client` module to seamlessly connect to the remote WinCLIP backend hosted on Hugging Face Spaces, offering a fully decoupled production web UI. |

---

### `huggingface/` — Cloud Backend API

| File | Purpose |
|---|---|
| `app.py` | A Gradio application that adapts the WinCLIP model for cloud deployment. It wraps the core prediction logic to expose inference as an API endpoint, connecting cleanly with `@gradio/client`. |
| `README.md` | Contains Hugging Face Space metadata (SDK version, hardware config) required for deployment. |
| `requirements.txt` | Dependency list explicitly filtered for the Hugging Face Space cloud environment. |

---

### `app/` — Local Flask Web Application

| File | Purpose |
|---|---|
| `__init__.py` | **App factory function** (`create_app()`). Creates a Flask instance, loads configuration from `src.config.Config`. |
| `routes.py` | Defines Flask routes. Lazy-loads the WinCLIP model locally on the first request. |
| `templates/...` | Jinja2 templates (`base.html`, `index.html`, `dashboard.html`) for the local UI interface. |
| `static/...` | Custom CSS/JS and temporary save locations for images during local operations. |

---

### `src/` — Core ML Pipeline

| File | Purpose |
|---|---|
| `config.py` | **Central configuration class** defining paths, hyperparameters (`MODEL_NAME`, `WINDOW_SIZE=224×224`), and prompt templates (136 total). |
| `models/winclip.py` | **Core model implementation**. The `WinCLIP` class loads the CLIP model, extracts sliding window patches, and aggregates predictions into heatmaps. |
| `evaluate.py` | **Benchmark evaluation script**. Computes AU-ROC and optimal F1-Score against the datasets. |
| `create_model_comparison.py` | Auto-generates detailed JSON comparative analysis against academic models. |

*(See the `src/utils` and `src/data` directories for logging, visualization, and preprocessing pipelines).*

---

### `data/` & `tests/`

- **`data/`:** Contains the raw downloaded MVTec dataset (`data/raw/`) and benchmarking tables (`data/processed/Comparison_Matrix.csv`).
- **`tests/`:** Validation suite enforcing model loading sanity (`test_model.py`) and an end-to-end local inference check (`test_inference.py`).

---

## Setup & Installation (Local)
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

5. **Run the Interactive Local Dashboard**:
   Double-click `run_dashboard.bat` OR run:
   ```bash
   python run.py
   ```
   Open [http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard) in your browser.

---

## Cloud Deployment (Hugging Face + GitHub Pages)
We have implemented a decoupled architecture resolving performance issues by separating the frontend from the ML execution backend:

1. **Backend (Hugging Face / Gradio):** 
   - Deploy the contents of the `huggingface/` directory to a Hugging Face Space running Gradio.
   - This sets up your dedicated, robust inference API.
2. **Frontend (GitHub Pages):**
   - Host `frontend/index.html` on GitHub Pages (or any static host). 
   - Uses the `@gradio/client` to execute non-blocking, asynchronous prediction calls to your Hugging Face Space endpoint.

---

## Dashboard Usage
*(Applies fundamentally to both the local and cloud interfaces)*
1. Select the **Object Category** (e.g., Bottle, Cable) from the dropdown.
2. Choose an input method:
   - **Upload Image**: Select an image file.
   - **Live Camera** *(supported locally)*: Grant webcam access and capture a snapshot (supports `rembg` background isolation).
3. Click **Analyze**.
4. View the resulting:
   - **Status** — "Normal" or "Defect Detected" (threshold logic)
   - **Anomaly Score** — Raw confidence probability
   - **Heatmap** — Overlay isolating suspected defect zones.

---

## Testing
To run the automated unit tests locally:
```bash
python tests/test_model.py
python tests/test_inference.py
```
*(Or double-click `run_tests.bat`)*

---

## Evaluation & Benchmarking
To evaluate WinCLIP on a specific MVTec AD category:
```bash
python src/evaluate.py bottle
```

To regenerate the full comparative model benchmark:
```bash
python src/create_model_comparison.py
```
This runs evaluation across all 15 categories and outputs results to `data/processed/model_comparison.json`.
