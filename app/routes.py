from flask import Blueprint, render_template, request, jsonify, url_for
import os
import base64
import io
from werkzeug.utils import secure_filename
from src.models.winclip import WinCLIP
from src.utils.visualization import Visualizer
from src.config import Config
from PIL import Image
import torch

main = Blueprint('main', __name__)

# Lazy load model to avoid circular imports during app creation
model = None

def get_model():
    global model
    if model is None:
        model = WinCLIP()
    return model

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', objects=Config.OBJECTS)

@main.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    category = request.form.get('category', 'object')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(Config.BASE_DIR, 'app', 'static', 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        file_path = os.path.join(upload_path, filename)
        file.save(file_path)
        
        # Run Inference
        try:
            winclip = get_model()
            image = Image.open(file_path).convert("RGB")
            
            # --- CPU Optimization: Downsample Huge Uploads ---
            image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            heatmap, max_score = winclip.predict(image, category)
            
            # Visualize
            result_filename = f"result_{filename}"
            result_path = Visualizer.save_results(image, heatmap, max_score, category, result_filename)
            
            # Determine status based on max score
            # Threshold is tunable, WinCLIP paper suggests varying thresholds. 
            # 0.5 is a reasonable starting point for normalized scores.
            status = 'Defect Detected' if max_score > 0.4 else 'Normal'
            
            return jsonify({
                'score': float(max_score),
                'result_image': url_for('static', filename=result_path),
                'status': status
            })
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

@main.route('/analyze_live', methods=['POST'])
def analyze_live():
    data = request.json
    if not data or 'image_base64' not in data:
        return jsonify({'error': 'No image data provided'}), 400
        
    category = data.get('category', 'object')
    image_b64 = data['image_base64']
    
    # Strip the data:image/jpeg;base64, header if it exists
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
        
    try:
        # Decode base64 to PIL Image
        image_data = base64.b64decode(image_b64)
        original_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # --- CPU Optimization: Downsample Huge Webcam Frames ---
        # A 1080p webcam produces ~350 sliding windows, which takes 60s+ on a CPU.
        # Resizing the longest edge to 400px drops it back to ~5 seconds.
        original_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
        
        # --- AI Foreground Extraction ---
        # We use rembg to perfectly isolate the object from the user's room
        from rembg import remove
        extracted_rgba = remove(original_image)
        
        # MVTec Dataset uses solid backgrounds. We create a generic black background.
        image = Image.new("RGB", original_image.size, (0, 0, 0))
        # Paste the extracted object (using its alpha channel as the cookie-cutter mask)
        image.paste(extracted_rgba, mask=extracted_rgba.split()[3])
        
        # We need to save the processed (black bg) image temporarily so Visualizer can use it
        import time
        filename = f"live_capture_{int(time.time())}.jpg"
        upload_path = os.path.join(Config.BASE_DIR, 'app', 'static', 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        file_path = os.path.join(upload_path, filename)
        image.save(file_path)
        
        # Run Inference
        winclip = get_model()
        heatmap, max_score = winclip.predict(image, category)
        
        # Visualize
        result_filename = f"result_{filename}"
        result_path = Visualizer.save_results(image, heatmap, max_score, category, result_filename)
        
        status = 'Defect Detected' if max_score > 0.4 else 'Normal'
        
        return jsonify({
            'score': float(max_score),
            'result_image': url_for('static', filename=result_path),
            'status': status
        })
    except Exception as e:
        print(f"Error during live analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
