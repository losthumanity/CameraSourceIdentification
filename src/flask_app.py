"""
Flask Web Application for Camera Source Identification
Demo UI for video upload and verification
Implementation from technical specification
Author: Pranav Patil | Sponsored by PiLabs
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
import json
import torch

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from camera_pipeline import CameraIdentificationPipeline
from forgery_detector import ForgeryDetector
from prnu_extractor import PRNUExtractor

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
pipeline = None
forgery_detector = None
model_loaded = False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    """Load trained models and reference patterns"""
    global pipeline, forgery_detector, model_loaded
    
    try:
        model_dir = './camera_model_data'
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'prnu_dataset', 'dataset_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Initialize pipeline
        pipeline = CameraIdentificationPipeline(num_classes=metadata['num_classes'])
        pipeline.class_names = metadata['class_names']
        
        # Load trained model
        model_path = os.path.join(model_dir, 'best_prnu_classifier.pth')
        pipeline.model.load_state_dict(torch.load(model_path, map_location=pipeline.device))
        pipeline.model.eval()
        
        # Load reference patterns for forgery detection
        ref_path = os.path.join(model_dir, 'prnu_dataset', 'reference_patterns.npz')
        forgery_detector = ForgeryDetector(reference_patterns_path=ref_path)
        
        model_loaded = True
        print("✅ Models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Please train the model first using main_train.py")
        return False


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and analysis"""
    if not model_loaded:
        return jsonify({'error': 'Models not loaded. Please train the model first.'}), 500
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: MP4, AVI, MOV, MKV, WEBM'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze video
        result = pipeline.predict_video_source(filepath)
        
        # Clean up uploaded file (optional)
        # os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': result['predicted_camera'],
            'confidence': round(result['confidence'] * 100, 2),
            'probabilities': {k: round(v * 100, 2) for k, v in result['all_probabilities'].items()}
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500


@app.route('/api/detect_forgery', methods=['POST'])
def detect_forgery():
    """Detect if video is forged/deepfake"""
    if not model_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    expected_camera = request.form.get('expected_camera', None)
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect forgery
        if expected_camera:
            result = forgery_detector.detect_forgery(filepath, expected_camera)
        else:
            result = forgery_detector.analyze_video_authenticity(filepath)
        
        # Clean up
        # os.remove(filepath)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Error detecting forgery: {str(e)}'}), 500


@app.route('/api/models/info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    if not model_loaded:
        return jsonify({
            'loaded': False,
            'message': 'Models not loaded. Please train the model first.'
        })
    
    return jsonify({
        'loaded': True,
        'num_classes': len(pipeline.class_names),
        'camera_models': pipeline.class_names,
        'device': str(pipeline.device),
        'reference_patterns': len(forgery_detector.prnu_extractor.reference_patterns)
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# HTML Template (embedded)
TEMPLATE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camera Source Identification System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 1.1em; }
        .content { padding: 40px; }
        .upload-section {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s;
        }
        .upload-section:hover { border-color: #764ba2; background: #f8f9ff; }
        .file-input {
            display: none;
        }
        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            border-radius: 50px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .upload-btn:hover { transform: scale(1.05); }
        .results {
            display: none;
            margin-top: 30px;
            padding: 30px;
            background: #f8f9ff;
            border-radius: 15px;
        }
        .result-item {
            padding: 15px;
            margin: 10px 0;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .progress-bar {
            width: 100%;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error { color: #e74c3c; padding: 20px; text-align: center; }
        .success { color: #27ae60; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📸 Camera Source Identification</h1>
            <p>PRNU-Based Forensic Video Analysis System</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Sponsored by PiLabs | Powered by Deep Learning</p>
        </div>
        
        <div class="content">
            <div class="upload-section">
                <h2>🎬 Upload Video for Analysis</h2>
                <p style="margin: 20px 0; color: #666;">
                    Supported formats: MP4, AVI, MOV, MKV, WEBM (Max: 100MB)
                </p>
                <input type="file" id="videoInput" class="file-input" accept="video/*">
                <button class="upload-btn" onclick="document.getElementById('videoInput').click()">
                    Choose Video File
                </button>
                <p id="fileName" style="margin-top: 15px; color: #667eea; font-weight: bold;"></p>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px;">Analyzing video...</p>
            </div>
            
            <div class="results" id="results">
                <h2>📊 Analysis Results</h2>
                <div class="result-item">
                    <h3>🎯 Predicted Camera Model</h3>
                    <h2 id="predictedCamera" style="color: #667eea; margin: 10px 0;">-</h2>
                </div>
                <div class="result-item">
                    <h3>🎲 Confidence Score</h3>
                    <h2 id="confidence" style="color: #27ae60; margin: 10px 0;">-</h2>
                    <div class="progress-bar">
                        <div class="progress-fill" id="confidenceBar" style="width: 0%"></div>
                    </div>
                </div>
                <div class="result-item">
                    <h3>📈 All Probabilities</h3>
                    <div id="allProbabilities"></div>
                </div>
            </div>
            
            <div id="error" class="error"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('videoInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('fileName').textContent = `Selected: ${file.name}`;
                uploadVideo(file);
            }
        });
        
        function uploadVideo(file) {
            const formData = new FormData();
            formData.append('video', file);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').textContent = '';
            
            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    document.getElementById('error').textContent = `Error: ${data.error}`;
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('error').textContent = `Error: ${error.message}`;
            });
        }
        
        function displayResults(data) {
            document.getElementById('results').style.display = 'block';
            document.getElementById('predictedCamera').textContent = data.prediction;
            document.getElementById('confidence').textContent = `${data.confidence}%`;
            document.getElementById('confidenceBar').style.width = `${data.confidence}%`;
            
            const probsHtml = Object.entries(data.probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([camera, prob]) => `
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>${camera}</span>
                            <span style="font-weight: bold;">${prob}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${prob}%"></div>
                        </div>
                    </div>
                `).join('');
            
            document.getElementById('allProbabilities').innerHTML = probsHtml;
        }
    </script>
</body>
</html>"""


# Create templates directory and save template
def create_template():
    """Create HTML template file"""
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(TEMPLATE_HTML)


if __name__ == '__main__':
    print("=" * 60)
    print("Camera Source Identification - Flask Demo")
    print("Sponsored by PiLabs")
    print("=" * 60)
    
    # Create template
    create_template()
    
    # Load models
    if not load_models():
        print("\n⚠️  Warning: Models not loaded!")
        print("Please train the model first by running:")
        print("  python src/main_train.py")
        print("\nStarting server anyway for development...")
    
    print("\n🚀 Starting Flask server...")
    print("📍 Open browser at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
