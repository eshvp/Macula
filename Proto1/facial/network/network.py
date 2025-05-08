from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import socket
from socket import SOCK_DGRAM
from pathlib import Path
import cv2
import numpy as np
from io import BytesIO

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from facial.utility.recognition import get_recognition_model

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Initialize face recognition model
recognizer = get_recognition_model()

def get_local_ip():
    """Get the local IP address for network access"""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/status')
def status():
    """Check if server is running"""
    return jsonify({
        'status': 'running',
        'recognizer': recognizer.__class__.__name__
    })

@app.route('/api/recognize', methods=['POST'])
def recognize():
    """Handle face recognition requests"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Convert uploaded file to OpenCV format
        image_file = request.files['image']
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
            
        # Process the image and get recognition results
        result = recognizer.recognize_face(img)
        return jsonify(result)
        
    except Exception as e:
        print(f"Recognition error: {e}")  # For debugging
        return jsonify({'error': str(e)}), 500

@app.route('/api/capture', methods=['POST'])
def capture():
    """Handle face capture requests for registration"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    if 'firstName' not in request.form or 'lastName' not in request.form:
        return jsonify({'error': 'Name information missing'}), 400
    
    try:
        image_file = request.files['image']
        first_name = request.form['firstName']
        last_name = request.form['lastName']
        
        # Save user data and process face
        result = recognizer.register_user(first_name, last_name, image_file)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get list of registered users"""
    try:
        users = recognizer.get_registered_users()
        return jsonify({'users': users})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files (images, CSS, JavaScript)"""
    return send_from_directory('static', path)

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

def create_static_folder():
    """Create static folder if it doesn't exist"""
    static_folder = Path(__file__).parent / 'static'
    static_folder.mkdir(exist_ok=True)
    
    # Create default index.html if it doesn't exist
    index_path = static_folder / 'index.html'
    if not index_path.exists():
        with open(index_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        video { width: 100%; max-width: 640px; }
        button { margin: 10px; padding: 8px 16px; }
        #result { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Face Recognition System</h1>
    <div id="camera">
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas" style="display:none;"></canvas>
    </div>
    <div id="controls">
        <button id="startBtn">Start Camera</button>
        <button id="captureBtn" disabled>Capture</button>
        <button id="recognizeBtn" disabled>Recognize</button>
    </div>
    <div id="result"></div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const startBtn = document.getElementById('startBtn');
        const captureBtn = document.getElementById('captureBtn');
        const recognizeBtn = document.getElementById('recognizeBtn');
        const resultDiv = document.getElementById('result');

        startBtn.onclick = startCamera;
        captureBtn.onclick = captureImage;
        recognizeBtn.onclick = recognizeFace;

        function startCamera() {
            navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'user' } 
            })
            .then(stream => {
                video.srcObject = stream;
                startBtn.disabled = true;
                captureBtn.disabled = false;
                recognizeBtn.disabled = false;
            })
            .catch(err => {
                resultDiv.innerHTML = `Error: ${err.message}`;
            });
        }

        function captureImage() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('image', blob);
                
                fetch('/api/capture', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    resultDiv.innerHTML = JSON.stringify(data, null, 2);
                })
                .catch(err => {
                    resultDiv.innerHTML = `Error: ${err.message}`;
                });
            }, 'image/jpeg');
        }

        function recognizeFace() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('image', blob);
                
                fetch('/api/recognize', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    resultDiv.innerHTML = JSON.stringify(data, null, 2);
                })
                .catch(err => {
                    resultDiv.innerHTML = `Error: ${err.message}`;
                });
            }, 'image/jpeg');
        }
    </script>
</body>
</html>
            """)

if __name__ == '__main__':
    # Create static folder and default files
    create_static_folder()
    
    # Force specific IP and port
    host = '172.18.4.60'  # Hardcoded IP address
    port = 5000
    
    print(f"\nStarting Face Recognition Server")
    print(f"=" * 40)
    print(f"Attempting to bind to: http://{host}:{port}")
    
    try:
        app.run(host=host, port=port, debug=True)
    except OSError as e:
        print(f"\nError: Could not bind to {host}:{port}")
        print("Falling back to all interfaces (0.0.0.0)")
        print(f"Error details: {e}")
        # Fallback to all interfaces if specific IP fails
        app.run(host='0.0.0.0', port=port, debug=True)

