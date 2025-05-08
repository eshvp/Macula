import http.server
import socketserver
import json
import os
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to Python path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Now we can import from utility
from facial.utility.recognition import get_recognition_model

# Initialize face recognition model
recognizer = get_recognition_model()

class FaceRecognitionHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/recognize':
            content_length = int(self.headers['Content-Length'])
            image_data = self.rfile.read(content_length)
            
            # Convert image data to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process image
            result = recognizer.recognize_face(img)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            return

        # Default to parent class handling
        super().do_POST()
        
    def do_GET(self):
        """Handle GET requests"""
        try:
            # Serve index.html for root path
            if self.path == '/':
                self.path = '/index.html'
            
            # Get file path
            file_path = Path(self.directory) / self.path.lstrip('/')
            
            # Check if file exists
            if not file_path.exists():
                self.send_error(404, "File not found")
                return

            # Set content type
            if self.path.endswith('.html'):
                content_type = 'text/html'
            elif self.path.endswith('.js'):
                content_type = 'application/javascript'
            elif self.path.endswith('.css'):
                content_type = 'text/css'
            else:
                content_type = 'application/octet-stream'
                
            # Send headers
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Send file content
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
                
        except Exception as e:
            print(f"Error serving {self.path}: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server(port=8000):
    """Run the server"""
    # Change to the static directory
    static_dir = Path(__file__).parent / 'static'
    os.chdir(static_dir)
    
    print(f"\nStarting Simple HTTP Server")
    print("=" * 40)
    
    # Try different ports
    while port < 8100:
        try:
            with socketserver.TCPServer(("", port), FaceRecognitionHandler) as httpd:
                print(f"\nServing at:")
                print(f"http://localhost:{port}")
                print(f"http://127.0.0.1:{port}")
                try:
                    # Try to get local IP
                    import socket
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(('8.8.8.8', 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                    print(f"http://{local_ip}:{port}")
                except:
                    pass
                print("\nPress Ctrl+C to stop the server")
                print("=" * 40)
                httpd.serve_forever()
                break
        except OSError:
            print(f"Port {port} is in use, trying next port...")
            port += 1
        except KeyboardInterrupt:
            print("\nServer stopped by user")
            break

if __name__ == '__main__':
    run_server()