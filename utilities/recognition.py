import cv2
import dlib
import numpy as np
import os
import time
import threading
from queue import Queue
from stored_data.UserProfiles import UserProfiles

class FaceRecognition:
    def __init__(self):
        # Enable dlib CUDA acceleration if available
        if dlib.DLIB_USE_CUDA:
            print("CUDA acceleration enabled for dlib")
        
        # Use frontal face detector with default parameters
        self.detector = dlib.get_frontal_face_detector()
        
        # Load models with proper error handling
        try:
            model_path = os.path.join("models", "dlib_face_recognition_resnet_model_v1.dat")
            shape_path = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
            
            if not os.path.exists(model_path) or not os.path.exists(shape_path):
                raise FileNotFoundError(f"Required model files not found. Please check: {model_path} and {shape_path}")
                
            self.face_recognizer = dlib.face_recognition_model_v1(model_path)
            self.shape_predictor = dlib.shape_predictor(shape_path)
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
        
        # Load user profiles
        self.user_profiles = UserProfiles()
        self.threshold = 0.6
        
        # Pre-load all user encodings
        self.reference_encodings = {}
        self._load_user_encodings()
        
        # Cache for recent recognitions to avoid redundant processing
        self.recognition_cache = {}
        self.cache_lifetime = 2.0  # seconds
        
        # Track the last detected face for faster tracking
        self.last_face_rect = None
        self.face_tracking_frames = 0
        self.max_tracking_frames = 30
        
    def _load_user_encodings(self):
        """Pre-load all user face encodings"""
        print("Pre-loading user face encodings...")
        start_time = time.time()
        users = self.user_profiles.get_all_users()
        loaded_count = 0
        
        for user_id, user_data in users.items():
            if 'embedding' in user_data:
                # If embedding is already saved in the user profile
                self.reference_encodings[user_id] = {
                    'encoding': np.array(user_data['embedding']),
                    'first_name': user_data['first_name'],
                    'last_name': user_data['last_name']
                }
                loaded_count += 1
            elif 'image_path' in user_data and os.path.exists(user_data['image_path']):
                # Load image and compute encoding
                try:
                    stored_image = cv2.imread(user_data['image_path'])
                    if stored_image is not None:
                        stored_rgb = cv2.cvtColor(stored_image, cv2.COLOR_BGR2RGB)
                        encoding = self.get_face_encoding(stored_rgb)
                        
                        if encoding is not None:
                            self.reference_encodings[user_id] = {
                                'encoding': encoding,
                                'first_name': user_data['first_name'],
                                'last_name': user_data['last_name']
                            }
                            
                            # Save encoding back to user profile for future use
                            user_data['embedding'] = encoding.tolist()
                            self.user_profiles.add_user(user_id, user_data)
                            loaded_count += 1
                except Exception as e:
                    print(f"Error loading image for user {user_id}: {e}")
        
        print(f"Loaded {loaded_count} user encodings in {time.time() - start_time:.2f} seconds")
                
    def get_face_encoding(self, image):
        """Extract face encoding from image"""
        try:
            faces = self.detector(image)
            if len(faces) == 0:
                return None
            
            face = faces[0]
            shape = self.shape_predictor(image, face)
            face_encoding = np.array(self.face_recognizer.compute_face_descriptor(image, shape))
            return face_encoding
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None
    
    def compare_faces(self, known_encoding, unknown_encoding):
        """Compare two face encodings and return distance"""
        if known_encoding is None or unknown_encoding is None:
            return 1.0
        
        return np.linalg.norm(known_encoding - unknown_encoding)

    def recognize_face(self, encoding, face_id=None):
        """Recognize face using pre-loaded encodings"""
        if encoding is None:
            return None
        
        # Check cache for this face_id if provided
        current_time = time.time()
        if face_id is not None and face_id in self.recognition_cache:
            cache_entry = self.recognition_cache[face_id]
            if current_time - cache_entry['time'] < self.cache_lifetime:
                # Cache hit - return cached result
                return cache_entry['result']
            
        # No cache hit, perform recognition
        best_match = None
        min_distance = float('inf')
        
        # Use numpy vectorization for faster comparison when possible
        if len(self.reference_encodings) > 0:
            # Create a single array of all encodings for vectorized comparison
            user_ids = list(self.reference_encodings.keys())
            encoding_array = np.array([self.reference_encodings[uid]['encoding'] for uid in user_ids])
            
            # Calculate distances to all encodings at once
            distances = np.linalg.norm(encoding_array - encoding, axis=1)
            
            # Find the minimum distance
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            if min_distance < self.threshold:
                user_id = user_ids[min_idx]
                user_data = self.reference_encodings[user_id]
                best_match = {
                    'first_name': user_data['first_name'],
                    'last_name': user_data['last_name'],
                    'user_id': user_id,
                    'distance': min_distance,
                    'confidence': 1.0 - (min_distance / self.threshold)
                }
                
                # Cache the result
                if face_id is not None:
                    self.recognition_cache[face_id] = {
                        'time': current_time,
                        'result': best_match
                    }
        
        return best_match

    def detect_and_recognize(self, frame, fast_mode=True):
        """
        Detect and recognize faces in frame
        Returns: List of dicts with face info
        """
        h, w = frame.shape[:2]
        results = []
        
        # Use fast mode with tracking optimizations
        if fast_mode and self.last_face_rect is not None and self.face_tracking_frames < self.max_tracking_frames:
            # Expand search region by 20%
            x, y, x2, y2 = self.last_face_rect
            expansion = 0.2
            search_left = max(0, int(x - w * expansion))
            search_top = max(0, int(y - h * expansion))
            search_right = min(w, int(x2 + w * expansion))
            search_bottom = min(h, int(y2 + h * expansion))
            
            # Extract ROI
            roi = frame[search_top:search_bottom, search_left:search_right]
            if roi.size > 0:
                # Detect in ROI
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                faces = self.detector(roi_gray)
                
                # If found in ROI, adjust coordinates
                if len(faces) > 0:
                    for face in faces:
                        x = face.left() + search_left
                        y = face.top() + search_top
                        w = face.width()
                        h = face.height()
                        right = x + w
                        bottom = y + h
                        
                        # Update tracking
                        self.last_face_rect = (x, y, right, bottom)
                        self.face_tracking_frames = 0
                        
                        # Extract face and create result
                        face_img = frame[y:bottom, x:right]
                        if face_img.size > 0:
                            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            face_encoding = self.get_face_encoding(face_rgb)
                            
                            # Generate unique face ID
                            face_id = f"face_{x}_{y}_{w}_{h}"
                            
                            # Get recognition results
                            match = self.recognize_face(face_encoding, face_id)
                            
                            results.append({
                                'rect': (x, y, right, bottom),
                                'encoding': face_encoding,
                                'match': match,
                                'face_id': face_id
                            })
                    
                    # Exit if faces found in ROI
                    if results:
                        return results
        
        # Fall back to full frame detection if ROI failed or not using fast mode
        self.face_tracking_frames += 1
        
        # Convert to gray for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.width()
            h = face.height()
            right = x + w
            bottom = y + h
            
            # Update tracking for next frame
            self.last_face_rect = (x, y, right, bottom)
            self.face_tracking_frames = 0
            
            # Extract face region
            face_img = frame[y:bottom, x:right]
            if face_img.size == 0:
                continue
                
            # Generate unique face ID
            face_id = f"face_{x}_{y}_{w}_{h}"
            
            # Get encoding and recognition
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_encoding = self.get_face_encoding(rgb_face)
            match = self.recognize_face(face_encoding, face_id)
            
            results.append({
                'rect': (x, y, right, bottom),
                'encoding': face_encoding,
                'match': match,
                'face_id': face_id
            })
        
        return results

def run_recognition():
    """Main function to run face recognition"""
    print("Initializing face recognition...")
    start_time = time.time()
    
    try:
        face_recognition = FaceRecognition()
        print(f"Initialization complete in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Failed to initialize face recognition: {e}")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # For FPS calculation
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    # Multi-threading setup
    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=2)
    stop_event = threading.Event()
    
    # Skip frames for better performance
    process_every_n_frames = 4  # Higher value = better performance but more lag
    frame_counter = 0
    
    # For display
    display_frame = None
    recognition_results = []
    resolution = (1280, 720)  # Default resolution
    
    def process_frames():
        """Background thread for face recognition"""
        while not stop_event.is_set():
            try:
                # Get frame from queue with timeout
                frame = frame_queue.get(timeout=0.1)
                
                # Process frame
                results = face_recognition.detect_and_recognize(frame, fast_mode=True)
                
                # Store results in queue
                if not result_queue.full():
                    result_queue.put((frame, results))
            except Exception as e:
                if not isinstance(e, TimeoutError) and not stop_event.is_set():
                    print(f"Error in processing thread: {e}")
    
    # Start processing thread
    processor = threading.Thread(target=process_frames)
    processor.daemon = True
    processor.start()
    
    print("Face recognition started. Press 'q' to quit, '+'/'-' to adjust processing rate.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
                
            # Get frame resolution if not set yet
            if resolution[0] == 0:
                h, w = frame.shape[:2]
                resolution = (w, h)
            
            frame_counter += 1
            
            # Queue frame for processing at specified intervals
            if frame_counter % process_every_n_frames == 0:
                # Replace any existing frame in the queue
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                
                # Add new frame to queue
                try:
                    frame_queue.put_nowait(frame.copy())
                except:
                    pass  # Queue is full, will try next frame
            
            # Get latest results if available
            try:
                if not result_queue.empty():
                    last_processed_frame, recognition_results = result_queue.get_nowait()
                    display_frame = last_processed_frame.copy()
            except:
                pass
                
            # Create display frame
            if display_frame is None:
                display_frame = frame.copy()
            else:
                # Use the latest frame but keep recognition results from processed frame
                display_frame = frame.copy()
            
            # Draw results on the display frame
            for face_data in recognition_results:
                x, y, right, bottom = face_data['rect']
                match = face_data['match']
                
                # Draw rectangle around detected face
                cv2.rectangle(display_frame, (x, y), (right, bottom), (0, 255, 0), 2)
                
                if match:
                    confidence = match['confidence'] * 100
                    name = f"{match['first_name']} {match['last_name']}"
                    conf_text = f"{confidence:.1f}%"
                    
                    # Color gradient from red (0%) to green (100%)
                    color = (
                        0,
                        int(255 * (confidence / 100)),
                        int(255 * (1 - confidence / 100))
                    )
                    
                    cv2.putText(display_frame, f"{name} ({conf_text})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                else:
                    cv2.putText(display_frame, "Unknown", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()
                
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
            # Show processing speed info
            cv2.putText(display_frame, f"Processing: 1/{process_every_n_frames} frames", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Face Recognition', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') and process_every_n_frames > 1:
                process_every_n_frames -= 1
                print(f"Processing 1/{process_every_n_frames} frames")
            elif key == ord('-'):
                process_every_n_frames += 1
                print(f"Processing 1/{process_every_n_frames} frames")

    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Clean up
        stop_event.set()
        if processor.is_alive():
            processor.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        print("Face recognition stopped.")

if __name__ == "__main__":
    run_recognition()