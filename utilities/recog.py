import cv2
import numpy as np
import os
import time
import threading
import tensorflow as tf
from queue import Queue
from retinaface import RetinaFace
from stored_data.UserProfiles import UserProfiles
import dlib  # Still needed for the 68-point landmark model only

# Set TensorFlow to use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled: {len(gpus)} GPU(s) found")
    except RuntimeError as e:
        print(f"Error enabling GPU acceleration: {e}")

class FaceRecognition:
    def __init__(self):
        """Initialize the face recognition system with RetinaFace detection"""
        # Initialize models and flags
        self.retinaface_model = None
        self.retinaface_enabled = False
        
        # Load only the 68-point landmark model from dlib
        try:
            # Use absolute paths based on the file location
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            shape_path = os.path.join(base_dir, "models", "shape_predictor_68_face_landmarks.dat")
            face_rec_path = os.path.join(base_dir, "models", "dlib_face_recognition_resnet_model_v1.dat")
            
            print(f"Looking for models at:\n{shape_path}\n{face_rec_path}")
            
            if not os.path.exists(shape_path):
                raise FileNotFoundError(f"Shape predictor file not found at: {shape_path}")
                
            if not os.path.exists(face_rec_path):
                raise FileNotFoundError(f"Face recognition model file not found at: {face_rec_path}")
                
            # Only load the shape predictor for detailed landmarks
            self.shape_predictor = dlib.shape_predictor(shape_path)
            self.face_recognizer = dlib.face_recognition_model_v1(face_rec_path)
            print("Face landmark and recognition models loaded successfully")
        except Exception as e:
            print(f"Error loading face landmark model: {e}")
            raise
        
        # Load user profiles
        self.user_profiles = UserProfiles()
        self.threshold = 0.6  # Recognition threshold
        
        # Pre-load all user encodings
        self.reference_encodings = {}
        self._load_user_encodings()
        
        # Cache for recent recognitions to avoid redundant processing
        self.recognition_cache = {}
        self.cache_lifetime = 2.0  # seconds
        
        # Start loading RetinaFace model in background
        self._init_retinaface()
        
    def _init_retinaface(self):
        """Load RetinaFace model in a background thread"""
        def load_retinaface():
            try:
                print("Loading RetinaFace model...")
                start_time = time.time()
                
                # Wrap RetinaFace model building in a try-catch with proper Keras handling
                try:
                    # Use RetinaFace.build_model() directly
                    self.retinaface_model = RetinaFace.build_model()
                    self.retinaface_enabled = True
                    print(f"RetinaFace model loaded in {time.time() - start_time:.2f}s")
                except Exception as e:
                    # Try alternative approach with TF 2.x compatibility
                    print(f"Standard RetinaFace loading failed: {e}")
                    print("Trying alternative loading method...")
                    
                    # Import the model directly to avoid Keras/TF compatibility issues
                    from tensorflow.keras.models import load_model
                    
                    # Look for the RetinaFace model in common locations
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    model_paths = [
                        os.path.join(base_dir, "models", "retinaface_model.h5"),
                        os.path.join(base_dir, "models", "retinaface.h5")
                    ]
                    
                    model_loaded = False
                    for path in model_paths:
                        if os.path.exists(path):
                            try:
                                self.retinaface_model = load_model(path, compile=False)
                                self.retinaface_enabled = True
                                model_loaded = True
                                print(f"RetinaFace model loaded from {path}")
                                break
                            except Exception as e2:
                                print(f"Failed loading from {path}: {e2}")
                    
                    if not model_loaded:
                        raise Exception("Could not load RetinaFace model from any location")
                        
            except Exception as e:
                print(f"Error loading RetinaFace model: {e}")
                print("Face detection will not work without RetinaFace")
        
        # Start loading in a separate thread
        retinaface_thread = threading.Thread(target=load_retinaface)
        retinaface_thread.daemon = True
        retinaface_thread.start()
    
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
                # Load image and extract face using RetinaFace when ready
                try:
                    # Store the path for later processing
                    self.reference_encodings[user_id] = {
                        'image_path': user_data['image_path'],
                        'first_name': user_data['first_name'],
                        'last_name': user_data['last_name'],
                        'encoding': None  # Will be filled later
                    }
                    # Will be processed once RetinaFace is loaded
                except Exception as e:
                    print(f"Error preparing user {user_id}: {e}")
                    
        print(f"Loaded {loaded_count} face encodings immediately")
        
        # Start a thread to process remaining encodings once RetinaFace is ready
        threading.Thread(target=self._process_pending_encodings, daemon=True).start()
        
    def _process_pending_encodings(self):
        """Process any pending encodings once RetinaFace is ready"""
        # Wait for RetinaFace to be loaded
        while not self.retinaface_enabled:
            time.sleep(0.5)
        
        print("RetinaFace ready, processing pending encodings...")
        pending_count = 0
        success_count = 0
        
        for user_id, user_data in list(self.reference_encodings.items()):
            if 'image_path' in user_data and user_data['encoding'] is None:
                pending_count += 1
                try:
                    image = cv2.imread(user_data['image_path'])
                    encoding = self.get_face_encoding(image)
                    
                    if encoding is not None:
                        user_data['encoding'] = encoding
                        
                        # Update user profile with the embedding
                        profile_data = self.user_profiles.get_user(user_id)
                        if profile_data:
                            profile_data['embedding'] = encoding.tolist()
                            self.user_profiles.add_user(user_id, profile_data)
                            
                        success_count += 1
                except Exception as e:
                    print(f"Error processing encoding for user {user_id}: {e}")
        
        print(f"Processed {pending_count} pending encodings with {success_count} successes")
    
    def get_face_encoding(self, image):
        """Extract face encoding from image using RetinaFace + dlib face recognition model"""
        if image is None or not self.retinaface_enabled:
            return None
            
        try:
            # Convert to RGB for RetinaFace
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces with RetinaFace
            faces = RetinaFace.detect_faces(rgb_image, model=self.retinaface_model)
            
            if not isinstance(faces, dict) or len(faces) == 0:
                return None
            
            # Use the face with highest confidence if multiple detected
            best_face = None
            best_score = -1
            
            for face_idx, face_data in faces.items():
                score = face_data.get('score', 0)
                if score > best_score:
                    best_score = score
                    best_face = face_data
            
            if best_face is None:
                return None
                
            # Get face box
            box = best_face["facial_area"]
            x1, y1, x2, y2 = box
            
            # Convert to dlib rectangle for the shape predictor
            rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
            
            # Get shape using the 68-point model
            shape = self.shape_predictor(rgb_image, rect)
            
            # Compute face descriptor
            face_encoding = np.array(self.face_recognizer.compute_face_descriptor(rgb_image, shape))
            return face_encoding
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None
    
    def compare_faces(self, known_encoding, unknown_encoding):
        """Compare two face encodings and return distance (lower is better)"""
        if known_encoding is None or unknown_encoding is None:
            return 1.0
        
        return np.linalg.norm(known_encoding - unknown_encoding)
    
    def recognize_face(self, encoding, face_id=None):
        """Recognize face using pre-loaded encodings with vectorized comparison"""
        if encoding is None:
            return None
        
        # Check cache first
        current_time = time.time()
        if face_id is not None and face_id in self.recognition_cache:
            cache_entry = self.recognition_cache[face_id]
            if current_time - cache_entry['time'] < self.cache_lifetime:
                return cache_entry['result']
        
        # No cache hit, perform recognition with vectorized comparison
        best_match = None
        min_distance = float('inf')
        
        if len(self.reference_encodings) > 0:
            # Create arrays for vectorized comparison
            valid_users = {}
            valid_encodings = []
            
            # Filter out entries without encodings
            for uid, data in self.reference_encodings.items():
                if 'encoding' in data and data['encoding'] is not None:
                    valid_users[len(valid_encodings)] = uid
                    valid_encodings.append(data['encoding'])
            
            if valid_encodings:
                # Convert to numpy array
                encoding_array = np.array(valid_encodings)
                
                # Calculate distances to all encodings at once
                distances = np.linalg.norm(encoding_array - encoding, axis=1)
                
                # Find minimum distance and corresponding user
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                
                # Check if the match is below threshold
                if min_distance < self.threshold:
                    matched_id = valid_users[min_idx]
                    first_name = self.reference_encodings[matched_id]['first_name']
                    last_name = self.reference_encodings[matched_id]['last_name']
                    
                    best_match = {
                        'id': matched_id,
                        'first_name': first_name,
                        'last_name': last_name,
                        'confidence': 1.0 - min_distance / self.threshold,
                        'distance': min_distance
                    }
        
        # Update cache
        if face_id is not None:
            self.recognition_cache[face_id] = {
                'time': current_time,
                'result': best_match
            }
        
        return best_match
    
    def detect_and_recognize(self, frame, fast_mode=True):
        """Detect and recognize faces using RetinaFace and 68-point landmarks"""
        if frame is None or not self.retinaface_enabled:
            return []
            
        results = []
        frame_height, frame_width = frame.shape[:2]
        
        try:
            # Convert to RGB for RetinaFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = RetinaFace.detect_faces(rgb_frame, model=self.retinaface_model)
            
            # Process detections
            if isinstance(faces, dict) and len(faces) > 0:
                for face_idx, face_data in faces.items():
                    box = face_data["facial_area"]
                    x1, y1, x2, y2 = box
                    
                    # Extract face region with margin
                    margin = 10
                    x1_safe = max(0, x1 - margin)
                    y1_safe = max(0, y1 - margin)
                    x2_safe = min(frame_width, x2 + margin)
                    y2_safe = min(frame_height, y2 + margin)
                    
                    face_img = frame[y1_safe:y2_safe, x1_safe:x2_safe]
                    if face_img.size == 0:
                        continue
                    
                    # Generate unique face ID
                    face_id = f"face_{face_idx}_{int(time.time())}"
                    
                    # Set up dlib rectangle for landmarks
                    rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
                    
                    # Get detailed 68-point landmarks
                    try:
                        landmarks = self.shape_predictor(rgb_frame, rect)
                        detailed_landmarks = []
                        
                        # Extract all 68 points
                        for i in range(68):
                            x = landmarks.part(i).x
                            y = landmarks.part(i).y
                            detailed_landmarks.append((x, y))
                            
                        # Group landmarks by feature for easier access
                        landmarks_by_feature = {
                            'jaw': detailed_landmarks[0:17],
                            'right_eyebrow': detailed_landmarks[17:22],
                            'left_eyebrow': detailed_landmarks[22:27],
                            'nose_bridge': detailed_landmarks[27:31],
                            'nose_tip': detailed_landmarks[31:36],
                            'right_eye': detailed_landmarks[36:42],
                            'left_eye': detailed_landmarks[42:48],
                            'outer_lips': detailed_landmarks[48:60],
                            'inner_lips': detailed_landmarks[60:68]
                        }
                        
                        # Calculate average positions for main features
                        main_landmarks = {
                            'left_eye': tuple(map(int, np.mean(landmarks_by_feature['left_eye'], axis=0))),
                            'right_eye': tuple(map(int, np.mean(landmarks_by_feature['right_eye'], axis=0))),
                            'nose': (landmarks.part(30).x, landmarks.part(30).y),
                            'left_mouth': (landmarks.part(48).x, landmarks.part(48).y),
                            'right_mouth': (landmarks.part(54).x, landmarks.part(54).y)
                        }
                        
                        # Get encoding directly using the detailed landmarks
                        face_encoding = np.array(self.face_recognizer.compute_face_descriptor(rgb_frame, landmarks))
                        
                        # Recognize face
                        match = self.recognize_face(face_encoding, face_id)
                        
                        # Store all information
                        results.append({
                            'rect': (x1, y1, x2, y2),
                            'encoding': face_encoding,
                            'match': match,
                            'face_id': face_id,
                            'confidence': face_data.get('score', 1.0),
                            'landmarks': main_landmarks,
                            'detailed_landmarks': detailed_landmarks,
                            'landmarks_by_feature': landmarks_by_feature
                        })
                    except Exception as e:
                        print(f"Error processing landmarks: {e}")
                        # Use basic RetinaFace landmarks if detailed extraction fails
                        basic_landmarks = {}
                        for key, point in face_data["landmarks"].items():
                            basic_landmarks[key] = (int(point[0]), int(point[1]))
                        
                        # Try to get encoding from face image
                        face_encoding = self.get_face_encoding(face_img)
                        match = self.recognize_face(face_encoding, face_id)
                        
                        results.append({
                            'rect': (x1, y1, x2, y2),
                            'encoding': face_encoding,
                            'match': match,
                            'face_id': face_id,
                            'confidence': face_data.get('score', 1.0),
                            'landmarks': basic_landmarks
                        })
        except Exception as e:
            print(f"Error in face detection: {e}")
        
        return results

def run_recognition():
    """Main function to run face recognition with multi-threading"""
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
    
    # Process every N frames for better performance
    process_every_n_frames = 2
    frame_counter = 0
    
    # Display settings
    display_frame = None
    recognition_results = []
    show_detailed_landmarks = False  # Toggle with 'd' key
    
    # Add these constants near the beginning of run_recognition()
    # Display settings for sidebar
    sidebar_width = 200
    show_sidebar = True  # Can be toggled with 's' key
    
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
                    result_queue.put((frame.copy(), results))
            except Exception as e:
                if not isinstance(e, TimeoutError) and not stop_event.is_set():
                    print(f"Error in processing thread: {e}")
    
    # Start processing thread
    processor = threading.Thread(target=process_frames)
    processor.daemon = True
    processor.start()
    
    print("Face recognition started. Press 'q' to quit, '+'/'-' to adjust processing rate, 'd' to toggle detailed landmarks.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
                
            frame_counter += 1
            
            # Process only every Nth frame
            if frame_counter % process_every_n_frames == 0:
                # Clear existing frame and add new one
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                
                try:
                    frame_queue.put_nowait(frame.copy())
                except:
                    pass
            
            # Get latest results if available
            try:
                if not result_queue.empty():
                    last_processed_frame, recognition_results = result_queue.get_nowait()
            except:
                pass
                
            # Create display frame with sidebar
            if show_sidebar:
                # Create a wider frame with sidebar
                h, w = frame.shape[:2]
                display_frame = np.zeros((h, w + sidebar_width, 3), dtype=np.uint8)
                display_frame[:, 0:w] = frame.copy()
                display_frame[:, w:] = (45, 45, 45)  # Dark gray sidebar
                
                # Add sidebar title
                cv2.putText(display_frame, "Recognized Users", (w + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.line(display_frame, (w + 10, 40), (w + sidebar_width - 10, 40), (200, 200, 200), 1)
                
                # Display recent recognitions in sidebar
                y_pos = 70
                recent_recognitions = {}
                
                # Collect recent recognitions from current frame
                for face_data in recognition_results:
                    match = face_data.get('match')
                    if match is not None:
                        user_id = match.get('id')
                        if user_id and user_id not in recent_recognitions:
                            recent_recognitions[user_id] = {
                                'name': f"{match['first_name']} {match['last_name']}",
                                'confidence': match['confidence'],
                                'last_seen': time.time()
                            }
                
                # Display recent recognitions
                if recent_recognitions:
                    for user_id, recognition_data in recent_recognitions.items():
                        name = recognition_data['name']
                        conf = recognition_data['confidence'] * 100
                        
                        # Draw name
                        cv2.putText(display_frame, name, (w + 10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        
                        # Draw confidence bar
                        bar_length = int((sidebar_width - 20) * conf / 100)
                        cv2.rectangle(display_frame, 
                                     (w + 10, y_pos + 5), 
                                     (w + 10 + bar_length, y_pos + 15),
                                     (0, int(255 * conf/100), 0), -1)
                        
                        # Draw confidence text
                        cv2.putText(display_frame, f"{conf:.1f}%", (w + 15, y_pos + 13), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        y_pos += 40
                else:
                    cv2.putText(display_frame, "No faces recognized", (w + 10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            else:
                display_frame = frame.copy()
            
            # Draw results on display frame
            for face_data in recognition_results:
                x1, y1, x2, y2 = face_data['rect']
                match = face_data['match']
                landmarks = face_data.get('landmarks', {})
                confidence = face_data.get('confidence', 0)
                
                # Draw face rectangle with different colors based on recognition
                if match is not None:
                    # Draw green rectangle for recognized face
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add name and confidence
                    name = f"{match['first_name']} {match['last_name']}"
                    conf_text = f"{match['confidence']*100:.1f}%"
                    cv2.putText(display_frame, name, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(display_frame, conf_text, (x1, y2+25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Red rectangle for unknown face
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, "Unknown", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Draw detailed landmarks if enabled
                if show_detailed_landmarks and 'detailed_landmarks' in face_data:
                    # Draw all 68 points
                    for point in face_data['detailed_landmarks']:
                        cv2.circle(display_frame, point, 1, (0, 255, 255), -1)
                    
                    # Draw contours by feature groups
                    if 'landmarks_by_feature' in face_data:
                        features = face_data['landmarks_by_feature']
                        
                        # Draw jawline
                        for i in range(len(features['jaw'])-1):
                            cv2.line(display_frame, features['jaw'][i], features['jaw'][i+1], (0, 255, 255), 1)
                        
                        # Draw eyebrows
                        for i in range(len(features['right_eyebrow'])-1):
                            cv2.line(display_frame, features['right_eyebrow'][i], features['right_eyebrow'][i+1], (0, 255, 255), 1)
                        for i in range(len(features['left_eyebrow'])-1):
                            cv2.line(display_frame, features['left_eyebrow'][i], features['left_eyebrow'][i+1], (0, 255, 255), 1)
                        
                        # Draw nose
                        for i in range(len(features['nose_bridge'])-1):
                            cv2.line(display_frame, features['nose_bridge'][i], features['nose_bridge'][i+1], (0, 255, 255), 1)
                        for i in range(len(features['nose_tip'])-1):
                            cv2.line(display_frame, features['nose_tip'][i], features['nose_tip'][i+1], (0, 255, 255), 1)
                        
                        # Draw eyes
                        for i in range(len(features['right_eye'])):
                            cv2.line(display_frame, features['right_eye'][i], 
                                    features['right_eye'][(i+1) % len(features['right_eye'])], (255, 0, 0), 1)
                        for i in range(len(features['left_eye'])):
                            cv2.line(display_frame, features['left_eye'][i], 
                                    features['left_eye'][(i+1) % len(features['left_eye'])], (255, 0, 0), 1)
                        
                        # Draw lips
                        for i in range(len(features['outer_lips'])):
                            cv2.line(display_frame, features['outer_lips'][i], 
                                    features['outer_lips'][(i+1) % len(features['outer_lips'])], (0, 0, 255), 1)
                        for i in range(len(features['inner_lips'])):
                            cv2.line(display_frame, features['inner_lips'][i], 
                                    features['inner_lips'][(i+1) % len(features['inner_lips'])], (0, 0, 255), 1)
                else:
                    # Draw main landmarks with different colors
                    for name, point in landmarks.items():
                        color = (0, 0, 255)  # Default red
                        if 'eye' in name:
                            color = (255, 0, 0)  # Blue for eyes
                        elif 'nose' in name:
                            color = (0, 255, 0)  # Green for nose
                        elif 'mouth' in name:
                            color = (255, 0, 255)  # Purple for mouth
                            
                        cv2.circle(display_frame, point, 3, color, -1)
                    
                    # Draw connecting lines between landmarks
                    if 'left_eye' in landmarks and 'right_eye' in landmarks:
                        cv2.line(display_frame, landmarks['left_eye'], 
                                landmarks['right_eye'], (255, 0, 0), 1)
                    
                    if 'left_mouth' in landmarks and 'right_mouth' in landmarks:
                        cv2.line(display_frame, landmarks['left_mouth'], 
                                landmarks['right_mouth'], (255, 0, 255), 1)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()
                
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Show processing rate
            cv2.putText(display_frame, f"Processing: 1/{process_every_n_frames} frames", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                       
            # Show landmarks display mode
            landmark_mode = "Detailed" if show_detailed_landmarks else "Basic"
            cv2.putText(display_frame, f"Landmarks: {landmark_mode}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Face Recognition", display_frame)
            
            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                process_every_n_frames = max(1, process_every_n_frames - 1)
                print(f"Processing rate: 1/{process_every_n_frames} frames")
            elif key == ord('-'):
                process_every_n_frames = min(10, process_every_n_frames + 1)
                print(f"Processing rate: 1/{process_every_n_frames} frames")
            elif key == ord('d'):
                show_detailed_landmarks = not show_detailed_landmarks
                print(f"Detailed landmarks: {show_detailed_landmarks}")
            elif key == ord('s'):
                show_sidebar = not show_sidebar
                print(f"Sidebar display: {'On' if show_sidebar else 'Off'}")
    
    finally:
        # Clean up
        stop_event.set()
        if processor.is_alive():
            processor.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()