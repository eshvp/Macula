import cv2
import dlib
import numpy as np
import time
from pathlib import Path
from data.users import user_manager
from engine.Core import DynamicFaceEvaluator

class FaceRecognizer:
    """Class to recognize faces of registered users"""
    
    def __init__(self, recognition_threshold=0.6):
        """Initialize the face recognizer with models and thresholds"""
        # Path to face recognition model
        models_dir = Path(__file__).parent.parent / "models"
        self.face_rec_model_path = models_dir / "dlib_face_recognition_resnet_model_v1.dat"
        self.shape_predictor_path = models_dir / "shape_predictor_68_face_landmarks.dat"
        
        # Initialize face detection/analysis components
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(str(self.shape_predictor_path))
        self.face_rec_model = dlib.face_recognition_model_v1(str(self.face_rec_model_path))
        
        # Set recognition threshold (lower = stricter matching)
        self.recognition_threshold = recognition_threshold
        
        # Use the face evaluator for better face position analysis
        self.face_evaluator = DynamicFaceEvaluator()
        
        # Cache of known face encodings
        self.known_face_encodings = []
        self.known_user_ids = []
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load all registered user faces and compute their encodings"""
        print("Loading known faces...")
        start_time = time.time()
        
        # Clear existing cache
        self.known_face_encodings = []
        self.known_user_ids = []
        
        # Get all registered users
        all_users = user_manager.get_all_users()
        total_images = 0
        
        for user in all_users:
            user_id = user['id']
            print(f"Processing user: {user['first_name']} {user['last_name']}")
            
            # Load user's face images
            image_tuples = user_manager.load_user_face_images(user_id)
            if not image_tuples:
                print(f"Warning: No face images found for user {user_id}")
                continue
                
            for img_path, img in image_tuples:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.detector(gray)
                if len(faces) == 0:
                    print(f"Warning: No face detected in image {img_path}")
                    continue
                
                # Use the largest face if multiple detected
                face_rect = max(faces, key=lambda rect: rect.area())
                
                # Get face landmarks
                shape = self.shape_predictor(img, face_rect)
                
                # Compute face encoding (128-dimensional face descriptor)
                face_encoding = self.face_rec_model.compute_face_descriptor(img, shape)
                face_encoding_np = np.array(face_encoding)
                
                # Add to known faces
                self.known_face_encodings.append(face_encoding_np)
                self.known_user_ids.append(user_id)
                total_images += 1
        
        elapsed_time = time.time() - start_time
        print(f"Loaded {len(self.known_face_encodings)} face encodings from {len(all_users)} users in {elapsed_time:.2f} seconds")
        return total_images
    
    def recognize_face(self, frame, min_quality_score=70):
        """Recognize a face in the given frame"""
        # Evaluate face quality
        score, feedback, annotated = self.face_evaluator.evaluate_face_position(frame)
        
        # If face quality is too low, don't attempt recognition
        if score < min_quality_score:
            return None, score, "Face quality too low for reliable recognition", annotated
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        if len(faces) == 0:
            return None, score, "No face detected", annotated
        
        # Use the largest face if multiple detected
        face_rect = max(faces, key=lambda rect: rect.area())
        
        # Get face landmarks
        shape = self.shape_predictor(frame, face_rect)
        
        # Compute face encoding
        face_encoding = self.face_rec_model.compute_face_descriptor(frame, shape)
        face_encoding_np = np.array(face_encoding)
        
        # If no known faces, return unknown
        if len(self.known_face_encodings) == 0:
            return None, score, "No known faces to compare with", annotated
        
        # Calculate distances to all known faces
        distances = []
        for known_encoding in self.known_face_encodings:
            # Euclidean distance between face encodings
            distance = np.linalg.norm(face_encoding_np - known_encoding)
            distances.append(distance)
        
        # Find the closest match
        best_match_idx = np.argmin(distances)
        best_match_distance = distances[best_match_idx]
        
        # Check if the match is close enough
        if best_match_distance <= self.recognition_threshold:
            recognized_user_id = self.known_user_ids[best_match_idx]
            
            # Get user details
            user_info = user_manager.get_user_by_id(recognized_user_id)
            first_name = user_info.get('first_name', 'Unknown')
            last_name = user_info.get('last_name', 'User')
            
            # Add recognition info to the annotated frame
            cv2.putText(annotated, f"Recognized: {first_name} {last_name}", 
                       (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, f"Match confidence: {(1-best_match_distance)*100:.1f}%", 
                       (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            return user_info, score, f"Recognized: {first_name} {last_name}", annotated
        else:
            # Add unknown user info to the annotated frame
            cv2.putText(annotated, "Unknown person", 
                       (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated, f"Best match distance: {best_match_distance:.3f}", 
                       (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return None, score, "Unknown person", annotated
    
    def recognition_loop(self):
        """Run a continuous recognition loop with camera input"""
        print("\nStarting face recognition...")
        print("Press 'q' or ESC to quit, 'r' to reload known faces")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # FPS tracking
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera")
                break
            
            # Update FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Perform recognition
            user, score, message, annotated = self.recognize_face(frame)
            
            # Add FPS display
            cv2.putText(annotated, f"FPS: {fps:.1f}", 
                       (annotated.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
            
            # Display the annotated frame
            cv2.imshow("Face Recognition", annotated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break
            elif key == ord('r'):  # r to reload face database
                print("Reloading face database...")
                self.load_known_faces()
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
    
    def recognize_single_frame(self, frame=None):
        """Capture a single frame and perform recognition, or use provided frame"""
        if frame is None:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Capture a single frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("Error reading from camera")
                return None, 0, "Camera error", None
        
        # Perform recognition
        return self.recognize_face(frame)