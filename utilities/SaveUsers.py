from stored_data.UserProfiles import UserProfiles
import cv2
import os
import time
import threading
import numpy as np
import tensorflow as tf
from retinaface import RetinaFace
from queue import Queue

# Initialize UserProfiles
user_profiles = UserProfiles()

# Create a folder to save user faces
save_folder = "user_faces"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Set TensorFlow to use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled: {len(gpus)} GPU(s) found")
    except RuntimeError as e:
        print(f"Error enabling GPU acceleration: {e}")

class FaceRegistrationSystem:
    def __init__(self):
        # Initialize face detection model
        self.retinaface_model = None
        self.retinaface_enabled = False
        
        # Load RetinaFace model immediately (not in background)
        try:
            print("Loading RetinaFace model...")
            start_time = time.time()
            self.retinaface_model = RetinaFace.build_model()
            self.retinaface_enabled = True
            print(f"RetinaFace model loaded in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Error loading RetinaFace model: {e}")
            raise
        
    def detect_face(self, frame):
        """Detect faces using RetinaFace"""
        if frame is None or not self.retinaface_enabled:
            return None, None
            
        try:
            # Convert to RGB for RetinaFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = RetinaFace.detect_faces(rgb_frame, model=self.retinaface_model)
            
            if isinstance(faces, dict) and len(faces) > 0:
                # Get first face (highest confidence)
                best_face = None
                best_score = -1
                
                for face_idx, face_data in faces.items():
                    score = face_data.get('score', 0)
                    if score > best_score:
                        best_score = score
                        best_face = face_data
                
                if best_face:
                    box = best_face['facial_area']
                    landmarks = best_face['landmarks']
                    return box, landmarks
        except Exception as e:
            print(f"RetinaFace error: {e}")
            
        return None, None
    
    def evaluate_face_position(self, frame, box, landmarks):
        """Evaluate face position quality based on positioning"""
        if frame is None or box is None or landmarks is None:
            return 0, {}
            
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        face_w = x2 - x1
        face_h = y2 - y1
        face_center_x = x1 + face_w // 2
        face_center_y = y1 + face_h // 2
        
        # Calculate normalized positions
        center_x_ratio = face_center_x / w
        center_y_ratio = face_center_y / h
        face_height_ratio = face_h / h
        
        # Ideal parameters
        ideal_face_height_ratio = 0.4
        ideal_face_center_x_ratio = 0.5
        ideal_face_center_y_ratio = 0.5
        position_threshold = 0.1
        size_threshold = 0.2
        tilt_threshold = 10.0
        
        scores = {}
        feedback = {}
        
        # 1. Center position score
        x_deviation = abs(center_x_ratio - ideal_face_center_x_ratio)
        y_deviation = abs(center_y_ratio - ideal_face_center_y_ratio)
        position_deviation = np.sqrt(x_deviation**2 + y_deviation**2)
        position_score = max(0, 1 - position_deviation / position_threshold)
        scores["position"] = position_score
        
        # 2. Size score
        size_deviation = abs(face_height_ratio - ideal_face_height_ratio) / ideal_face_height_ratio
        size_score = max(0, 1 - size_deviation)
        scores["size"] = size_score
        
        # 3. Tilt score
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        tilt_score = max(0, 1 - angle / tilt_threshold)
        scores["tilt"] = tilt_score
        
        # Calculate overall score
        weights = {"position": 0.4, "size": 0.3, "tilt": 0.3}
        overall_score = 0
        for metric, score in scores.items():
            overall_score += score * weights.get(metric, 0)
        overall_score = int(overall_score * 100)  # Convert to percentage
        
        return overall_score, scores
        
    def extract_face_embedding(self, frame, box, landmarks):
        """Extract face embedding using RetinaFace's features"""
        if frame is None or box is None:
            return None
            
        try:
            # Extract face region with margin
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            
            # Add margin (20% on each side)
            margin_w = int(0.2 * (x2 - x1))
            margin_h = int(0.2 * (y2 - y1))
            
            # Make sure coordinates are within image bounds
            x1_margin = max(0, x1 - margin_w)
            y1_margin = max(0, y1 - margin_h)
            x2_margin = min(w, x2 + margin_w)
            y2_margin = min(h, y2 + margin_h)
            
            # Extract face region
            face_img = frame[y1_margin:y2_margin, x1_margin:x2_margin]
            if face_img.size == 0:
                return None
                
            # Resize to standard size
            face_img = cv2.resize(face_img, (112, 112))
            
            # Convert to RGB if not already
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                if face_img.dtype != np.float32:
                    face_img = face_img.astype(np.float32)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Create a simple embedding based on normalized facial landmarks
            # This is a placeholder for a proper face embedding model
            embedding = []
            
            # Normalize landmark positions relative to face box
            face_width = x2 - x1
            face_height = y2 - y1
            
            for name, point in landmarks.items():
                # Normalize coordinates to 0-1 range within face box
                norm_x = (point[0] - x1) / face_width
                norm_y = (point[1] - y1) / face_height
                embedding.extend([norm_x, norm_y])
                
            # Add face proportions to embedding
            embedding.extend([face_width / w, face_height / h])
            
            # Add eye distance ratio
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
            eye_distance_ratio = eye_distance / face_width
            embedding.append(eye_distance_ratio)
            
            # Add eye-to-mouth distance ratio
            nose = landmarks['nose']
            left_mouth = landmarks['left_mouth']
            right_mouth = landmarks['right_mouth']
            mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
            mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
            eye_to_mouth = np.sqrt((mouth_center_x - (left_eye[0] + right_eye[0])/2)**2 + 
                                 (mouth_center_y - (left_eye[1] + right_eye[1])/2)**2)
            eye_to_mouth_ratio = eye_to_mouth / face_height
            embedding.append(eye_to_mouth_ratio)
            
            # Return as normalized numpy array
            embedding = np.array(embedding, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize to unit length
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None

# Function to save the face with embedding
def save_face(frame, box, embedding, first_name, last_name):
    """Save face image, embedding and user data"""
    # Extract face region
    x1, y1, x2, y2 = box
    face_img = frame[y1:y2, x1:x2]
    
    # Create a filename using first and last name
    timestamp = int(time.time())
    file_path = os.path.join(save_folder, f"{first_name}_{last_name}_{timestamp}.jpg")
    cv2.imwrite(file_path, face_img)
    
    # Create user data dictionary with embedding
    user_data = {
        "first_name": first_name,
        "last_name": last_name,
        "image_path": file_path,
        "timestamp": timestamp,
        "embedding": embedding.tolist() if embedding is not None else None
    }
    
    # Add user to profiles with timestamp as ID
    user_id = str(timestamp)
    user_profiles.add_user(user_id, user_data)
    
    print(f"Face saved to {file_path}")
    print(f"User profile created with ID: {user_id}")
    if embedding is not None:
        print("Face embedding successfully stored")
    else:
        print("Warning: No face embedding was stored")

# Function to capture face with feedback
def capture_and_save_face(first_name, last_name):
    """Capture face with position feedback and save when optimal"""
    # Initialize face system
    face_system = None
    try:
        print("Initializing face registration system...")
        face_system = FaceRegistrationSystem()
        print("System ready. Please position your face in front of the camera.")
    except Exception as e:
        print(f"Failed to initialize face registration system: {e}")
        return
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Variables for tracking good positioning
    good_frames_count = 0
    best_frame = None
    best_box = None
    best_landmarks = None
    best_score = 0
    required_good_frames = 5
    min_acceptable_score = 85
    
    print("\nCAPTURING FACIAL DATA")
    print("=====================")
    print(f"User: {first_name} {last_name}")
    print("Position your face in the frame until the position score reaches at least 85%")
    print("Hold still for a few seconds when ready")
    print("Press 'q' to cancel at any time\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Detect face
        box, landmarks = face_system.detect_face(frame)
        
        display_frame = frame.copy()
        
        # Draw ideal face position guide
        h, w = frame.shape[:2]
        ideal_face_height_ratio = 0.4
        ideal_h = int(h * ideal_face_height_ratio)
        ideal_w = ideal_h * 3 // 4
        ideal_x = int(w * 0.5 - ideal_w // 2)
        ideal_y = int(h * 0.5 - ideal_h // 2)
        cv2.rectangle(display_frame, (ideal_x, ideal_y), (ideal_x + ideal_w, ideal_y + ideal_h),
                     (0, 255, 0), 2)
        
        # If face detected, evaluate position
        if box is not None and landmarks is not None:
            x1, y1, x2, y2 = box
            
            # Draw face rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw landmarks
            for name, point in landmarks.items():
                color = (0, 0, 255)  # Default red
                if 'eye' in name:
                    color = (255, 0, 0)  # Blue for eyes
                elif 'nose' in name:
                    color = (0, 255, 0)  # Green for nose
                cv2.circle(display_frame, (int(point[0]), int(point[1])), 3, color, -1)
                
            # Draw eye line
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            cv2.line(display_frame, 
                    (int(left_eye[0]), int(left_eye[1])),
                    (int(right_eye[0]), int(right_eye[1])),
                    (255, 0, 0), 2)

            # Draw mouth line if mouth landmarks exist
            if 'left_mouth' in landmarks and 'right_mouth' in landmarks:
                left_mouth = landmarks['left_mouth']
                right_mouth = landmarks['right_mouth']
                cv2.line(display_frame,
                        (int(left_mouth[0]), int(left_mouth[1])),
                        (int(right_mouth[0]), int(right_mouth[1])),
                        (0, 0, 255), 2)
                
            # Evaluate face position
            score, scores = face_system.evaluate_face_position(frame, box, landmarks)
            
            # Display scores
            cv2.putText(display_frame, f"Position Score: {score}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add positioning feedback
            if score < min_acceptable_score:
                y_pos = 60
                if scores.get('position', 0) < 0.7:
                    cv2.putText(display_frame, "Center your face in the green box", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_pos += 30
                    
                if scores.get('size', 0) < 0.7:
                    face_height_ratio = (y2 - y1) / h
                    if face_height_ratio < 0.4:
                        cv2.putText(display_frame, "Move closer to the camera", 
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame, "Move back from the camera", 
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_pos += 30
                    
                if scores.get('tilt', 0) < 0.7:
                    cv2.putText(display_frame, "Keep your head level", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                # Reset counter when position is not good
                good_frames_count = 0
            else:
                # Increment counter for consecutive good frames
                good_frames_count += 1
                
                # Update best frame if this is better
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
                    best_box = box
                    best_landmarks = landmarks
                    
                # Show countdown
                cv2.putText(display_frame, 
                           f"Good position! Hold still... {good_frames_count}/{required_good_frames}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Check if we have enough consecutive good frames
                if good_frames_count >= required_good_frames:
                    print(f"\nExcellent! Captured high-quality face image with score: {best_score}%")
                    
                    # Extract face embedding using RetinaFace landmarks
                    embedding = face_system.extract_face_embedding(best_frame, best_box, best_landmarks)
                    
                    # Save the face with embedding
                    save_face(best_frame, best_box, embedding, first_name, last_name)
                    
                    # Show success message
                    cv2.putText(display_frame, "Registration Complete!", 
                               (int(w/2)-150, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow("Face Registration", display_frame)
                    cv2.waitKey(1500)
                    break
        else:
            cv2.putText(display_frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            good_frames_count = 0
        
        # Display instructions
        cv2.putText(display_frame, "Position your face in the green box", 
                   (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to cancel", 
                   (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow("Face Registration", display_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Registration canceled by user")
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Welcome to User Face Registration")
    print("---------------------------------")
    first_name = input("Please enter your first name: ").strip()
    last_name = input("Please enter your last name: ").strip()

    # Start the capture and save face process
    capture_and_save_face(first_name, last_name)
