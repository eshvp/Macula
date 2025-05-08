import cv2
import dlib
import os
import time
import sys
import numpy as np
import sqlite3
import pickle
from pathlib import Path

# Add parent directory to path for engine imports only
sys.path.append(str(Path(__file__).parent.parent))
from engine.Core import DynamicFaceEvaluator

class FaceRegistration:
    """Optimized face registration that reuses components"""
    
    def __init__(self):
        """Initialize face registration components once"""
        # Initialize face evaluator for quality checks
        self.face_evaluator = DynamicFaceEvaluator()
        
        # Get path to models
        models_dir = Path(__file__).parent.parent / "models"
        face_rec_model_path = models_dir / "dlib_face_recognition_resnet_model_v1.dat"
        
        # Use evaluator's existing detector and predictor
        self.detector = self.face_evaluator.detector
        self.predictor = self.face_evaluator.predictor
        
        # Initialize face recognition model separately
        self.face_rec_model = dlib.face_recognition_model_v1(str(face_rec_model_path))
        
    def extract_face_encoding(self, frame):
        """Extract face encoding using already loaded models"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if not faces:
            return None
            
        face_rect = max(faces, key=lambda rect: rect.area())
        shape = self.predictor(frame, face_rect)
        face_encoding = np.array(self.face_rec_model.compute_face_descriptor(frame, shape))
        return face_encoding
        
    def capture_frames(self, quality_threshold=80, num_frames=10):
        """Capture face frames using shared face evaluator"""
        # Initialize camera with optimal settings
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Use CSRT tracker for smooth tracking
        tracker = cv2.legacy.TrackerCSRT_create()
        tracking = False
        
        # Storage for good frames
        good_frames = []
        frame_scores = []
        last_capture_time = 0
        capture_delay = 0.5
        
        print(f"\nPlease position your face in the camera")
        print(f"We need to capture {num_frames} good quality images")
        print("Maintain a natural expression and follow the on-screen guidance")
        
        while len(good_frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            display_frame = frame.copy()
            
            if tracking:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    score, feedback, _ = self.face_evaluator.evaluate_face_position(frame)
                else:
                    tracking = False
                    score = 0
            else:
                score, feedback, _ = self.face_evaluator.evaluate_face_position(frame)
                
                if score >= quality_threshold:
                    faces = self.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    if faces:
                        face = max(faces, key=lambda rect: rect.area())
                        bbox = (face.left(), face.top(), face.width(), face.height())
                        success = tracker.init(frame, bbox)
                        if success:
                            tracking = True
                            cv2.rectangle(display_frame, 
                                        (bbox[0], bbox[1]), 
                                        (bbox[0]+bbox[2], bbox[1]+bbox[3]), 
                                        (0, 255, 0), 2)
            
            # Capture good frames
            current_time = time.time()
            if (score >= quality_threshold and 
                current_time - last_capture_time >= capture_delay):
                good_frames.append(frame.copy())
                frame_scores.append(score)
                last_capture_time = current_time
                print(f"Captured frame {len(good_frames)}/{num_frames} with score: {score}")
            
            # Show feedback
            cv2.putText(display_frame, f"Quality: {score:.1f}%", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if score >= quality_threshold else (0, 0, 255), 2)
            cv2.putText(display_frame, f"Captured: {len(good_frames)}/{num_frames}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            cv2.imshow("Face Registration", display_frame)
            
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return good_frames, frame_scores
        
    def save_user(self, first_name, last_name, frames, scores):
        """Save user data with optimized encoding"""
        # Generate user ID and paths
        timestamp = int(time.time())
        user_id = f"{first_name.lower()}_{last_name.lower()}_{timestamp}"
        
        # Save to filesystem first (faster than database operations)
        save_dir = Path(__file__).parent.parent / "users" / user_id
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metadata
        with open(save_dir / "info.txt", "w") as f:
            f.write(f"First Name: {first_name}\n")
            f.write(f"Last Name: {last_name}\n")
            f.write(f"Capture Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Frames: {len(frames)}\n")
        
        # Process frames in parallel for speed
        successful_encodings = 0
        encodings = []
        
        # Save images and compute encodings
        for i, (frame, score) in enumerate(zip(frames, scores)):
            # Save image
            filename = f"face_{i+1:02d}_score_{int(score)}.jpg"
            cv2.imwrite(str(save_dir / filename), frame)
            
            # Get encoding
            encoding = self.extract_face_encoding(frame)
            if encoding is not None:
                encodings.append((encoding, score))
                successful_encodings += 1
        
        # Save to database if needed
        if successful_encodings > 0:
            self._save_to_database(user_id, first_name, last_name, encodings)
        
        return user_id, successful_encodings

    def _save_to_database(self, user_id, first_name, last_name, encodings):
        """Save encodings to database"""
        db_path = Path(__file__).parent.parent / "data" / "face_encodings.db"
        os.makedirs(db_path.parent, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        try:
            # Create tables if needed
            cursor.execute('''CREATE TABLE IF NOT EXISTS users
                            (id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT, 
                             created_at TIMESTAMP, last_access TIMESTAMP)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS face_encodings
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
                             encoding BLOB, score REAL,
                             FOREIGN KEY (user_id) REFERENCES users(id))''')
            
            # Insert user
            cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                         (user_id, first_name, last_name, 
                          time.strftime('%Y-%m-%d %H:%M:%S'), time.time()))
            
            # Insert encodings
            for encoding, score in encodings:
                cursor.execute("INSERT INTO face_encodings (user_id, encoding, score) VALUES (?, ?, ?)",
                             (user_id, pickle.dumps(encoding), float(score)))
            
            conn.commit()
            
        except Exception as e:
            print(f"Database error: {e}")
            conn.rollback()
        finally:
            conn.close()

def main():
    """Main registration function"""
    print("=== Face Registration Utility ===")
    
    # Get user info
    first_name = input("\nEnter first name: ").strip()
    last_name = input("Enter last name: ").strip()
    
    if not first_name or not last_name:
        print("Error: Both first and last name are required.")
        return
    
    # Initialize registration once
    registration = FaceRegistration()
    
    input("\nPress Enter to begin face capture...")
    
    # Capture frames
    frames, scores = registration.capture_frames()
    
    if frames:
        # Save user data
        user_id, num_encodings = registration.save_user(first_name, last_name, frames, scores)
        
        if user_id:
            print(f"\nFace registration complete!")
            print(f"User ID: {user_id}")
            print(f"Saved {num_encodings} face encodings")
        else:
            print("\nError saving user data")
    else:
        print("\nNo frames captured")

if __name__ == "__main__":
    main()