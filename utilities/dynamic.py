import cv2
import time
import dlib
import numpy as np
import math
import threading
import tensorflow as tf
import os
from insightface.app import FaceAnalysis

# Set TensorFlow to use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled: {len(gpus)} GPU(s) found")
    except RuntimeError as e:
        print(f"Error enabling GPU acceleration: {e}")

startup_time = time.time()

class DynamicFaceEvaluator:
    def __init__(self):
        # Load face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Path to the shape predictor - ensure this file exists
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "models", "shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(model_path)
        
        # Tracking parameters
        self.tracking_enabled = False
        self.reference_face_position = None
        self.reference_face_size = None
        
        # Parameters for good positioning
        self.min_face_height_ratio = 0.25  # Minimum acceptable face height
        self.max_face_height_ratio = 0.6   # Maximum acceptable face height
        self.tilt_threshold = 10.0         # Maximum acceptable tilt in degrees
        
        # Tracker object for smoother tracking
        self.tracker = None
        self.tracking_face = None
        self.tracking_confidence = 0
        
        # Track initialization time
        self.init_time_ms = int((time.time() - startup_time) * 1000)
        print(f"Face evaluator initialized in {self.init_time_ms} ms")
        
        # Add to __init__
        self.frames_since_landmark_update = 0
        self.landmark_update_interval = 30  # Update landmarks every 30 frames
        
        # Initialize RetinaFace detector
        try:
            print("Initializing RetinaFace detector...")
            retinaface_start = time.time()
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            retinaface_time = int((time.time() - retinaface_start) * 1000)
            print(f"RetinaFace initialized in {retinaface_time} ms")
            self.use_retinaface = True
        except Exception as e:
            print(f"Error initializing RetinaFace: {e}")
            print("Falling back to dlib detector")
            self.use_retinaface = False
        
        # Add to __init__
        self.frames_since_detection_update = 0
        self.detection_update_interval = 30  # Run RetinaFace every 30 frames
        
    def initialize_tracking(self, frame, face_rect):
        """Initialize tracking with the current face position"""
        h, w = frame.shape[:2]
        
        # Store reference values
        self.reference_face_position = (
            (face_rect.left() + face_rect.width() // 2) / w,
            (face_rect.top() + face_rect.height() // 2) / h
        )
        self.reference_face_size = face_rect.height() / h
        
        # Try to create CSRT tracker (more accurate but slower)
        try:
            # Try newer OpenCV 4.5.x+ API
            self.tracker = cv2.TrackerCSRT.create()  # Use CSRT instead of KCF
        except AttributeError:
            try:
                # Try legacy OpenCV 3.x/4.x API
                self.tracker = cv2.TrackerCSRT_create()  # Use CSRT instead of KCF
            except AttributeError:
                try:
                    # Try OpenCV 4.5.5+ API with different syntax
                    tracker_types = {
                        'CSRT': cv2.legacy.TrackerCSRT_create,  # Try CSRT first
                        'KCF': cv2.legacy.TrackerKCF_create,    # KCF as fallback
                        'MOSSE': cv2.legacy.TrackerMOSSE_create # MOSSE as last resort
                    }
                    # Try trackers in order of speed (fastest to most accurate)
                    for tracker_name in ['MOSSE', 'KCF', 'CSRT']:  # MOSSE is fastest
                        try:
                            self.tracker = tracker_types[tracker_name]()
                            print(f"Using {tracker_name} tracker")
                            break
                        except (AttributeError, KeyError):
                            continue
                except (AttributeError, ModuleNotFoundError):
                    # Fallback to simple tracker implementation
                    print("Advanced tracking unavailable in this OpenCV version. Using basic tracking.")
                    self.tracker = None
                    self.tracking_enabled = True
                    self.tracking_face = face_rect
                    self.tracking_confidence = 1.0
                    return True
        
        # If we got here but tracker is None, all attempts failed
        if self.tracker is None:
            print("Could not initialize any tracker. Using basic tracking.")
            self.tracking_enabled = True
            self.tracking_face = face_rect
            self.tracking_confidence = 1.0
            return True
            
        # Initialize tracker with face rectangle
        bbox = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
        success = self.tracker.init(frame, bbox)
        
        if success:
            self.tracking_enabled = True
            self.tracking_face = face_rect
            self.tracking_confidence = 1.0
            return True
        else:
            self.tracking_enabled = False
            return False
        
    def evaluate_face_position(self, frame, initialize_if_needed=True):
        """Evaluate face position with dynamic adaptation"""
        if frame is None:
            return 0, {"error": "No frame provided"}, None
        
        # Make a copy for annotations
        annotated = frame.copy()
        h, w = frame.shape[:2]
        frame_center_x, frame_center_y = w // 2, h // 2
        
        # Initialize scores and feedback
        scores = {}
        feedback = {}
        face = None
        
        # Try tracking first if enabled
        if self.tracking_enabled and self.tracker is not None:
            try:
                success, bbox = self.tracker.update(frame)
                if success:
                    x, y, w_face, h_face = [int(v) for v in bbox]
                    # Convert bbox to dlib rectangle for consistency
                    face = dlib.rectangle(
                        left=x, 
                        top=y,
                        right=x + w_face,
                        bottom=y + h_face
                    )
                    self.tracking_face = face
                    self.tracking_confidence = 0.8  # Estimated confidence for tracker
                    
                    # Draw tracking box
                    cv2.rectangle(annotated, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)
                else:
                    # Tracking failed, disable it
                    self.tracking_enabled = False
                    self.tracker = None
            except Exception as e:
                print(f"Tracking error: {e}")
                self.tracking_enabled = False
                self.tracker = None
        elif self.tracking_enabled and self.tracker is None:
            # Using basic tracking (last detected position)
            face = self.tracking_face
            # Decrease confidence over time for basic tracking
            self.tracking_confidence *= 0.9
            if self.tracking_confidence < 0.5:
                self.tracking_enabled = False
        
        # Replace the face detection section in evaluate_face_position
        # If not tracking or tracking failed, use detector
        if face is None:
            # Convert to BGR for RetinaFace if using RGB
            detect_frame = frame.copy()
            
            # Check which detector to use
            if self.use_retinaface:
                # Run RetinaFace detector
                try:
                    faces_info = self.face_app.get(detect_frame)
                    if len(faces_info) == 0:
                        return 0, {"error": "No face detected"}, annotated
                    
                    # Use the largest face if multiple detected
                    if len(faces_info) > 1:
                        # Sort by bbox size and take the largest
                        faces_info = sorted(faces_info, key=lambda x: 
                                           (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), 
                                           reverse=True)
                    
                    # Convert RetinaFace bbox to dlib rectangle
                    face_info = faces_info[0]
                    bbox = face_info.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    face = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
                    
                    # Draw detected face with RetinaFace
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, "RetinaFace", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"RetinaFace error: {e}, falling back to dlib")
                    # Fall back to dlib detector
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.detector(gray)
                    if len(faces) == 0:
                        return 0, {"error": "No face detected"}, annotated
                    face = faces[0] if len(faces) == 1 else max(faces, key=lambda f: f.area())
            else:
                # Use dlib detector
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                if len(faces) == 0:
                    return 0, {"error": "No face detected"}, annotated
                face = faces[0] if len(faces) == 1 else max(faces, key=lambda f: f.area())
                    
            # Initialize tracking if needed
            if initialize_if_needed and not self.tracking_enabled:
                self.initialize_tracking(frame, face)
        
        # Get face box
        x, y, w_face, h_face = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(annotated, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
        
        # Get face center and dimensions
        face_center_x = x + w_face // 2
        face_center_y = y + h_face // 2
        
        # Calculate normalized positions
        center_x_ratio = face_center_x / frame.shape[1]
        center_y_ratio = face_center_y / frame.shape[0]
        face_height_ratio = h_face / frame.shape[0]
        
        # 1. Size score (is face a good size on screen?)
        if face_height_ratio < self.min_face_height_ratio:
            size_score = max(0, face_height_ratio / self.min_face_height_ratio)
            feedback["size"] = "Move closer"
            cv2.putText(annotated, "Move closer", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif face_height_ratio > self.max_face_height_ratio:
            size_score = max(0, self.max_face_height_ratio / face_height_ratio)
            feedback["size"] = "Move back"
            cv2.putText(annotated, "Move back", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Face size is good - normalize to 0-1 range within acceptable bounds
            size_score = 1.0 - abs((face_height_ratio - 
                                  (self.min_face_height_ratio + self.max_face_height_ratio) / 2) / 
                                 ((self.max_face_height_ratio - self.min_face_height_ratio) / 2))
            size_score = max(0, min(1, size_score))
        
        scores["size"] = size_score
        
        # 2. Tilt score (is face upright?)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if 'gray' not in locals() else gray
        landmarks = self.predictor(gray, face)
        tilt_score = 1.0
        
        if landmarks.num_parts >= 68:  # Full set of landmarks
            # Get eye points
            left_eye_x = (landmarks.part(36).x + landmarks.part(39).x) / 2
            left_eye_y = (landmarks.part(36).y + landmarks.part(39).y) / 2
            right_eye_x = (landmarks.part(42).x + landmarks.part(45).x) / 2
            right_eye_y = (landmarks.part(42).y + landmarks.part(45).y) / 2
            
            # Calculate angle
            dx = right_eye_x - left_eye_x
            dy = right_eye_y - left_eye_y
            angle = math.degrees(math.atan2(dy, dx))
            
            # Adjust tilt score
            tilt_score = max(0, 1 - abs(angle) / self.tilt_threshold)
            scores["tilt"] = tilt_score
            
            # Tilt feedback
            if abs(angle) > 5:  # More than 5 degrees
                if angle > 0:
                    feedback["tilt"] = "Tilt head clockwise"
                    cv2.putText(annotated, "Tilt head clockwise", (20, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    feedback["tilt"] = "Tilt head counter-clockwise"
                    cv2.putText(annotated, "Tilt head counter-clockwise", (20, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw facial landmarks
            for i in range(68):
                pos = (landmarks.part(i).x, landmarks.part(i).y)
                cv2.circle(annotated, pos, 2, (0, 0, 255), -1)
            
            # Draw lines connecting eyes
            cv2.line(annotated, 
                    (landmarks.part(36).x, landmarks.part(36).y),
                    (landmarks.part(39).x, landmarks.part(39).y),
                    (0, 255, 0), 1)
            cv2.line(annotated, 
                    (landmarks.part(42).x, landmarks.part(42).y),
                    (landmarks.part(45).x, landmarks.part(45).y),
                    (0, 255, 0), 1)
            
            # Draw line for mouth
            cv2.line(annotated,
                    (landmarks.part(48).x, landmarks.part(48).y),
                    (landmarks.part(54).x, landmarks.part(54).y),
                    (0, 255, 0), 1)
        
        # 3. Position score - now becomes a tracking score
        # Instead of requiring the user to be in a specific position,
        # we track face and provide feedback relative to the tracking position
        if self.tracking_enabled:
            # Position score is now proximity to being in frame
            # As long as most of the face is visible, score is high
            margin = 0.05  # 5% margin from frame edge
            x_in_frame = (center_x_ratio > margin and center_x_ratio < (1 - margin))
            y_in_frame = (center_y_ratio > margin and center_y_ratio < (1 - margin))
            
            if x_in_frame and y_in_frame:
                position_score = 1.0
            else:
                # Calculate how far outside margins
                x_outside = 0
                if center_x_ratio < margin:
                    x_outside = margin - center_x_ratio
                elif center_x_ratio > (1 - margin):
                    x_outside = center_x_ratio - (1 - margin)
                    
                y_outside = 0
                if center_y_ratio < margin:
                    y_outside = margin - center_y_ratio
                elif center_y_ratio > (1 - margin):
                    y_outside = center_y_ratio - (1 - margin)
                    
                outside_distance = math.sqrt(x_outside**2 + y_outside**2)
                position_score = max(0, 1 - outside_distance * 5)  # Scale to reduce score faster
                
                # Position feedback
                if center_x_ratio < margin:
                    feedback["position_x"] = "Moving out of frame (left edge)"
                    cv2.putText(annotated, "Moving out of frame (left)", (20, 80),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif center_x_ratio > (1 - margin):
                    feedback["position_x"] = "Moving out of frame (right edge)"
                    cv2.putText(annotated, "Moving out of frame (right)", (20, 80),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                if center_y_ratio < margin:
                    feedback["position_y"] = "Moving out of frame (top edge)"
                    cv2.putText(annotated, "Moving out of frame (top)", (20, 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif center_y_ratio > (1 - margin):
                    feedback["position_y"] = "Moving out of frame (bottom edge)"
                    cv2.putText(annotated, "Moving out of frame (bottom)", (20, 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # No tracking yet - position score is baseline
            position_score = 0.7  # Default reasonably good score
            
        scores["position"] = position_score
        
        # Add this in evaluate_face_position method where tracking is updated

        # Periodically re-run dlib facial landmark detection to ensure accuracy
        # even when using CSRT tracker
        if face is not None and self.tracking_enabled:
            self.frames_since_landmark_update += 1
            
            # Every N frames, refresh landmarks with dlib
            if self.frames_since_landmark_update >= self.landmark_update_interval:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if 'gray' not in locals() else gray
                # Re-detect landmarks (but keep current tracking)
                landmarks = self.predictor(gray, face)
                self.frames_since_landmark_update = 0

        # Update the periodic refresh section
        # Periodically re-run face detection to ensure accuracy
        if face is not None and self.tracking_enabled:
            self.frames_since_landmark_update += 1
            self.frames_since_detection_update += 1
            
            # Every N frames, refresh landmarks with dlib
            if self.frames_since_landmark_update >= self.landmark_update_interval:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if 'gray' not in locals() else gray
                # Re-detect landmarks (but keep current tracking)
                landmarks = self.predictor(gray, face)
                self.frames_since_landmark_update = 0
                
            # Every M frames, refresh detection with RetinaFace
            if self.frames_since_detection_update >= self.detection_update_interval and self.use_retinaface:
                try:
                    faces_info = self.face_app.get(frame)
                    if len(faces_info) > 0:
                        # Sort by IoU with current face if multiple detected
                        if len(faces_info) > 1:
                            current_box = [face.left(), face.top(), face.right(), face.bottom()]
                            
                            def calculate_iou(box):
                                bbox = box.bbox.astype(int)
                                # Calculate intersection
                                x1 = max(bbox[0], current_box[0])
                                y1 = max(bbox[1], current_box[1])
                                x2 = min(bbox[2], current_box[2])
                                y2 = min(bbox[3], current_box[3])
                                
                                if x2 < x1 or y2 < y1:
                                    return 0.0
                                    
                                intersection = (x2 - x1) * (y2 - y1)
                                area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                area2 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                                
                                return intersection / float(area1 + area2 - intersection)
                            
                            faces_info = sorted(faces_info, key=calculate_iou, reverse=True)
                        
                        # Get the best face match
                        face_info = faces_info[0]
                        bbox = face_info.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        
                        # Reinitialize tracker with updated position
                        face_rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
                        self.initialize_tracking(frame, face_rect)
                        
                        # Draw detection update indicator
                        cv2.putText(annotated, "Detection Updated", (20, h-30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    self.frames_since_detection_update = 0
                except Exception as e:
                    print(f"RetinaFace update error: {e}")
                    # We'll try again next interval

        # Calculate overall score
        weights = {"position": 0.3, "size": 0.4, "tilt": 0.3}
        overall_score = 0
        for metric, score in scores.items():
            overall_score += score * weights.get(metric, 0)
        overall_score = int(overall_score * 100)
        
        # Add overall score text
        cv2.putText(annotated, f"Score: {overall_score}%", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        # Add tracking status indicator
        if self.tracking_enabled:
            cv2.putText(annotated, "Tracking: ON", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "Tracking: OFF", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return overall_score, feedback, annotated

# Add this to dynamic_face_position_loop function:
def dynamic_face_position_loop(camera_index=0, resolution=(1280, 720)):
    """Run dynamic face positioning evaluation in a high-performance loop"""
    # Initialize face evaluator
    face_evaluator = DynamicFaceEvaluator()
    
    # Time camera initialization
    camera_start_time = time.time()
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Calculate camera initialization time
    camera_init_time_ms = int((time.time() - camera_start_time) * 1000)
    print(f"Camera initialized in {camera_init_time_ms} ms")
    
    # Store camera init time for display
    camera_init_ms = camera_init_time_ms
    
    # Initialize variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Frame processing thread
    stop_thread = False
    frame_queue = []
    results_queue = []
    
    def process_frames():
        while not stop_thread:
            if len(frame_queue) > 0:
                # Get the latest frame
                frame = frame_queue.pop()
                # Clear queue to process only latest frame
                frame_queue.clear()
                
                # Process frame
                try:
                    process_start = time.time()
                    score, feedback, annotated = face_evaluator.evaluate_face_position(frame)
                    process_time = time.time() - process_start
                    process_fps = 1.0 / process_time if process_time > 0 else 0
                    cv2.putText(annotated, f"Process FPS: {process_fps:.1f}", (annotated.shape[1] - 220, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
                    
                    # Store result
                    results_queue.append((score, feedback, annotated))
                    if len(results_queue) > 3:  # Keep only recent results
                        results_queue.pop(0)
                except Exception as e:
                    print(f"Error processing frame: {e}")
            
            # Sleep briefly to reduce CPU load
            time.sleep(0.005)
    
    # Start processing thread
    process_thread = threading.Thread(target=process_frames)
    process_thread.daemon = True
    process_thread.start()
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
        
        # Add frame to queue for processing
        frame_queue.append(frame.copy())
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Get latest processed result
        if results_queue:
            _, _, display_frame = results_queue[-1]
            
            # Add FPS info
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 120, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show the frame
            cv2.imshow("Dynamic Face Evaluation", display_frame)
        else:
            # If no processed frames yet, show the raw frame
            cv2.putText(frame, "Initializing...", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Dynamic Face Evaluation", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q to quit
            break
        elif key == ord('r'):  # r to reset tracking
            if face_evaluator.tracking_enabled:
                face_evaluator.tracking_enabled = False
                face_evaluator.tracker = None
                print("Tracking reset")
    
    # Clean up
    stop_thread = True
    if process_thread.is_alive():
        process_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()

def dynamic_capture_when_ready(optimal_threshold=80, wait_frames=10):
    """Run face positioning loop with option to capture when good position reached"""
    # Initialize face evaluator
    face_evaluator = DynamicFaceEvaluator()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Variables for tracking good frames
    good_frames_count = 0
    optimal_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
        
        # Evaluate face position
        score, feedback, annotated = face_evaluator.evaluate_face_position(frame)
        
        # Check if we reached good positioning
        if score >= optimal_threshold:
            good_frames_count += 1
            if good_frames_count == 1:  # First good frame
                print(f"Good position detected! Holding for {wait_frames} frames...")
            
            # Store the frame with the best score
            if optimal_frame is None or score > optimal_frame[0]:
                optimal_frame = (score, frame.copy())
                
            # Add countdown
            cv2.putText(annotated, f"Holding good position: {good_frames_count}/{wait_frames}", 
                       (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Check if we've had enough consecutive good frames
            if good_frames_count >= wait_frames:
                print(f"Position optimal! Final score: {score}")
                cap.release()
                cv2.destroyAllWindows()
                return optimal_frame[1]  # Return the best frame
        else:
            # Reset counter if position is not optimal
            if good_frames_count > 0:
                print("Position lost, resetting...")
            good_frames_count = 0
            optimal_frame = None
        
        # Display the frame with feedback
        cv2.imshow("Position your face", annotated)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q to quit
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Face Position Evaluation")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera resolution width")
    parser.add_argument("--height", type=int, default=720, help="Camera resolution height")
    parser.add_argument("--capture", action="store_true", 
                      help="Capture frame when position is good")
    parser.add_argument("--threshold", type=int, default=80,
                      help="Good position threshold (0-100)")
    
    args = parser.parse_args()
    
    if args.capture:
        frame = dynamic_capture_when_ready(
            optimal_threshold=args.threshold,
            wait_frames=10
        )
        if frame is not None:
            print("Successfully captured frame with good positioning!")
    else:
        dynamic_face_position_loop(
            camera_index=args.camera,
            resolution=(args.width, args.height)
        )
