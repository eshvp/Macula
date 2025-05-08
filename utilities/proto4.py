# Not working. I believe there is an incompatability with opencv-contrib-python, Python itself, and the CSRT tracker.

import cv2
import dlib
import numpy as np
import math
import time
import threading
import tensorflow as tf
import os
import sys
# Set TensorFlow to use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled: {len(gpus)} GPU(s) found")
    except RuntimeError as e:
        print(f"Error enabling GPU acceleration: {e}")

class DynamicFaceEvaluator:
    def __init__(self):
        # Load dlib's face detector and landmark predictor
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
        
        # Tracker object for smoother tracking - using CSRT as specified
        self.tracker = None
        self.tracking_face = None
        self.tracking_confidence = 0
        
    def initialize_tracking(self, frame, face_rect):
        """Initialize tracking with the current face position using CSRT tracker only"""
        h, w = frame.shape[:2]
        
        # Store reference values
        self.reference_face_position = (
            (face_rect.left() + face_rect.width() // 2) / w,
            (face_rect.top() + face_rect.height() // 2) / h
        )
        self.reference_face_size = face_rect.height() / h
        
        # Try to create CSRT tracker based on OpenCV version
        try:
            # Get OpenCV version
            opencv_version = cv2.__version__.split('.')
            major_version = int(opencv_version[0])
            minor_version = int(opencv_version[1])
            
            print(f"OpenCV version: {cv2.__version__}")
            
            if major_version >= 4 and minor_version >= 5:
                # OpenCV 4.5+
                try:
                    # First try the new API
                    self.tracker = cv2.TrackerCSRT.create()
                    print("Using OpenCV 4.5+ API for CSRT")
                except AttributeError:
                    # Next try the legacy API
                    self.tracker = cv2.legacy.TrackerCSRT_create()
                    print("Using OpenCV 4.5+ legacy API for CSRT")
            else:
                # OpenCV 3.x or 4.0-4.4
                self.tracker = cv2.TrackerCSRT_create()
                print("Using OpenCV 3.x/4.x API for CSRT")
                
        except Exception as e:
            print(f"Error creating CSRT tracker: {e}")
            print("Please make sure opencv-contrib-python is properly installed.")
            print("Current OpenCV version:", cv2.__version__)
            
            # Show detailed information about available tracker types
            try:
                tracker_types = dir(cv2)
                tracker_classes = [t for t in tracker_types if "Tracker" in t]
                print("Available tracker classes:", tracker_classes)
            except:
                pass
                
            # Fail explicitly - we only want CSRT
            raise ValueError("CSRT tracker unavailable. Tracking disabled.")
        
        # Initialize tracker with face rectangle
        bbox = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
        success = self.tracker.init(frame, bbox)
        
        if success:
            self.tracking_enabled = True
            self.tracking_face = face_rect
            self.tracking_confidence = 1.0
            print("CSRT tracker initialized successfully")
            return True
        else:
            self.tracking_enabled = False
            print("Failed to initialize CSRT tracker")
            return False
        
    def evaluate_face_position(self, frame, initialize_if_needed=True):
        """Evaluate face position with dynamic adaptation using dlib landmarks"""
        if frame is None:
            return 0, {"error": "No frame provided"}, None
        
        # Make a copy for annotations
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Initialize scores and feedback
        scores = {}
        feedback = {}
        face = None
        
        # Try tracking first if enabled
        if self.tracking_enabled and self.tracker is not None:
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
        
        # If not tracking or tracking failed, use detector
        if face is None:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using dlib
            faces = self.detector(gray)
            if len(faces) == 0:
                cv2.putText(annotated, "No face detected", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return 0, {"error": "No face detected"}, annotated
            
            # Use the largest face if multiple detected
            if len(faces) > 1:
                max_area = 0
                largest_face_idx = 0
                for i, f in enumerate(faces):
                    area = f.area()
                    if area > max_area:
                        max_area = area
                        largest_face_idx = i
                face = faces[largest_face_idx]
            else:
                face = faces[0]
                
            # Initialize tracking if needed
            if initialize_if_needed and not self.tracking_enabled:
                self.initialize_tracking(frame, face)
        
        # Get face box
        x, y, w_face, h_face = face.left(), face.top(), face.width(), face.height()
        
        # Draw detected face box in green
        if not self.tracking_enabled or face is not self.tracking_face:
            cv2.rectangle(annotated, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
        
        # Get face center and dimensions
        face_center_x = x + w_face // 2
        face_center_y = y + h_face // 2
        
        # Calculate normalized positions
        center_x_ratio = face_center_x / w
        center_y_ratio = face_center_y / h
        face_height_ratio = h_face / h
        
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
        
        # 2. Tilt score using dlib's 68 landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if 'gray' not in locals() else gray
        landmarks = self.predictor(gray, face)
        tilt_score = 1.0
        
        if landmarks.num_parts == 68:  # Full set of landmarks
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
            
            # Draw all 68 facial landmarks
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
            cv2.putText(annotated, "Tracking: ON (CSRT)", (w - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "Tracking: OFF", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return overall_score, feedback, annotated

def dynamic_face_position_loop(camera_index=0, resolution=(1280, 720)):
    """Run dynamic face positioning evaluation in a high-performance loop"""
    # Initialize face evaluator
    face_evaluator = DynamicFaceEvaluator()
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
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
                    score, feedback, annotated = face_evaluator.evaluate_face_position(frame)
                    
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
    
    parser = argparse.ArgumentParser(description="Dynamic Face Position Evaluation with dlib and CSRT")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera resolution width")
    parser.add_argument("--height", type=int, default=720, help="Camera resolution height")
    parser.add_argument("--capture", action="store_true", 
                      help="Capture frame when position is good")
    parser.add_argument("--threshold", type=int, default=80,
                      help="Good position threshold (0-100)")
    
    args = parser.parse_args()
    
    # Add at beginning of main function
    print("\n---- OpenCV Environment Check ----")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV path: {cv2.__path__}")
    print(f"Python executable: {sys.executable}")

    if args.capture:
        frame = dynamic_capture_when_ready(
            optimal_threshold=args.threshold,
            wait_frames=10
        )
        if frame is not None:
            print("Successfully captured frame with good positioning!")
            # Optionally save the captured frame
            cv2.imwrite("captured_face.jpg", frame)
    else:
        dynamic_face_position_loop(
            camera_index=args.camera,
            resolution=(args.width, args.height)
        )