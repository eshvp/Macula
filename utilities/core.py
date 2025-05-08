import cv2
import dlib
import numpy as np
import math
import time
import threading
import tensorflow as tf
import os

# Set TensorFlow to use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled: {len(gpus)} GPU(s) found")
    except RuntimeError as e:
        print(f"Error enabling GPU acceleration: {e}")

class FacePositionEvaluator:
    def __init__(self):
        # Load face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Path to the shape predictor - ensure this file exists
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "models", "shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(model_path)
        
        # Ideal positioning parameters
        self.ideal_face_height_ratio = 0.4
        self.ideal_face_center_x_ratio = 0.5
        self.ideal_face_center_y_ratio = 0.5
        
        # Thresholds for good positioning
        self.position_threshold = 0.1
        self.size_threshold = 0.2
        self.tilt_threshold = 10.0  # degrees
    
    def evaluate_face_position(self, frame):
        """Evaluate face position and provide feedback"""
        if frame is None:
            return 0, {"error": "No frame provided"}, None
        
        # Make a copy for annotations
        annotated = frame.copy()
        h, w = frame.shape[:2]
        frame_center_x, frame_center_y = w // 2, h // 2
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        if len(faces) == 0:
            return 0, {"error": "No face detected"}, annotated
        
        # Use the largest face if multiple detected
        if len(faces) > 1:
            max_area = 0
            largest_face_idx = 0
            for i, face in enumerate(faces):
                area = face.area()
                if area > max_area:
                    max_area = area
                    largest_face_idx = i
            face = faces[largest_face_idx]
        else:
            face = faces[0]
        
        # Get face box
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Get face center and dimensions
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Calculate normalized positions
        center_x_ratio = face_center_x / frame.shape[1]
        center_y_ratio = face_center_y / frame.shape[0]
        face_height_ratio = h / frame.shape[0]
        
        # Initialize scores and feedback
        scores = {}
        feedback = {}
        
        # 1. Center position score (is face centered?)
        x_deviation = abs(center_x_ratio - self.ideal_face_center_x_ratio)
        y_deviation = abs(center_y_ratio - self.ideal_face_center_y_ratio)
        position_deviation = math.sqrt(x_deviation**2 + y_deviation**2)
        position_score = max(0, 1 - position_deviation / self.position_threshold)
        scores["position"] = position_score
        
        # Position feedback
        if position_deviation > self.position_threshold:
            # Determine direction
            if center_x_ratio < self.ideal_face_center_x_ratio - self.position_threshold/2:
                feedback["position_x"] = "Move right"
                cv2.putText(annotated, "Move right", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif center_x_ratio > self.ideal_face_center_x_ratio + self.position_threshold/2:
                feedback["position_x"] = "Move left"
                cv2.putText(annotated, "Move left", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
            if center_y_ratio < self.ideal_face_center_y_ratio - self.position_threshold/2:
                feedback["position_y"] = "Move down"
                cv2.putText(annotated, "Move down", (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif center_y_ratio > self.ideal_face_center_y_ratio + self.position_threshold/2:
                feedback["position_y"] = "Move up"
                cv2.putText(annotated, "Move up", (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 2. Size score (is face the right size in frame?)
        size_deviation = abs(face_height_ratio - self.ideal_face_height_ratio) / self.ideal_face_height_ratio
        size_score = max(0, 1 - size_deviation)
        scores["size"] = size_score
        
        # Size feedback
        if size_deviation > self.size_threshold:
            if face_height_ratio < self.ideal_face_height_ratio:
                feedback["size"] = "Move closer"
                cv2.putText(annotated, "Move closer", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                feedback["size"] = "Move back"
                cv2.putText(annotated, "Move back", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 3. Tilt score (is face upright?)
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
        
        # 4. Draw face guide
        # Draw ideal face position rectangle
        ideal_h = int(frame.shape[0] * self.ideal_face_height_ratio)
        ideal_w = ideal_h * 3 // 4  # Assuming face aspect ratio of 3:4
        ideal_x = int(frame.shape[1] * self.ideal_face_center_x_ratio - ideal_w // 2)
        ideal_y = int(frame.shape[0] * self.ideal_face_center_y_ratio - ideal_h // 2)
        
        # Draw ideal position rectangle
        cv2.rectangle(annotated, (ideal_x, ideal_y), (ideal_x + ideal_w, ideal_y + ideal_h),
                     (0, 255, 0), 2)
        
        # 5. Calculate overall score
        weights = {"position": 0.4, "size": 0.3, "tilt": 0.3}
        overall_score = 0
        for metric, score in scores.items():
            overall_score += score * weights.get(metric, 0)
        overall_score = int(overall_score * 100)
        
        # Add overall score text
        cv2.putText(annotated, f"Position Score: {overall_score}%", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        # Draw reference lines
        cv2.line(annotated, (frame_center_x, 0), (frame_center_x, frame.shape[0]), (0, 255, 255), 1)
        cv2.line(annotated, (0, frame_center_y), (frame.shape[1], frame_center_y), (0, 255, 255), 1)
        
        return overall_score, feedback, annotated

def face_position_loop(camera_index=0, resolution=(1280, 720)):
    """Run face positioning evaluation in a high-performance loop"""
    # Initialize face evaluator
    face_evaluator = FacePositionEvaluator()
    
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
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show the frame
            cv2.imshow("Face Position Evaluation", display_frame)
        else:
            # If no processed frames yet, show the raw frame
            cv2.putText(frame, "Initializing...", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Face Position Evaluation", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q to quit
            break
    
    # Clean up
    stop_thread = True
    if process_thread.is_alive():
        process_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()

def position_evaluation_loop(capture_when_optimal=True, optimal_threshold=85, wait_frames=10):
    """Run face positioning loop with option to capture when optimal position reached"""
    # Initialize face evaluator
    face_evaluator = FacePositionEvaluator()
    
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
        
        # Check if we reached optimal positioning
        if capture_when_optimal and score >= optimal_threshold:
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
    
    parser = argparse.ArgumentParser(description="Face Position Evaluation with High Performance")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera resolution width")
    parser.add_argument("--height", type=int, default=720, help="Camera resolution height")
    parser.add_argument("--capture", action="store_true", 
                      help="Capture frame when position is optimal")
    parser.add_argument("--threshold", type=int, default=85,
                      help="Optimal position threshold (0-100)")
    
    args = parser.parse_args()
    
    if args.capture:
        frame = position_evaluation_loop(
            capture_when_optimal=True,
            optimal_threshold=args.threshold,
            wait_frames=10
        )
        if frame is not None:
            print("Successfully captured frame with good positioning!")
    else:
        face_position_loop(
            camera_index=args.camera,
            resolution=(args.width, args.height)
        )