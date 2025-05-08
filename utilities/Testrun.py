import cv2
import time
import argparse
import threading
import numpy as np
from retinaface import RetinaFace
import tensorflow as tf
import datetime
import os
import pickle
import dlib
import math

# Set TensorFlow to use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled: {len(gpus)} GPU(s) found")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Error enabling GPU acceleration: {e}")

def laptop_camera_monitor(camera_index=0, resolution=None, show_fps=True, continuous_tracking=True):
    """
    Real-time monitoring using the laptop's camera with RetinaFace overlay
    for continuous 24/7 tracking
    
    Args:
        camera_index: Camera device index (default 0 is usually the built-in webcam)
        resolution: Optional tuple of (width, height) to set camera resolution
        show_fps: Whether to display the FPS counter
        continuous_tracking: Whether to enable persistent tracking features
    """
    print("Starting 24/7 laptop camera monitoring...")
    print("Press 'q' to quit, 's' to save a snapshot, 'r' to toggle RetinaFace")
    print("RetinaFace will activate in 10 seconds...")
    
    # Create log directory if it doesn't exist
    log_dir = "tracking_logs"
    if continuous_tracking and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Initialize the camera
    cap = cv2.VideoCapture(camera_index)
    
    # Set resolution if specified
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Create window with normal size that can be resized
    cv2.namedWindow("24/7 Tracking Monitor", cv2.WINDOW_NORMAL)
    
    # For FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    # Control flags for threads
    stop_threads = False
    retinaface_enabled = False
    retinaface_ready = False
    retinaface_model = None
    
    # Processing queue
    frames_to_process = []
    face_detections = []
    
    # For activity tracking
    face_present = False
    last_face_time = time.time()
    face_present_duration = 0
    face_absent_duration = 0
    tracking_stats = {
        "session_start": datetime.datetime.now(),
        "total_tracking_time": 0,
        "face_present_time": 0,
        "face_absent_time": 0,
        "detection_count": 0,
        "snapshot_count": 0
    }
    
    # For face identification and persistence
    known_faces = {}
    face_ids_counter = 0
    
    # For RetinaFace notification
    notification_time = 0
    show_notification = False
    
    # Start time for RetinaFace activation countdown
    start_time = time.time()
    
    # For periodic snapshot saving (every 10 minutes if face detected)
    last_auto_snapshot_time = time.time()
    auto_snapshot_interval = 600  # 10 minutes
    
    # Add face position evaluator
    class FacePositionEvaluator:
        def __init__(self):
            # Ideal positioning parameters
            self.ideal_face_height_ratio = 0.4
            self.ideal_face_center_x_ratio = 0.5
            self.ideal_face_center_y_ratio = 0.5
            
            # Thresholds for good positioning
            self.position_threshold = 0.1
            self.size_threshold = 0.2
            self.tilt_threshold = 10.0  # degrees
            
        def evaluate_face_position(self, frame, face_data):
            """Evaluate face position and provide feedback"""
            if frame is None or face_data is None:
                return 0, {"error": "No face detected"}, frame
                
            scores = {}
            feedback = {}
            annotated = frame.copy()
            h, w = frame.shape[:2]
            
            # Extract face box and landmarks
            box = face_data['facial_area']
            landmarks = face_data['landmarks']
            
            # Get face dimensions and position
            x, y, right, bottom = box
            face_w = right - x
            face_h = bottom - y
            face_center_x = x + face_w // 2
            face_center_y = y + face_h // 2
            
            # Calculate normalized positions
            center_x_ratio = face_center_x / w
            center_y_ratio = face_center_y / h
            face_height_ratio = face_h / h
            
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
            
            # 3. Tilt score using RetinaFace landmarks
            # RetinaFace gives: left_eye, right_eye, nose, left_mouth, right_mouth
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            
            # Calculate angle
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
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
            
            # 4. Draw face guide
            # Draw ideal face position rectangle
            ideal_h = int(h * self.ideal_face_height_ratio)
            ideal_w = ideal_h * 3 // 4  # Assuming face aspect ratio of 3:4
            ideal_x = int(w * self.ideal_face_center_x_ratio - ideal_w // 2)
            ideal_y = int(h * self.ideal_face_center_y_ratio - ideal_h // 2)
            
            # Draw ideal position rectangle
            cv2.rectangle(annotated, (ideal_x, ideal_y), (ideal_x + ideal_w, ideal_y + ideal_h),
                         (0, 255, 0), 2)
            
            # Draw horizontal eye line through eye landmarks
            eye_y = int((left_eye[1] + right_eye[1]) / 2)
            cv2.line(annotated, (x, eye_y), (right, eye_y), (0, 255, 0), 1)
            
            # Draw eye centers
            cv2.circle(annotated, (int(left_eye[0]), int(left_eye[1])), 5, (0, 0, 255), -1)
            cv2.circle(annotated, (int(right_eye[0]), int(right_eye[1])), 5, (0, 0, 255), -1)
            
            # Draw mouth centers if available
            left_mouth = landmarks.get("left_mouth")
            right_mouth = landmarks.get("right_mouth")
            if left_mouth is not None and right_mouth is not None:
                cv2.circle(annotated, (int(left_mouth[0]), int(left_mouth[1])), 5, (255, 0, 0), -1)
                cv2.circle(annotated, (int(right_mouth[0]), int(right_mouth[1])), 5, (255, 0, 0), -1)
                
                # Draw mouth line
                mouth_y = int((left_mouth[1] + right_mouth[1]) / 2)
                cv2.line(annotated, (int(left_mouth[0]), mouth_y), 
                         (int(right_mouth[0]), mouth_y), (255, 0, 0), 1)
            
            # 5. Calculate overall score
            weights = {"position": 0.4, "size": 0.3, "tilt": 0.3}
            overall_score = 0
            for metric, score in scores.items():
                overall_score += score * weights.get(metric, 0)
            overall_score = int(overall_score * 100)
            
            # Add overall score text
            cv2.putText(annotated, f"Position Score: {overall_score}%", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            
            # Draw reference lines for face center
            frame_center_x = w // 2
            frame_center_y = h // 2
            cv2.line(annotated, (frame_center_x, 0), (frame_center_x, h), (0, 255, 255), 1)
            cv2.line(annotated, (0, frame_center_y), (w, frame_center_y), (0, 255, 255), 1)
            
            return overall_score, feedback, annotated
    
    # Create face position evaluator instance
    face_evaluator = FacePositionEvaluator()
    
    # For positioning evaluation
    position_scores = []
    last_position_evaluation = time.time()
    show_position_evaluation = False  # Toggle with 'p' key
    
    # Thread to load RetinaFace after delay and process frames
    def retinaface_worker():
        nonlocal retinaface_enabled, retinaface_ready, retinaface_model
        nonlocal face_present, last_face_time, face_ids_counter, known_faces
        
        # Wait 10 seconds before loading RetinaFace
        time.sleep(10)
        
        print("Loading RetinaFace model...")
        try:
            # Load RetinaFace model
            retinaface_model = RetinaFace.build_model()
            retinaface_ready = True
            retinaface_enabled = True
            print("RetinaFace model loaded successfully")
        except Exception as e:
            print(f"Error loading RetinaFace: {e}")
            return
            
        # Process frames
        last_processed_time = time.time()
        min_process_interval = 0.03  # Max ~33 fps for processing
        
        # For periodic stats saving (every 5 minutes)
        last_stats_save_time = time.time()
        stats_save_interval = 300  # 5 minutes
        
        while not stop_threads:
            # Save tracking stats periodically
            if continuous_tracking and time.time() - last_stats_save_time > stats_save_interval:
                save_tracking_stats()
                last_stats_save_time = time.time()
            
            # Throttle processing rate to reduce lag
            current_time = time.time()
            if current_time - last_processed_time < min_process_interval:
                time.sleep(0.005)
                continue
                
            # Check if there are frames to process
            if frames_to_process:
                # Get the latest frame
                frame = frames_to_process.pop()
                frames_to_process.clear()  # Clear any backlog
                
                try:
                    # Apply any needed preprocessing
                    # Resize for faster processing while maintaining aspect ratio
                    height, width = frame.shape[:2]
                    if max(width, height) > 640:
                        scale = 640 / max(width, height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        small_frame = cv2.resize(frame, (new_width, new_height))
                    else:
                        small_frame = frame
                    
                    # Process with RetinaFace
                    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    faces = RetinaFace.detect_faces(rgb_frame, model=retinaface_model)
                    
                    # Update face present status
                    face_detected_now = isinstance(faces, dict) and len(faces) > 0
                    
                    # Update tracking statistics
                    tracking_stats["detection_count"] += 1
                    
                    if face_detected_now:
                        if not face_present:
                            # Face just appeared
                            face_present = True
                            print(f"Face detected at {datetime.datetime.now().strftime('%H:%M:%S')}")
                        
                        last_face_time = current_time
                        face_present_duration = current_time - last_face_time
                        tracking_stats["face_present_time"] += min_process_interval
                    else:
                        if face_present and (current_time - last_face_time > 2.0):  
                            # Face has been gone for more than 2 seconds
                            face_present = False
                            print(f"Face lost at {datetime.datetime.now().strftime('%H:%M:%S')}")
                        
                        face_absent_duration = current_time - last_face_time
                        tracking_stats["face_absent_time"] += min_process_interval
                    
                    # If we resized, adjust coordinates back to original frame size
                    if small_frame is not frame:
                        scale_back = width / new_width
                        if isinstance(faces, dict):
                            for face_idx in faces:
                                # Adjust face box
                                box = faces[face_idx]['facial_area']
                                faces[face_idx]['facial_area'] = [
                                    int(box[0] * scale_back),
                                    int(box[1] * scale_back),
                                    int(box[2] * scale_back),
                                    int(box[3] * scale_back)
                                ]
                                
                                # Adjust landmarks
                                for landmark in faces[face_idx]['landmarks']:
                                    faces[face_idx]['landmarks'][landmark] = [
                                        faces[face_idx]['landmarks'][landmark][0] * scale_back,
                                        faces[face_idx]['landmarks'][landmark][1] * scale_back
                                    ]
                    
                    # Assign persistent IDs to faces for continuous tracking
                    if isinstance(faces, dict) and len(faces) > 0:
                        # Try to match with known faces
                        for face_idx, face_data in faces.items():
                            # Extract face info for matching
                            box = face_data['facial_area']
                            landmarks = face_data['landmarks']
                            
                            # Try to find matching face in known faces
                            matched = False
                            for known_id, known_face in known_faces.items():
                                if is_same_face(box, landmarks, known_face):
                                    # Update this face with the known ID
                                    face_data['persistent_id'] = known_id
                                    # Update the known face data with current info
                                    known_faces[known_id] = {
                                        'box': box,
                                        'landmarks': landmarks,
                                        'last_seen': current_time
                                    }
                                    matched = True
                                    break
                            
                            # If no match found, add as new face
                            if not matched:
                                new_id = f"face_{face_ids_counter}"
                                face_ids_counter += 1
                                face_data['persistent_id'] = new_id
                                known_faces[new_id] = {
                                    'box': box,
                                    'landmarks': landmarks,
                                    'first_seen': current_time,
                                    'last_seen': current_time
                                }
                    
                    # Store detections with timestamp
                    face_detections.append((frame.copy(), faces, current_time))
                    
                    # Keep only the last few detections (for smoothing)
                    max_detections = 3
                    if len(face_detections) > max_detections:
                        face_detections.pop(0)
                        
                    # Update last processed time
                    last_processed_time = current_time
                        
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    
            # Sleep to avoid high CPU usage
            time.sleep(0.005)
    
    def is_same_face(box1, landmarks1, known_face):
        """Determine if a detected face is likely the same as a known face"""
        box2 = known_face['box']
        landmarks2 = known_face['landmarks']
        
        # Calculate center points of face boxes
        center1 = ((box1[0] + box1[2])/2, (box1[1] + box1[3])/2)
        center2 = ((box2[0] + box2[2])/2, (box2[1] + box2[3])/2)
        
        # Calculate distance between centers
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        
        # If centers are very close, it's likely the same face
        max_distance = (box2[2] - box2[0]) * 0.5  # Half the width of the known face
        
        return distance < max_distance
    
    def save_tracking_stats():
        """Save tracking statistics to file"""
        if not continuous_tracking:
            return
            
        # Update total tracking time
        tracking_stats["total_tracking_time"] = time.time() - start_time
        
        # Save to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = os.path.join(log_dir, f"tracking_stats_{timestamp}.pkl")
        
        try:
            with open(stats_file, 'wb') as f:
                pickle.dump(tracking_stats, f)
            print(f"Saved tracking stats to {stats_file}")
        except Exception as e:
            print(f"Error saving tracking stats: {e}")
    
    def take_auto_snapshot():
        """Take automatic snapshot if face is detected"""
        if not face_present:
            return None
            
        # Make sure we have at least one detection
        if not face_detections:
            return None
            
        # Take snapshot from latest frame with face
        _, faces, _ = face_detections[-1]
        if isinstance(faces, dict) and len(faces) > 0:
            snapshot_file = os.path.join(log_dir, f"auto_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(snapshot_file, display_frame)
            tracking_stats["snapshot_count"] += 1
            print(f"Auto snapshot saved: {snapshot_file}")
            return snapshot_file
        
        return None
    
    # Start RetinaFace thread
    retinaface_thread = threading.Thread(target=retinaface_worker)
    retinaface_thread.daemon = True
    retinaface_thread.start()
    
    # For activity display
    activity_log = []
    max_log_entries = 5
    
    # Initialize frame counter and tracking interval
    frame_counter = 0
    tracking_interval = 5  # Process every 5th frame
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting...")
            break
            
        # Increment frame counter
        frame_counter += 1
        
        # Calculate FPS
        fps_counter += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:
            fps = fps_counter / elapsed_time
            fps_counter = 0
            fps_start_time = time.time()
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Add FPS to the frame
        if show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 0), 2)
        
        # Show countdown to RetinaFace activation if not yet enabled
        if not retinaface_enabled:
            seconds_passed = time.time() - start_time
            seconds_left = max(0, 10 - int(seconds_passed))
            countdown_text = f"RetinaFace activating in: {seconds_left}s"
            cv2.putText(display_frame, countdown_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Process RetinaFace detections if ready
        if retinaface_ready and retinaface_enabled:
            # Queue frame for processing only on specific intervals
            if frame_counter % tracking_interval == 0:  # Process every Nth frame
                frames_to_process.append(frame.copy())
                last_processed_frame = frame.copy()
            
            # Check if time for auto snapshot
            if continuous_tracking and time.time() - last_auto_snapshot_time > auto_snapshot_interval:
                snapshot = take_auto_snapshot()
                if snapshot:
                    # Add to activity log
                    activity_log.append(f"{datetime.datetime.now().strftime('%H:%M:%S')}: Auto snapshot captured")
                    if len(activity_log) > max_log_entries:
                        activity_log.pop(0)
                last_auto_snapshot_time = time.time()
            
            # Check if we have face detections
            if face_detections:
                # Get latest detection
                latest_frame, faces, detection_time = face_detections[-1]
                
                # Calculate detection delay
                detection_delay = time.time() - detection_time
                
                # Show 24/7 tracking status with time stats
                if face_present:
                    status_color = (0, 255, 0)  # Green for active
                    time_present = time.time() - last_face_time
                    status_text = f"Face Tracked: {int(time_present)}s"
                else:
                    status_color = (0, 0, 255)  # Red for inactive
                    time_absent = time.time() - last_face_time
                    status_text = f"No Face: {int(time_absent)}s"
                
                cv2.putText(display_frame, status_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Draw frame counter to see the tracking interval
                interval_text = f"Frame: {frame_counter} (Processing: {frame_counter % tracking_interval == 0})"
                cv2.putText(display_frame, interval_text, (display_frame.shape[1] - 280, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw face detections
                if isinstance(faces, dict) and len(faces) > 0:
                    # Show RetinaFace active status
                    tracking_text = f"Tracking Active (Delay: {detection_delay*1000:.0f}ms)"
                    cv2.putText(display_frame, tracking_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Apply smoothing if we have multiple detections
                    smoothed_faces = {}
                    
                    # Process each detected face
                    for face_idx, face_data in faces.items():
                        # Try to apply motion prediction based on detection delay
                        # Get face box
                        box = face_data['facial_area'].copy()
                        x, y, right, bottom = box
                        
                        # Apply smoothing when we have enough history
                        if len(face_detections) >= 2:
                            # Find the same face in previous frames for smoothing
                            previous_boxes = []
                            for i in range(len(face_detections)-1, -1, -1):
                                prev_frame, prev_faces, prev_time = face_detections[i]
                                if isinstance(prev_faces, dict) and face_idx in prev_faces:
                                    prev_box = prev_faces[face_idx]['facial_area']
                                    previous_boxes.append(prev_box)
                            
                            # If we have previous positions, smooth the current position
                            if previous_boxes:
                                # Calculate average movement direction
                                if len(previous_boxes) >= 2:
                                    # Calculate velocity
                                    oldest_box = previous_boxes[-1]
                                    newest_box = previous_boxes[0]
                                    
                                    dx = (newest_box[0] - oldest_box[0]) / len(previous_boxes)
                                    dy = (newest_box[1] - oldest_box[1]) / len(previous_boxes)
                                    dw = (newest_box[2] - newest_box[0] - (oldest_box[2] - oldest_box[0])) / len(previous_boxes)
                                    dh = (newest_box[3] - newest_box[1] - (oldest_box[3] - oldest_box[1])) / len(previous_boxes)
                                    
                                    # Calculate extra prediction based on frame skip
                                    # Add prediction factor for skipped frames
                                    skip_factor = (frame_counter % tracking_interval) / tracking_interval
                                    total_prediction = detection_delay * 30 + skip_factor  # 30fps assumption
                                    
                                    # Predict position based on delay and frame skip
                                    prediction_factor = min(1.5, total_prediction)  # Cap at 1.5x
                                    
                                    x = int(x + dx * prediction_factor)
                                    y = int(y + dy * prediction_factor)
                                    right = int(right + dx * prediction_factor + dw * prediction_factor)
                                    bottom = int(bottom + dy * prediction_factor + dh * prediction_factor)
                        
                        # Calculate width and height
                        w = right - x
                        h = bottom - y
                        
                        # Get persistent ID if available
                        face_id = face_data.get('persistent_id', f"face_{face_idx}")
                        
                        # Set color based on tracking duration
                        if face_id in known_faces:
                            tracking_duration = time.time() - known_faces[face_id].get('first_seen', time.time())
                            # Color changes from yellow to green over time
                            if tracking_duration < 30:  # Less than 30 seconds
                                color = (0, 165, 255)  # Orange
                            elif tracking_duration < 300:  # Less than 5 minutes
                                color = (0, 255, 255)  # Yellow
                            else:
                                color = (0, 255, 0)  # Green
                        else:
                            color = (0, 0, 255)  # Red for new/unknown faces
                        
                        # Draw rectangle around face with predicted position
                        cv2.rectangle(display_frame, (x, y), (right, bottom), color, 2)
                        
                        # Draw landmarks with prediction
                        landmarks = face_data['landmarks']
                        for landmark_name, landmark_pos in landmarks.items():
                            # Apply same prediction to landmarks
                            if len(face_detections) >= 2 and previous_boxes:
                                lm_x, lm_y = landmark_pos
                                lm_x = int(lm_x + dx * prediction_factor)
                                lm_y = int(lm_y + dy * prediction_factor)
                                pos = (lm_x, lm_y)
                            else:
                                pos = (int(landmark_pos[0]), int(landmark_pos[1]))
                                
                            # Draw landmark with different colors
                            if landmark_name == "left_eye":
                                color = (0, 0, 255)  # Red
                            elif landmark_name == "right_eye":
                                color = (0, 0, 255)  # Red
                            elif landmark_name == "nose":
                                color = (255, 0, 0)  # Blue
                            else:
                                color = (0, 255, 255)  # Yellow
                                
                            cv2.circle(display_frame, pos, 3, color, -1)
                        
                        # Show persistent ID and tracking duration
                        if face_id in known_faces:
                            tracking_duration = time.time() - known_faces[face_id].get('first_seen', time.time())
                            duration_text = f"{int(tracking_duration)}s"
                            cv2.putText(display_frame, f"ID: {face_id.split('_')[1]} ({duration_text})", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show notification when RetinaFace first activates
            if not show_notification:
                notification_time = time.time()
                show_notification = True
        
        # Show activation notification for 3 seconds
        if show_notification and time.time() - notification_time < 3:
            # Create semi-transparent overlay
            overlay = display_frame.copy()
            h, w = overlay.shape[:2]
            cv2.rectangle(overlay, (w//4, h//2-30), (3*w//4, h//2+30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
            
            # Add notification text
            cv2.putText(display_frame, "24/7 Face Tracking Active", (w//4+20, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Show activity log on screen
        if continuous_tracking and activity_log:
            log_y = display_frame.shape[0] - 60
            cv2.putText(display_frame, "Recent Activity:", (10, log_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, log_entry in enumerate(activity_log):
                cv2.putText(display_frame, log_entry, (10, log_y + 20 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Display timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display session duration
        session_time = int(time.time() - start_time)
        hours = session_time // 3600
        minutes = (session_time % 3600) // 60
        seconds = session_time % 60
        session_text = f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}"
        cv2.putText(display_frame, session_text, (10, display_frame.shape[0] - 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Process face position evaluation if enabled
        if show_position_evaluation and retinaface_enabled and face_detections:
            # Get latest detection
            latest_frame, faces, detection_time = face_detections[-1]
            
            # If faces detected, evaluate position
            if isinstance(faces, dict) and len(faces) > 0:
                # Get first face for evaluation
                face_idx = sorted(faces.keys())[0]
                face_data = faces[face_idx]
                
                # Evaluate face position
                position_score, feedback, annotated_position = face_evaluator.evaluate_face_position(
                    latest_frame, face_data
                )
                
                # Store score for tracking
                position_scores.append(position_score)
                if len(position_scores) > 10:
                    position_scores.pop(0)
                
                # Use the annotated frame for display
                display_frame = annotated_position
        
        # Display the resulting frame
        cv2.imshow("24/7 Tracking Monitor", display_frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Break loop on 'q' key
        if key == ord('q'):
            print("Exiting camera monitor...")
            break
            
        # Save snapshot on 's' key
        elif key == ord('s'):
            snapshot_file = os.path.join(log_dir, f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(snapshot_file, display_frame)
            print(f"Snapshot saved as {snapshot_file}")
            
            # Add to activity log
            activity_log.append(f"{datetime.datetime.now().strftime('%H:%M:%S')}: Manual snapshot taken")
            if len(activity_log) > max_log_entries:
                activity_log.pop(0)
            
        # Toggle RetinaFace on 'r' key
        elif key == ord('r'):
            if retinaface_ready:
                retinaface_enabled = not retinaface_enabled
                status = "enabled" if retinaface_enabled else "disabled"
                print(f"RetinaFace {status}")
                
                # Add to activity log
                activity_log.append(f"{datetime.datetime.now().strftime('%H:%M:%S')}: RetinaFace {status}")
                if len(activity_log) > max_log_entries:
                    activity_log.pop(0)
                
                # Show notification if enabled
                if retinaface_enabled:
                    notification_time = time.time()
                    show_notification = True
        
        # Add toggle handler for position evaluation mode
        elif key == ord('p'):
            show_position_evaluation = not show_position_evaluation
            status = "enabled" if show_position_evaluation else "disabled"
            print(f"Position evaluation {status}")
            
            # Add to activity log
            activity_log.append(f"{datetime.datetime.now().strftime('%H:%M:%S')}: Position evaluation {status}")
            if len(activity_log) > max_log_entries:
                activity_log.pop(0)
    
    # Final stats saving
    save_tracking_stats()
    
    # Clean up
    stop_threads = True
    if retinaface_thread.is_alive():
        retinaface_thread.join(timeout=1.0)
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laptop Camera Monitor with RetinaFace and Face Positioning")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera resolution width")
    parser.add_argument("--height", type=int, default=720, help="Camera resolution height")
    parser.add_argument("--no-fps", action="store_false", dest="show_fps", help="Hide FPS counter")
    parser.add_argument("--positioning", action="store_true", help="Enable face positioning evaluation by default")
    
    args = parser.parse_args()
    
    laptop_camera_monitor(
        camera_index=args.camera,
        resolution=(args.width, args.height),
        show_fps=args.show_fps
    )