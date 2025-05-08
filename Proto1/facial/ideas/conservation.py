import cv2
import time
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utility.recognition import OptimizedFaceRecognizer

class SmartSecurityCamera:
    """Camera system that only runs facial recognition when motion is detected"""
    
    def __init__(self, motion_threshold=2500, cooldown_seconds=3.0):
        """Initialize with configurable sensitivity"""
        # State management
        self.environment_state = "static"  # "static" or "dynamic"
        self.last_motion_time = 0
        self.cooldown_period = cooldown_seconds
        
        # Motion detection
        self.motion_threshold = motion_threshold
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False)
            
        # Components - initialize lazily to save resources
        self._face_recognizer = None
        
        # Performance tracking
        self.total_frames = 0
        self.recognition_frames = 0
    
    @property
    def face_recognizer(self):
        """Lazy initialization of face recognizer"""
        if self._face_recognizer is None:
            print("Initializing facial recognition system...")
            self._face_recognizer = OptimizedFaceRecognizer()
        return self._face_recognizer
    
    def detect_motion(self, frame):
        """Detect if significant motion exists in the frame"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove noise with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Count non-zero pixels in the mask (motion amount)
        motion_level = np.count_nonzero(fg_mask)
        
        # Create motion heatmap for visualization
        motion_display = cv2.applyColorMap(
            cv2.convertScaleAbs(fg_mask, alpha=0.8), cv2.COLORMAP_HOT)
            
        return motion_level > self.motion_threshold, motion_level, motion_display
    
    def update_environment_state(self, frame):
        """Update the environment state based on motion detection"""
        has_motion, motion_level, motion_display = self.detect_motion(frame)
        current_time = time.time()
        
        # State transition logic
        if has_motion:
            # Motion detected - set to dynamic
            if self.environment_state == "static":
                print("Motion detected! Activating facial recognition...")
            self.environment_state = "dynamic"
            self.last_motion_time = current_time
        elif current_time - self.last_motion_time > self.cooldown_period:
            # No motion for a while - switch to static
            if self.environment_state == "dynamic":
                print("Environment static. Deactivating facial recognition.")
            self.environment_state = "static"
            
        return motion_display, motion_level
    
    def process_frame(self, frame):
        """Process a single frame with dynamic recognition"""
        self.total_frames += 1
        motion_display, motion_level = self.update_environment_state(frame)
        
        # Prepare display frame (start with original)
        display_frame = frame.copy()
        result = None
        
        # Show environment state
        state_color = (0, 255, 0) if self.environment_state == "dynamic" else (0, 0, 255)
        cv2.putText(display_frame, f"State: {self.environment_state.upper()}", 
                  (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv2.putText(display_frame, f"Motion: {motion_level}", 
                  (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Only perform recognition in dynamic state
        if self.environment_state == "dynamic":
            try:
                self.recognition_frames += 1
                result, confidence, message, annotated_frame = self.face_recognizer.recognize_face(frame)
                
                # Use the annotated frame from recognition
                display_frame = annotated_frame
                
                # Re-add state indicator (may have been overwritten by recognition)
                cv2.putText(display_frame, f"State: {self.environment_state.upper()}", 
                          (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                          
            except Exception as e:
                cv2.putText(display_frame, f"Recognition error: {str(e)}", 
                         (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Small motion heatmap in corner
        h, w = motion_display.shape[:2]
        small_motion = cv2.resize(motion_display, (w//4, h//4))
        display_frame[20:20+h//4, display_frame.shape[1]-w//4-20:display_frame.shape[1]-20] = small_motion
        
        # Add efficiency stats
        efficiency = 100 * (1 - (self.recognition_frames / max(1, self.total_frames)))
        cv2.putText(display_frame, f"Resource saving: {efficiency:.1f}%", 
                  (20, display_frame.shape[0] - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame, result
    
    def run(self, camera_index=0):
        """Run the smart camera system"""
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Smart Security Camera Active")
        print("- 'q' to quit")
        print("- 's' to toggle state manually")
        
        # FPS calculation variables
        fps = 0
        frame_count = 0
        fps_start = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read error")
                break
                
            # Process frame
            display_frame, recognition_result = self.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()
            
            # Display FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                      (display_frame.shape[1] - 120, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Smart Security Camera", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Manual state toggle
                self.environment_state = "dynamic" if self.environment_state == "static" else "static"
                print(f"Manually set state to: {self.environment_state}")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()