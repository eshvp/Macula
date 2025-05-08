import cv2
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural.utils.preprocessing import PosePreprocessor
from neural.models.pose_classifier import PoseClassifier

class StaticPoseCapture:
    def __init__(self):
        self.preprocessor = PosePreprocessor()
        self.pose_classifier = PoseClassifier()
        self._load_model()
        
        # Enhanced visualization settings
        self.joint_color = (0, 255, 0)  # Green for joints
        self.skeleton_color = (255, 255, 0)  # Yellow for skeleton
        self.label_color = (255, 255, 255)  # White for labels
        self.joint_radius = 8  # Increased radius for better visibility
        self.line_thickness = 2
        
        # Add landmark visualization data
        self.landmark_info = {
            "Nose": {"color": (255, 0, 0), "radius": 10},      # Red
            "Neck": {"color": (255, 128, 0), "radius": 10},    # Orange
            "RShoulder": {"color": (0, 255, 255), "radius": 8}, # Cyan
            "LShoulder": {"color": (255, 255, 0), "radius": 8}, # Yellow
            "RElbow": {"color": (0, 255, 0), "radius": 8},     # Green
            "LElbow": {"color": (255, 0, 255), "radius": 8},   # Magenta
            "RWrist": {"color": (128, 0, 255), "radius": 8},   # Purple
            "LWrist": {"color": (0, 128, 255), "radius": 8},   # Light blue
            "RHip": {"color": (255, 128, 128), "radius": 8},   # Pink
            "LHip": {"color": (128, 255, 128), "radius": 8},   # Light green
            "RKnee": {"color": (128, 128, 255), "radius": 8},  # Light purple
            "LKnee": {"color": (255, 255, 128), "radius": 8},  # Light yellow
            "RAnkle": {"color": (128, 255, 255), "radius": 8}, # Light cyan
            "LAnkle": {"color": (255, 128, 255), "radius": 8}  # Light magenta
        }
        
        # Define joint names as class attribute
        self.joint_names = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
            "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
            "RAnkle", "LHip", "LKnee", "LAnkle"
        ]

    def _load_model(self):
        """Load pre-trained model if available"""
        model_path = Path(__file__).parent.parent / "models/saved_models/pose_model.h5"
        if model_path.exists():
            self.pose_classifier.load_weights(str(model_path))

    def capture_pose(self, countdown_seconds=7):
        """Capture pose after countdown"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        start_time = time.time()
        captured_frame = None

        print("\nPrepare for pose capture...")
        print("Position yourself and stay still")

        final_display = None  # Store the final annotated frame
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not access camera")
                break

            current_time = time.time()
            elapsed = current_time - start_time
            remaining = max(0, countdown_seconds - elapsed)

            display = frame.copy()

            if remaining > 0:
                # Show countdown
                cv2.putText(display, f"Capturing in: {int(remaining)}s", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.joint_color, 2)
            else:
                if captured_frame is None:
                    captured_frame = frame.copy()
                    print("\nPose captured!")
                    
                    # Process frame and detect pose
                    processed, transform = self.preprocessor.preprocess_frame(captured_frame)
                    keypoints, pose = self.pose_classifier.detect_pose(processed)
                    confidence = self.pose_classifier.get_confidence()
                    
                    # Debug info
                    print(f"Number of keypoints detected: {len(keypoints)}")
                    print(f"Number of joint names: {len(self.joint_names)}")
                    
                    # Create final visualization
                    final_display = captured_frame.copy()
                    
                    # Adjust and draw keypoints
                    x_offset, y_offset, scale = transform
                    adjusted_keypoints = []
                    
                    # Only process valid keypoints
                    for i, (x, y) in enumerate(keypoints):
                        if i >= len(self.joint_names):
                            break
                            
                        orig_x = int((x - x_offset) / scale)
                        orig_y = int((y - y_offset) / scale)
                        adjusted_keypoints.append((orig_x, orig_y))
                        
                        # Draw enhanced landmark
                        self._draw_landmarks(
                            final_display, 
                            (orig_x, orig_y),
                            self.joint_names[i],
                            orig_x,
                            orig_y
                        )

                    # Draw skeleton only if we have enough keypoints
                    if len(adjusted_keypoints) >= 14:  # Expected number of joints
                        self._draw_skeleton(final_display, adjusted_keypoints)
                    
                    # Add confidence score
                    cv2.putText(final_display, f"Confidence: {confidence:.2f}", 
                               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, self.skeleton_color, 2)

            # Display appropriate frame
            cv2.imshow("Pose Capture", final_display if final_display is not None else display)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or (
                final_display is not None and elapsed > countdown_seconds + 3):
                break

        cap.release()
        cv2.destroyAllWindows()
        return final_display if final_display is not None else captured_frame

    def _draw_skeleton(self, frame, keypoints):
        """Draw skeleton connections between keypoints"""
        # Define connections between keypoints
        connections = [
            (0, 1),  # nose to neck
            (1, 2), (2, 3), (3, 4),  # right arm
            (1, 5), (5, 6), (6, 7),  # left arm
            (1, 8), (8, 9), (9, 10),  # right leg
            (1, 11), (11, 12), (12, 13)  # left leg
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = tuple(map(int, keypoints[start_idx]))
                end_point = tuple(map(int, keypoints[end_idx]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    def _draw_landmarks(self, frame, keypoint, joint_name, x, y):
        """Draw a single landmark with enhanced visualization"""
        info = self.landmark_info[joint_name]
        
        # Draw filled circle for joint
        cv2.circle(frame, (x, y), info["radius"], info["color"], -1)
        
        # Draw white border around joint
        cv2.circle(frame, (x, y), info["radius"], (255, 255, 255), 1)
        
        # Add label with background
        label = f"{joint_name}: ({x}, {y})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Draw semi-transparent background for label
        bg_pts = np.array([
            [x + 5, y - text_height - 5],
            [x + text_width + 10, y - text_height - 5],
            [x + text_width + 10, y + 5],
            [x + 5, y + 5]
        ], np.int32)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [bg_pts], (0, 0, 0))
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw text
        cv2.putText(frame, label, (x + 7, y), 
                    font, font_scale, self.label_color, thickness)

def main():
    """Test the pose capture"""
    capture = StaticPoseCapture()
    annotated_frame = capture.capture_pose()
    if annotated_frame is not None:
        save_path = Path(__file__).parent.parent / "captured_poses"
        save_path.mkdir(exist_ok=True)
        cv2.imwrite(str(save_path / f"pose_annotated_{int(time.time())}.jpg"), annotated_frame)
        print(f"Annotated frame saved to {save_path}")

if __name__ == "__main__":
    main()