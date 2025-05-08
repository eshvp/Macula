import numpy as np
import cv2

class PosePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def preprocess_frame(self, frame):
        """Preprocess frame for pose detection"""
        # Resize while maintaining aspect ratio
        h, w = frame.shape[:2]
        scale = min(self.target_size[0]/w, self.target_size[1]/h)
        new_size = (int(w*scale), int(h*scale))
        
        resized = cv2.resize(frame, new_size)
        
        # Create blank canvas
        canvas = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        # Center the image
        y_offset = (self.target_size[1] - new_size[1]) // 2
        x_offset = (self.target_size[0] - new_size[0]) // 2
        canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
        
        return canvas, (x_offset, y_offset, scale)