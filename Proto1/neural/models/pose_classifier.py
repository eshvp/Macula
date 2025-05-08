import tensorflow as tf
import numpy as np

class PoseClassifier(tf.keras.Model):
    def __init__(self, num_keypoints=14, num_poses=5):
        super(PoseClassifier, self).__init__()
        
        # Feature extraction
        self.conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D()
        
        # Keypoint detection branch
        self.keypoint_conv = tf.keras.layers.Conv2D(512, 1, activation='relu')
        self.keypoint_output = tf.keras.layers.Dense(num_keypoints * 2)  # x,y coordinates
        
        # Pose classification branch
        self.pose_flatten = tf.keras.layers.Flatten()
        self.pose_dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.pose_dropout = tf.keras.layers.Dropout(0.5)
        self.pose_output = tf.keras.layers.Dense(num_poses, activation='softmax')
        
    def call(self, x):
        # Shared features
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        features = self.pool3(x)
        
        # Keypoint detection
        kp = self.keypoint_conv(features)
        keypoints = self.keypoint_output(kp)
        
        # Pose classification
        pose = self.pose_flatten(features)
        pose = self.pose_dense1(pose)
        pose = self.pose_dropout(pose)
        pose_class = self.pose_output(pose)
        
        return keypoints, pose_class
        
    def detect_pose(self, frame):
        """Detect pose from a frame"""
        # Preprocess frame
        x = tf.image.resize(frame, (224, 224))
        x = x / 255.0
        x = tf.expand_dims(x, 0)
        
        # Get predictions
        keypoints, pose = self(x)
        
        return (
            tf.reshape(keypoints[0], (-1, 2)).numpy(),
            tf.argmax(pose[0]).numpy()
        )
        
    def get_confidence(self):
        """Get confidence of last prediction"""
        return self.pose_output.weights[0].numpy().max()