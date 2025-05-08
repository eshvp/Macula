# Core.py Documentation

Core.py is the most bare bones version of the facial recognition software. It is saved here to reference for each new function being created when need be, allowing for a clean construction of each new function.

## Overview

Core.py is a foundational module in the facial recognition system that provides essential face detection, tracking, and position evaluation capabilities.

## Key Components

### DynamicFaceEvaluator Class

This class is the heart of the facial recognition engine:

- **Face Detection**: Uses dlib.get_frontal_face_detector() to identify faces in images
- **Landmark Detection**: Employs the 68-point facial landmark model to precisely map facial features
- **Face Tracking**: Implements tracking to maintain detection between frames without repeated full analysis
- **Position Evaluation**: Assesses face positioning based on parameters like:
  - Face height ratio (min: 0.25, max: 0.6)
  - Tilt threshold (10 degrees)

### Main Functions

- `initialize_tracking()`: Sets up tracking for a detected face
- `evaluate_face_position()`: Analyzes if a face is properly positioned
- `dynamic_face_position_loop()`: Runs continuous face detection in high-performance loop

## How It Works

The system initializes by loading the face detector and landmark predictor models. When processing frames:

1. It first attempts to use the tracker for efficiency
2. Periodically runs full face detection to refresh landmarks (every 30 frames)
3. Evaluates face position quality using size and orientation metrics
4. Provides feedback on position quality

## Applications

Core.py can be used for:

- **Real-time Face Detection**: Continuously locate faces in a video feed
- **Position Guidance**: Determine if a face is properly positioned for photos
- **Image Capture**: Automatically capture images when face position is optimal
- **Foundation for Advanced Features**: Serve as the base for more complex recognition tasks

The command-line interface allows running the system with different parameters:

```bash
python Core.py --camera 0 --width 1280 --height 720 --capture --threshold 80
```

This module provides a reference implementation that can be extended for more advanced facial recognition applications.