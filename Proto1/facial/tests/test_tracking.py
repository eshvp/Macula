import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_opencv_capabilities():
    """Test OpenCV version and available tracking algorithms"""
    print(f"\nOpenCV Version: {cv2.__version__}")
    print(f"OpenCV Path: {cv2.__file__}")
    
    # Test contrib modules
    has_contrib = hasattr(cv2, 'TrackerCSRT_create')
    print(f"OpenCV Contrib Modules: {'Available' if has_contrib else 'Not Available'}")
    
    # Test legacy trackers
    legacy_trackers = {
        'CSRT': 'cv2.legacy.TrackerCSRT_create',
        'KCF': 'cv2.legacy.TrackerKCF_create',
        'MOSSE': 'cv2.legacy.TrackerMOSSE_create',
        'MIL': 'cv2.legacy.TrackerMIL_create'
    }
    
    print("\nChecking Legacy Trackers:")
    available_trackers = {}
    
    for name, create_func in legacy_trackers.items():
        try:
            # Try to create tracker using legacy module
            if hasattr(cv2, 'legacy'):
                tracker_create = eval(create_func)
                tracker = tracker_create()
                available_trackers[name] = tracker_create
                print(f"{name}: Available (Legacy)")
            else:
                # Try original location
                tracker_create = getattr(cv2, f'Tracker{name}_create', None)
                if tracker_create:
                    tracker = tracker_create()
                    available_trackers[name] = tracker_create
                    print(f"{name}: Available (Original)")
                else:
                    print(f"{name}: Not Available")
                    available_trackers[name] = None
        except Exception as e:
            print(f"{name}: Error ({str(e)})")
            available_trackers[name] = None
    
    return available_trackers

def run_tracking_test():
    """Run a live test of face tracking"""
    # Initialize camera with higher resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Create named window with proper flags
    window_name = "Tracking Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Get screen resolution
    screen_width = 1920  # Default to common resolution
    screen_height = 1080
    try:
        import ctypes
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
    except:
        print("Could not get screen resolution, using defaults")
    
    # Calculate window size (80% of screen)
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    
    # Set initial window size and position
    cv2.resizeWindow(window_name, window_width, window_height)
    x_pos = (screen_width - window_width) // 2
    y_pos = (screen_height - window_height) // 2
    cv2.moveWindow(window_name, x_pos, y_pos)

    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Get available trackers
    trackers = test_opencv_capabilities()
    current_tracker = None
    tracking = False
    bbox = None
    
    print("\nControls:")
    print("SPACE - Reset and detect face")
    print("T - Switch tracker")
    print("ESC/Q - Quit")
    
    tracker_names = list(trackers.keys())
    current_tracker_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Maintain aspect ratio while scaling
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        
        # Calculate dimensions that maintain aspect ratio
        if window_width / aspect_ratio <= window_height:
            display_width = window_width
            display_height = int(window_width / aspect_ratio)
        else:
            display_height = window_height
            display_width = int(window_height * aspect_ratio)
        
        # Resize frame
        frame = cv2.resize(frame, (display_width, display_height))
            
        # Display current tracker
        tracker_text = f"Tracker: {tracker_names[current_tracker_idx]}"
        text_size = cv2.getTextSize(tracker_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = 20
        text_y = text_size[1] + 20
        
        cv2.putText(frame, tracker_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if tracking and bbox:
            # Update tracker
            success, bbox = current_tracker.update(frame)
            if success:
                # Draw tracking box
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Lost Track", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                tracking = False
        
        # Show frame in the window
        cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to detect and start tracking
            try:
                # Detect face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Take the largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    bbox = (x, y, w, h)
                    
                    # Initialize tracker with detailed error checking
                    tracker_name = tracker_names[current_tracker_idx]
                    tracker_class = trackers[tracker_name]
                    
                    if tracker_class:
                        try:
                            current_tracker = tracker_class()
                            init_success = current_tracker.init(frame, bbox)
                            
                            if init_success:
                                tracking = True
                                print(f"\nSuccessfully initialized {tracker_name} tracker")
                            else:
                                print(f"\nFailed to initialize {tracker_name} tracker")
                        except Exception as e:
                            print(f"\nError creating {tracker_name} tracker: {str(e)}")
                    else:
                        print(f"\nError: {tracker_name} tracker not available")
            except Exception as e:
                print(f"\nError during face detection/tracking: {str(e)}")
        
        elif key == ord('t'):  # Switch tracker
            current_tracker_idx = (current_tracker_idx + 1) % len(tracker_names)
            tracking = False
            print(f"\nSwitched to {tracker_names[current_tracker_idx]} tracker")
        
        elif key == 27 or key == ord('q'):  # ESC or q to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracking_test()