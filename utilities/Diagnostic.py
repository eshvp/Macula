# Save this as check_opencv_trackers.py
import cv2
import sys
import importlib

def check_tracker_availability():
    """Test all possible ways to create a CSRT tracker"""
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV path: {cv2.__file__}")
    print("\nChecking tracker availability...\n")
    
    methods = [
        ("cv2.TrackerCSRT_create()", 
            "try:\n    tracker = cv2.TrackerCSRT_create()\n    return True\nexcept:\n    return False"),
        
        ("cv2.TrackerCSRT.create()", 
            "try:\n    tracker = cv2.TrackerCSRT.create()\n    return True\nexcept:\n    return False"),
        
        ("cv2.legacy.TrackerCSRT_create()", 
            "try:\n    tracker = cv2.legacy.TrackerCSRT_create()\n    return True\nexcept:\n    return False"),
        
        ("direct legacy import", 
            "try:\n    from cv2 import legacy\n    tracker = legacy.TrackerCSRT_create()\n    return True\nexcept:\n    return False"),
        
        ("from legacy.tracker_csrt import TrackerCSRT", 
            "try:\n    from cv2.legacy.tracker_csrt import TrackerCSRT\n    tracker = TrackerCSRT()\n    return True\nexcept:\n    return False")
    ]
    
    success_method = None
    
    for name, code in methods:
        print(f"Testing: {name}")
        try:
            result = eval(compile(code, "<string>", "exec"))
            if result:
                print(f"✓ SUCCESS: {name} works!\n")
                success_method = name
            else:
                print(f"✗ FAILED: {name} did not work\n")
        except Exception as e:
            print(f"✗ ERROR: {name} failed with: {e}\n")
    
    # Check if legacy module exists
    print("Checking for cv2.legacy module:")
    has_legacy = hasattr(cv2, 'legacy')
    print(f"{'✓' if has_legacy else '✗'} cv2.legacy module {'exists' if has_legacy else 'does not exist'}")
    
    if has_legacy:
        print("\nListing available trackers in legacy module:")
        try:
            legacy_dir = dir(cv2.legacy)
            trackers = [item for item in legacy_dir if 'Tracker' in item]
            for tracker in trackers:
                print(f"- {tracker}")
        except Exception as e:
            print(f"Error listing legacy trackers: {e}")
    
    # Recommendation
    if success_method:
        print(f"\nRECOMMENDED METHOD: Use {success_method}")
    else:
        print("\nNo working tracker method found. Try installing opencv-contrib-python:")
        print("pip uninstall opencv-python")
        print("pip install opencv-contrib-python==4.5.5.64")

if __name__ == "__main__":
    check_tracker_availability()