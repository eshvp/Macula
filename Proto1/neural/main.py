import os
import cv2
import time
import sys
from pathlib import Path

# Ensure neural module is in path
sys.path.append(str(Path(__file__).parent.parent))
from neural.functions.staticPose import StaticPoseCapture

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls')

def display_menu():
    """Display the main menu options."""
    clear_screen()
    print("=" * 40)
    print("       POSE DETECTION SYSTEM")
    print("=" * 40)
    print("1. Static Pose Detection")
    print("2. Exit")
    print("=" * 40)

def main():
    pose_capture = StaticPoseCapture()
    
    while True:
        display_menu()
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == '1':
            clear_screen()
            print("Static Pose Detection Selected")
            print("\nGet ready for pose capture!")
            print("Press 'q' at any time to cancel")
            
            # Capture pose using StaticPoseCapture
            frame = pose_capture.capture_pose(countdown_seconds=7)
            
            if frame is not None:
                # Save frame
                save_path = Path(__file__).parent / "captured_poses"
                save_path.mkdir(exist_ok=True)
                save_file = save_path / f"pose_{int(time.time())}.jpg"
                cv2.imwrite(str(save_file), frame)
                print(f"\nPose captured and saved to: {save_file}")
            
            input("\nPress Enter to return to main menu...")
            
        elif choice == '2':
            clear_screen()
            print("Thank you for using Pose Detection System")
            break
            
        else:
            print("\nInvalid choice. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()