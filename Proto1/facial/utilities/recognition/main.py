from .base import FaceRecognizer

def main():
    """Main function to run the face recognition utility"""
    print("=== Face Recognition Utility ===")
    recognizer = FaceRecognizer()
    recognizer.recognition_loop()

if __name__ == "__main__":
    main()