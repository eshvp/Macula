import dlib
import cv2
from utilities.distance import calculate_distance

# Load pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Known face width in cm (average human face width)
KNOWN_FACE_WIDTH_CM = 14.0

# Focal length in pixels (this is an approximation)
FOCAL_LENGTH_PIXELS = 600

def run_live_detection():
    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        # Draw rectangles around detected faces
        for face in faces:  # Added the for loop
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate and display distance for each face
            distance = calculate_distance(w, KNOWN_FACE_WIDTH_CM, FOCAL_LENGTH_PIXELS)
            cv2.putText(frame, f"Distance: {distance:.2f} cm",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Real-Time Face Detection", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()