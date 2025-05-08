import cv2
import dlib
import os

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()


def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection coordinates
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # Calculate intersection area
    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area
    return 0.0


def get_face_accuracy(image, expected_face_location=None):
    """
    Calculate face detection accuracy using IoU or bounding box area
    Args:
        image: Input image
        expected_face_location: Tuple of (x, y, width, height) for ground truth
    Returns:
        float: IoU score or face area
    """
    if image is None:
        print("Error: Invalid image")
        return 0.0

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        print("Error: Failed to convert image to grayscale")
        return 0.0

    # Detect faces
    faces = detector(gray, 1)

    if len(faces) == 0:
        print("No faces detected!")
        return 0.0

    # Get the first detected face
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    print(f"Detected face at location: ({x}, {y}), Width: {w}, Height: {h}")

    # If ground truth is provided, calculate IoU
    if expected_face_location:
        print(f"Expected face location: {expected_face_location}")
        accuracy = calculate_iou((x, y, w, h), expected_face_location)
        print(f"IoU (Accuracy) Score: {accuracy:.2f}")
        return accuracy

    return w * h  # Return area if no expected location


def main():
    # Create test_images directory if it doesn't exist
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
        print("Created 'test_images' directory")

    image_path = os.path.join('test_images', 'sample.jpg')
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Please ensure 'sample.jpg' exists in the 'test_images' directory")
        return

    # Example ground truth bounding box
    expected_face_location = (100, 100, 150, 150)
    accuracy = get_face_accuracy(image, expected_face_location)
    print(f"Face detection accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()