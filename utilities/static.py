def run_static_image():
    import dlib
    import cv2
    import time

    # Start total time measurement
    total_start_time = time.time()

    # Load pre-trained face detector
    detector = dlib.get_frontal_face_detector()

    # Load the image
    load_start_time = time.time()
    image_path = "testimg/P1.jpg"  # Replace with your image path
    image = cv2.imread(image_path)
    load_time = time.time() - load_start_time

    # Check if image is loaded correctly
    if image is None:
        print("Error: Image not found!")
        return

    # Convert to grayscale
    preprocess_start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocess_time = time.time() - preprocess_start_time

    # Detect faces
    detection_start_time = time.time()
    faces = detector(gray)
    detection_time = time.time() - detection_start_time

    # Calculate detection time in milliseconds
    detection_ms = detection_time * 1000
    
    # Print detection metrics to console
    print(f"\nFace Detection Results:")
    print(f"Detection Time: {detection_ms:.1f}ms")
    print(f"Faces Detected: {len(faces)}")

    # Draw rectangles around detected faces and add detection time text
    drawing_start_time = time.time()
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add vertical line down the center of face
        center_x = x + w//2
        cv2.line(image, (center_x, y), (center_x, y + h), (0, 0, 255), 2)  # Red vertical line
        
        # Add horizontal line across the middle of face
        center_y = y + h//2
        cv2.line(image, (x, center_y), (x + w, center_y), (0, 0, 255), 2)  # Red horizontal line
        
        # Add horizontal line between eyes (approximately 1/3 from top of face)
        eye_level_y = y + (h // 3)
        cv2.line(image, (x, eye_level_y), (x + w, eye_level_y), (0, 0, 255), 2)  # Red eye line
        
        # Add detection time text above the face rectangle
        text = f"Detection Time: {detection_time:.2f}s"
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    drawing_time = time.time() - drawing_start_time

    # Calculate total processing time
    total_time = time.time() - total_start_time

    # Create metrics text
    metrics_text = [
        f"Face Detection: {detection_ms:.1f}ms",
        f"Faces Detected: {len(faces)}"
    ]

    # Add metrics to the image
    y_position = 60  # Starting position
    for text in metrics_text:
        cv2.putText(image, text, (10, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)  # Changed color to blue (255,0,0)
        y_position += 60  # Increased spacing for better readability

    # Create a resizable window
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    
    # Get screen dimensions
    screen_width = cv2.getWindowImageRect("Face Detection")[2]
    screen_height = cv2.getWindowImageRect("Face Detection")[3]
    
    # Calculate aspect ratio
    img_height, img_width = image.shape[:2]
    aspect_ratio = img_width / img_height
    
    # Calculate new dimensions to fit screen while maintaining aspect ratio
    if img_width > screen_width or img_height > screen_height:
        if screen_width / screen_height > aspect_ratio:
            new_height = screen_height
            new_width = int(screen_height * aspect_ratio)
        else:
            new_width = screen_width
            new_height = int(screen_width / aspect_ratio)
        
        # Resize window to fit screen
        cv2.resizeWindow("Face Detection", new_width, new_height)
    
    # Display the result
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
