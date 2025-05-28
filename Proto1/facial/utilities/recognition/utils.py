from .base import FaceRecognizer
from .optimized import OptimizedFaceRecognizer

def get_recognition_model():
    """Returns a configured face recognition model for use by other modules"""
    try:
        recognizer = OptimizedFaceRecognizer()
        print("Using optimized face recognizer")
    except Exception as e:
        print(f"Falling back to standard recognizer: {e}")
        recognizer = FaceRecognizer()
    return recognizer

def recognize_face(face_image, recognition_model=None):
    """Recognize a face and return name and confidence score
    
    Args:
        face_image: Image containing a face to recognize
        recognition_model: Optional pre-loaded recognition model
        
    Returns:
        Tuple of (name, confidence_score)
    """
    if recognition_model is None:
        recognition_model = get_recognition_model()
    
    # Get recognition result (expecting a tuple of result, score, message, annotated_image)
    result, score, message, _ = recognition_model.recognize_face(face_image)
    
    # Extract name if result is a dictionary 
    if isinstance(result, dict):
        if 'first_name' in result and 'last_name' in result:
            name = f"{result['first_name']} {result['last_name']}"
        else:
            name = str(result.get('id', 'Unknown'))
        return name, score
    
    return result, score