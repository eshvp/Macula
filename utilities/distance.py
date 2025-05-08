# utilities/distance.py

import math

def calculate_distance(face_width_pixels, known_face_width_cm, focal_length_pixels):
    """
    Calculate the distance to the face based on the width of the face in the image
    and the known width of the face in real life.

    :param face_width_pixels: Width of the face in pixels in the image
    :param known_face_width_cm: The real-world width of the face in centimeters (usually around 14 cm)
    :param focal_length_pixels: Focal length of the camera in pixels
    :return: The estimated distance to the face in centimeters
    """
    # Distance formula: (real_face_width * focal_length) / face_width_in_image = distance
    distance = (known_face_width_cm * focal_length_pixels) / face_width_pixels
    return distance