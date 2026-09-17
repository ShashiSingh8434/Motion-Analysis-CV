import cv2

def convert_to_grayscale(frame):
    """Converts a BGR frame to grayscale."""
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

def apply_gaussian_blur(frame, ksize=(5, 5)):
    """Applies Gaussian Blur to smooth the image and reduce noise."""
    return cv2.GaussianBlur(frame, ksize, 0)
