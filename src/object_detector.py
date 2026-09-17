import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, min_area=500):
        self.min_area = min_area
        
    def detect(self, fg_mask):
        """
        Detects moving objects from the foreground mask.
        Applies morphological operations to remove noise and fill holes.
        Returns a list of bounding boxes and a list of centroids.
        """
        # Threshold to remove shadows (which are usually gray in MOG2)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        centroids = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                
                bboxes.append((x, y, w, h))
                centroids.append((cx, cy))
                
        return bboxes, centroids, closing
