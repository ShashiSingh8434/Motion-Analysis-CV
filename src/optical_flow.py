import cv2
import numpy as np

class DenseOpticalFlow:
    def __init__(self):
        self.prev_gray = None
        
    def compute(self, current_gray):
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return None, 0.0, 0.0, "STATIONARY"
            
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, current_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_gray = current_gray
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        
        mean_mag = np.mean(magnitude)
        max_mag = np.max(magnitude)
        
        # Dominant direction
        if mean_mag < 0.5: # Configurable threshold for noise
            dominant_dir = "STATIONARY"
        else:
            # We can use a histogram of angles weighted by magnitude to find the dominant direction
            hist, bins = np.histogram(angle.flatten(), bins=8, range=(0, 360), weights=magnitude.flatten())
            dominant_bin = np.argmax(hist)
            dirs = ["RIGHT", "DOWN-RIGHT", "DOWN", "DOWN-LEFT", "LEFT", "UP-LEFT", "UP", "UP-RIGHT"]
            dominant_dir = dirs[dominant_bin]
            
        return flow, mean_mag, max_mag, dominant_dir
