import cv2

class BackgroundSubtractor:
    def __init__(self, history=500, var_threshold=16, detect_shadows=True):
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        
    def apply(self, frame):
        """Applies background subtraction to get foreground mask."""
        fg_mask = self.back_sub.apply(frame)
        return fg_mask
