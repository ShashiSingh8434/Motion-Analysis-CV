import numpy as np
import math

class MotionAnalyzer:
    def __init__(self):
        pass

    def compute_direction(self, dx, dy):
        if abs(dx) < 2 and abs(dy) < 2:
            return "STATIONARY"
        
        angle = math.degrees(math.atan2(-dy, dx)) # -dy because y increases downwards
        if angle < 0:
            angle += 360
            
        if 22.5 <= angle < 67.5:
            return "UP-RIGHT"
        elif 67.5 <= angle < 112.5:
            return "UP"
        elif 112.5 <= angle < 157.5:
            return "UP-LEFT"
        elif 157.5 <= angle < 202.5:
            return "LEFT"
        elif 202.5 <= angle < 247.5:
            return "DOWN-LEFT"
        elif 247.5 <= angle < 292.5:
            return "DOWN"
        elif 292.5 <= angle < 337.5:
            return "DOWN-RIGHT"
        else:
            return "RIGHT"

    def analyze_trajectory(self, history):
        """
        Calculates motion statistics for a single object's trajectory.
        history: list of (x, y) centroids
        """
        if len(history) < 2:
            return {
                "distance_travelled": 0.0,
                "avg_pixel_velocity": 0.0,
                "max_pixel_velocity": 0.0,
                "direction": "STATIONARY",
                "frames_tracked": len(history),
                "is_moving": False
            }

        total_distance = 0.0
        max_vel = 0.0
        
        for i in range(1, len(history)):
            p1 = np.array(history[i-1])
            p2 = np.array(history[i])
            dist = np.linalg.norm(p2 - p1)
            total_distance += dist
            if dist > max_vel:
                max_vel = dist
                
        avg_vel = total_distance / (len(history) - 1)
        
        start_p = history[0]
        end_p = history[-1]
        dx = end_p[0] - start_p[0]
        dy = end_p[1] - start_p[1]
        
        direction = self.compute_direction(dx, dy)
        is_moving = total_distance > 10.0 # Configurable threshold
        
        return {
            "distance_travelled": total_distance,
            "avg_pixel_velocity": avg_vel,
            "max_pixel_velocity": max_vel,
            "direction": direction,
            "frames_tracked": len(history),
            "is_moving": is_moving
        }
