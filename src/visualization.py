import cv2
import numpy as np

def draw_tracking(frame, objects, history):
    vis_frame = frame.copy()
    
    for object_id, (centroid, bbox) in objects.items():
        x, y, w, h = bbox
        
        # Bounding box
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Centroid
        cv2.circle(vis_frame, centroid, 4, (0, 0, 255), -1)
        
        # ID text
        text = f"ID: {object_id}"
        cv2.putText(vis_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Trajectory
        if object_id in history:
            track = history[object_id]
            for i in range(1, len(track)):
                cv2.line(vis_frame, track[i - 1], track[i], (255, 0, 0), 2)
                
    return vis_frame

def draw_optical_flow(frame, flow, step=16):
    vis_frame = frame.copy()
    h, w = frame.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T
    
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    cv2.polylines(vis_frame, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis_frame, (x1, y1), 1, (0, 255, 0), -1)
        
    return vis_frame

def draw_klt(frame, tracks):
    vis_frame = frame.copy()
    for track in tracks:
        if len(track) > 1:
            for i in range(1, len(track)):
                p1 = (int(track[i-1][0]), int(track[i-1][1]))
                p2 = (int(track[i][0]), int(track[i][1]))
                cv2.line(vis_frame, p1, p2, (0, 255, 0), 2)
        if len(track) > 0:
            p = (int(track[-1][0]), int(track[-1][1]))
            cv2.circle(vis_frame, p, 3, (0, 0, 255), -1)
    return vis_frame
