import cv2
import numpy as np
import pandas as pd
import argparse
import os

def generate_test_video(output_video, output_csv, num_frames=300, width=640, height=480, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    np.random.seed(42)
    
    # Define objects: [id, type, color, x, y, vx, vy, w, h]
    objects = [
        {'id': 1, 'type': 'rect', 'color': (0, 0, 255), 'x': 50, 'y': 100, 'vx': 2, 'vy': 0, 'w': 40, 'h': 40},
        {'id': 2, 'type': 'circle', 'color': (0, 255, 0), 'x': 500, 'y': 50, 'vx': -1.5, 'vy': 1.5, 'w': 30, 'h': 30},
        {'id': 3, 'type': 'rect', 'color': (255, 0, 0), 'x': 50, 'y': 300, 'vx': 4, 'vy': 0.5, 'w': 20, 'h': 60}
    ]
    
    ground_truth = []
    
    for frame_idx in range(num_frames):
        # Create a noisy background
        bg = np.random.randint(20, 50, (height, width, 3), dtype=np.uint8)
        
        # Add a static element
        cv2.rectangle(bg, (200, 200), (300, 400), (60, 60, 60), -1)
        
        for obj in objects:
            # Update position
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']
            
            x, y, w, h = int(obj['x']), int(obj['y']), int(obj['w']), int(obj['h'])
            
            # Draw
            if obj['type'] == 'rect':
                cv2.rectangle(bg, (x, y), (x + w, y + h), obj['color'], -1)
            elif obj['type'] == 'circle':
                cv2.circle(bg, (x + w//2, y + h//2), w//2, obj['color'], -1)
                
            # Add to ground truth
            ground_truth.append({
                'frame': frame_idx + 1,
                'object_id': obj['id'],
                'centroid_x': x + w / 2,
                'centroid_y': y + h / 2,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h
            })
            
        out.write(bg)
        
    out.release()
    
    df = pd.DataFrame(ground_truth)
    df.to_csv(output_csv, index=False)
    
    print(f"Generated synthetic video: {output_video}")
    print(f"Generated ground truth: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.ground_truth), exist_ok=True)
    
    generate_test_video(args.output, args.ground_truth, num_frames=args.frames)
