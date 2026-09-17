import cv2
import os
import time
import json
import pandas as pd
from .preprocessing import convert_to_grayscale, apply_gaussian_blur
from .background_subtractor import BackgroundSubtractor
from .object_detector import ObjectDetector
from .centroid_tracker import CentroidTracker
from .optical_flow import DenseOpticalFlow
from .klt_tracker import KLTTracker
from .motion_analyzer import MotionAnalyzer
from .visualization import draw_tracking, draw_optical_flow, draw_klt
from .utils import ensure_dir, save_json

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        
    def process(self):
        input_path = self.config['input']
        output_dir = self.config['output']
        method = self.config['method']
        
        if not os.path.exists(input_path):
            print(f"ERROR: Could not open video:\n{input_path}\nPlease verify that the file exists and is a valid video.")
            return False

        ensure_dir(output_dir)
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video:\n{input_path}\nPlease verify that the file exists and is a valid video.")
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print("="*50)
        print("CLASSICAL COMPUTER VISION VIDEO ANALYSIS")
        print("="*50)
        print(f"Input: {input_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Frames: {total_frames}")
        print(f"Duration: {duration:.2f} sec")
        print(f"Method: {method}")
        print("Processing:")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writers = {}
        if method in ['background', 'all']:
            writers['background'] = cv2.VideoWriter(os.path.join(output_dir, 'background_tracking.mp4'), fourcc, fps, (width, height))
        if method in ['optical_flow', 'all']:
            writers['optical_flow'] = cv2.VideoWriter(os.path.join(output_dir, 'optical_flow.mp4'), fourcc, fps, (width, height))
        if method in ['klt', 'all']:
            writers['klt'] = cv2.VideoWriter(os.path.join(output_dir, 'klt_tracking.mp4'), fourcc, fps, (width, height))
            
        # Initialize modules
        bg_subtractor = BackgroundSubtractor(history=self.config.get('history', 500), var_threshold=self.config.get('var_threshold', 16))
        obj_detector = ObjectDetector(min_area=self.config.get('min_area', 500))
        tracker = CentroidTracker(max_disappeared=self.config.get('max_disappeared', 15), max_distance=self.config.get('max_distance', 50))
        dense_flow = DenseOpticalFlow()
        klt_tracker = KLTTracker()
        motion_analyzer = MotionAnalyzer()
        
        tracking_data = []
        flow_data = []
        klt_data = []
        
        frame_idx = 0
        start_time = time.time()
        
        warmup_frames = self.config.get('warmup_frames', 30)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Preprocessing
            gray = convert_to_grayscale(frame)
            blurred = apply_gaussian_blur(gray)
            
            # --- Background Subtraction ---
            if method in ['background', 'all']:
                fg_mask = bg_subtractor.apply(blurred)
                
                # Only track if past warmup
                if frame_idx > warmup_frames:
                    bboxes, centroids, mask = obj_detector.detect(fg_mask)
                    objects, history = tracker.update(bboxes, centroids)
                    
                    for obj_id, (centroid, bbox) in objects.items():
                        tracking_data.append({
                            'frame': frame_idx,
                            'object_id': obj_id,
                            'centroid_x': centroid[0],
                            'centroid_y': centroid[1],
                            'bbox_x': bbox[0],
                            'bbox_y': bbox[1],
                            'bbox_w': bbox[2],
                            'bbox_h': bbox[3]
                        })
                        
                    vis_frame = draw_tracking(frame, objects, history)
                    writers['background'].write(vis_frame)
                else:
                    writers['background'].write(frame)
                    
            # --- Optical Flow ---
            if method in ['optical_flow', 'all']:
                flow, mean_mag, max_mag, dominant_dir = dense_flow.compute(gray)
                if flow is not None:
                    flow_data.append({
                        'frame': frame_idx,
                        'mean_magnitude': mean_mag,
                        'max_magnitude': max_mag,
                        'dominant_direction': dominant_dir
                    })
                    vis_flow = draw_optical_flow(frame, flow)
                    writers['optical_flow'].write(vis_flow)
                else:
                    writers['optical_flow'].write(frame)
                    
            # --- KLT ---
            if method in ['klt', 'all']:
                tracks, avg_disp = klt_tracker.compute(gray)
                klt_data.append({
                    'frame': frame_idx,
                    'num_features': len(tracks),
                    'avg_displacement': avg_disp
                })
                vis_klt = draw_klt(frame, tracks)
                writers['klt'].write(vis_klt)
                
            # Progress bar
            progress = int((frame_idx / total_frames) * 30) if total_frames > 0 else 0
            print(f"\r[{'#' * progress}{' ' * (30 - progress)}] {int((frame_idx / total_frames) * 100)}%", end="")
            
        print()
        processing_time = time.time() - start_time
        avg_processing_fps = frame_idx / processing_time if processing_time > 0 else 0
        
        # Save CSVs
        if method in ['background', 'all']:
            if tracking_data:
                df_track = pd.DataFrame(tracking_data)
                df_track.to_csv(os.path.join(output_dir, 'tracking.csv'), index=False)
            
        if method in ['optical_flow', 'all']:
            if flow_data:
                df_flow = pd.DataFrame(flow_data)
                df_flow.to_csv(os.path.join(output_dir, 'optical_flow.csv'), index=False)
                
        if method in ['klt', 'all']:
            if klt_data:
                df_klt = pd.DataFrame(klt_data)
                df_klt.to_csv(os.path.join(output_dir, 'klt.csv'), index=False)
                
        # Generate summary
        summary = {
            "video": input_path,
            "frame_count": frame_idx,
            "fps": fps,
            "total_processing_time": processing_time,
            "avg_processing_fps": avg_processing_fps
        }
        
        if method in ['background', 'all']:
            summary['objects_detected'] = tracker.next_object_id - 1
            summary['objects_tracked'] = tracker.next_object_id - 1 # Assuming they are all tracked
            
        save_json(summary, os.path.join(output_dir, 'summary.json'))
        
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")
                
        # Clean up
        cap.release()
        for w in writers.values():
            w.release()
            
        print(f"Average processing FPS: {avg_processing_fps:.1f}")
        print(f"Processing time: {processing_time:.2f} sec")
        print("Results saved to:")
        print(output_dir)
        print("="*50)
        
        return True
