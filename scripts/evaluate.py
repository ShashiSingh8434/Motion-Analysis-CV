import pandas as pd
import numpy as np
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics import bb_iou

def evaluate(ground_truth_path, predictions_path):
    if not os.path.exists(predictions_path):
        print(f"Error: {predictions_path} not found.")
        return
        
    gt = pd.read_csv(ground_truth_path)
    pred = pd.read_csv(predictions_path)
    
    if len(pred) == 0:
        print("No predictions to evaluate.")
        return
        
    frames = sorted(list(set(gt['frame']).intersection(set(pred['frame']))))
    
    total_iou = 0
    total_centroid_error = 0
    matched_count = 0
    
    tp = 0
    fp = 0
    fn = 0
    
    for frame in frames:
        gt_frame = gt[gt['frame'] == frame]
        pred_frame = pred[pred['frame'] == frame]
        
        matched_gt = set()
        matched_pred = set()
        
        for p_idx, p_row in pred_frame.iterrows():
            best_iou = 0
            best_gt_idx = -1
            best_c_error = 0
            
            p_box = [p_row['bbox_x'], p_row['bbox_y'], p_row['bbox_w'], p_row['bbox_h']]
            p_c = np.array([p_row['centroid_x'], p_row['centroid_y']])
            
            for g_idx, g_row in gt_frame.iterrows():
                if g_idx in matched_gt:
                    continue
                    
                g_box = [g_row['bbox_x'], g_row['bbox_y'], g_row['bbox_w'], g_row['bbox_h']]
                g_c = np.array([g_row['centroid_x'], g_row['centroid_y']])
                
                iou = bb_iou(p_box, g_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
                    best_c_error = np.linalg.norm(p_c - g_c)
                    
            if best_iou > 0.3: # Match threshold
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
                total_iou += best_iou
                total_centroid_error += best_c_error
                matched_count += 1
                tp += 1
            else:
                fp += 1
                
        fn += len(gt_frame) - len(matched_gt)
        
    avg_iou = total_iou / matched_count if matched_count > 0 else 0
    avg_centroid_error = total_centroid_error / matched_count if matched_count > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("="*40)
    print("EVALUATION RESULTS (BACKGROUND TRACKING)")
    print("="*40)
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Centroid Error: {avg_centroid_error:.4f} pixels")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("="*40)
    
    return {
        'avg_iou': avg_iou,
        'avg_centroid_error': avg_centroid_error,
        'precision': precision,
        'recall': recall
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    
    evaluate(args.ground_truth, args.predictions)
