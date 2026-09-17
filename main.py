import argparse
from src.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="Classical Computer Vision Video Analysis")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Directory to save results")
    parser.add_argument("--method", choices=["background", "optical_flow", "klt", "all"], default="all", help="Tracking method to use")
    
    # Optional parameters for background subtraction & tracking
    parser.add_argument("--min-area", type=int, default=500, help="Minimum contour area to be considered an object")
    parser.add_argument("--history", type=int, default=500, help="History for MOG2 background subtractor")
    parser.add_argument("--var-threshold", type=float, default=16.0, help="Variance threshold for MOG2")
    parser.add_argument("--max-disappeared", type=int, default=15, help="Max frames an object can be lost before deregistering")
    parser.add_argument("--max-distance", type=float, default=50.0, help="Max Euclidean distance for centroid matching")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames to let MOG2 background model stabilize")

    args = parser.parse_args()
    
    config = {
        'input': args.input,
        'output': args.output,
        'method': args.method,
        'min_area': args.min_area,
        'history': args.history,
        'var_threshold': args.var_threshold,
        'max_disappeared': args.max_disappeared,
        'max_distance': args.max_distance,
        'warmup_frames': args.warmup_frames
    }
    
    processor = VideoProcessor(config)
    processor.process()

if __name__ == "__main__":
    main()
