import os
import subprocess

def run():
    print("Running experiments...")
    
    # 1. Generate test video
    subprocess.run([
        "python", "scripts/generate_test_video.py", 
        "--output", "data/synthetic_test.mp4",
        "--ground-truth", "data/synthetic_ground_truth.csv",
        "--frames", "300"
    ])
    
    # 2. Run pipeline
    subprocess.run([
        "python", "main.py",
        "--input", "data/synthetic_test.mp4",
        "--output", "results/synthetic",
        "--method", "all",
        "--warmup-frames", "10"
    ])
    
    # 3. Run evaluation
    subprocess.run([
        "python", "scripts/evaluate.py",
        "--ground-truth", "data/synthetic_ground_truth.csv",
        "--predictions", "results/synthetic/tracking.csv"
    ])
    
if __name__ == "__main__":
    run()
