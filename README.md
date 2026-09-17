# Video Object Tracking and Motion Analysis Using Classical Computer Vision

## Overview
This is a university Computer Vision project that demonstrates classical techniques for detecting and tracking moving objects in video. It deliberately avoids deep learning methods (like YOLO or SSD) to focus on fundamental image processing algorithms.

## Problem Statement
Detecting and tracking moving objects in video streams is a core problem in Computer Vision. While modern solutions rely on deep learning, understanding the classical approaches provides a fundamental understanding of motion, features, and pixel-level analysis.

## Motivation
To demonstrate the efficacy and understand the limitations of classical techniques like Background Subtraction, Optical Flow, and Feature Tracking.

## Objectives
- Implement moving object detection using Background Subtraction (MOG2).
- Implement multi-object tracking using a custom centroid-based tracker.
- Implement Dense Optical Flow for motion estimation.
- Implement Sparse Optical Flow (KLT) for feature tracking.
- Compare these methodologies quantitatively using a synthetic dataset.

## Computer Vision Concepts Used
- **Background Subtraction**: Modeling the background to isolate moving foreground objects.
- **Image Preprocessing**: Grayscale conversion and Gaussian blur to reduce noise.
- **Segmentation**: Morphological operations and contour detection to extract objects from foreground masks.
- **Feature Extraction**: Shi-Tomasi corner detection for finding trackable points.
- **Optical Flow**: Farneback (Dense) and Lucas-Kanade (Sparse) methods to estimate pixel motion between frames based on the brightness constancy assumption.
- **Object Tracking**: Centroid matching using Euclidean distance over consecutive frames.
- **Motion Estimation**: Computing velocity and direction based on displacement.
- **Trajectory Analysis**: Analyzing the path of an object over time.

## System Architecture
Input Video -> Preprocessing -> [Method: Background Sub, Optical Flow, KLT] -> Output Videos & Data

## Installation

```bash
# Clone the repository
git clone <url>
cd computer-vision-object-tracking

# Create virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate synthetic test video and ground truth
python scripts/generate_test_video.py --output data/synthetic_test.mp4 --ground-truth data/synthetic_ground_truth.csv --frames 300

# 2. Run the tracking pipeline (Headless mode)
python main.py --input data/synthetic_test.mp4 --output results/synthetic/ --method all --warmup-frames 10

# 3. Evaluate the results
python scripts/evaluate.py --ground-truth data/synthetic_ground_truth.csv --predictions results/synthetic/tracking.csv

# Or run everything with one script:
python scripts/run_experiments.py
```

## Synthetic Dataset
The project includes a synthetic dataset generator (`generate_test_video.py`) that creates deterministic moving geometric shapes over a noisy background to evaluate the tracking algorithms objectively.

## Running the Project
- `--method background`: Runs MOG2 Background Subtraction + Centroid Tracking.
- `--method optical_flow`: Runs Dense Farneback Optical Flow.
- `--method klt`: Runs Sparse KLT Feature Tracking.
- `--method all`: Runs all methods.

## Output
Outputs are saved in the specified `--output` directory and include:
- `background_tracking.mp4`: Video with bounding boxes and trajectories.
- `optical_flow.mp4`: Video with dense optical flow vectors.
- `klt_tracking.mp4`: Video with tracked KLT features.
- `tracking.csv`: Centroid tracker output data.
- `optical_flow.csv`: Dense optical flow statistics.
- `klt.csv`: KLT tracking statistics.
- `summary.json` & `summary.txt`: Run summary.

## Evaluation
The evaluation script compares the centroid tracking output against the generated ground truth using IoU (Intersection over Union), Precision, Recall, and Centroid Error.

## Testing
Run unit tests with:
```bash
pytest
```
