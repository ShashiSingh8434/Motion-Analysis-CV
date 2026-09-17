# Problem Statement
Moving object detection and tracking in video sequences is a critical task in computer vision with applications in surveillance, traffic monitoring, and autonomous navigation. The objective of this project is to build a robust, headless pipeline to detect, track, and analyze moving objects entirely using classical computer vision techniques, without relying on deep learning object detectors.

## Project Scope
- Video ingestion and preprocessing.
- Foreground extraction via Gaussian Mixture-based Background/Foreground Segmentation (MOG2).
- Multi-object tracking using Euclidean centroid matching.
- Motion estimation via Dense (Farneback) and Sparse (Lucas-Kanade) Optical Flow.
- Quantitative evaluation on deterministic synthetic video data.

## Target Users
- Researchers and students studying fundamental computer vision.
- Evaluators looking for non-deep-learning baselines for object tracking.

## Proposed Solution
A modular Python application executing a multi-stage pipeline:
1. **Input**: MP4 video file.
2. **Processing**: Frame-by-frame analysis applying MOG2, Farneback, and KLT techniques. Data structures log trajectory, velocity, and dominant direction.
3. **Output**: Processed tracking videos (MP4), coordinate and metric data (CSV), and summary statistics (JSON).
