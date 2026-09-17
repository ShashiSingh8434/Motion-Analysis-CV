# Methodology and Academic Report

## 1. Introduction
This project implements and evaluates classical computer vision techniques for moving object detection, tracking, and motion analysis in video sequences.

## 2. Problem Statement
To detect and track moving objects without utilizing pre-trained deep learning models, leveraging fundamental image processing algorithms.

## 3. Objectives
Implement Background Subtraction, Dense Optical Flow, and Sparse Feature Tracking. Construct a multi-object centroid tracker. Evaluate quantitatively against ground-truth data.

## 4. Proposed Approach
The architecture consists of:
- **Preprocessing**: Grayscale conversion and Gaussian Blurring.
- **Background Subtraction (MOG2)**: Isolates moving objects.
- **Centroid Tracking**: Matches objects across frames using Euclidean distance.
- **Optical Flow**: Farneback (Dense) and Lucas-Kanade (Sparse) for motion vectors.

## 5. Mathematical Foundations

### Euclidean Distance
Used for centroid matching in the tracker:
`d = sqrt((x2 - x1)^2 + (y2 - y1)^2)`

### Pixel Displacement
`Δx = x_t - x_(t-1)`
`Δy = y_t - y_(t-1)`

### Pixel Velocity
`v = sqrt(Δx² + Δy²) / Δt`

### Trajectory Distance
`D = Σ sqrt((x_i - x_(i-1))² + (y_i - y_(i-1))²)`

### Optical Flow (Brightness Constancy Assumption)
Assuming the brightness of a pixel doesn't change as it moves:
`I(x, y, t) = I(x + u, y + v, t + 1)`

### Intersection over Union (IoU)
Used for evaluation:
`IoU = Area(Intersection) / Area(Union)`

## 6. Algorithms

### MOG2 Background Subtraction
Models the background of a scene using a mixture of K Gaussian distributions per pixel. 

### Centroid Tracking
1. Receive bounding box coordinates and compute centroids.
2. Calculate Euclidean distance between new centroids and existing tracked objects.
3. Assign IDs to new centroids minimizing distance.
4. Deregister objects if they disappear for `max_disappeared` frames.

### Farneback Dense Optical Flow
Computes optical flow for all points in the frame using polynomial expansion.

### Lucas-Kanade (KLT) Sparse Optical Flow
Calculates optical flow for sparse feature sets (corners) detected by Shi-Tomasi.

## 7. Evaluation Methodology
A synthetic video generator creates deterministic scenarios of moving shapes. A matching algorithm pairs predicted objects with ground truth objects (IoU > 0.3) to compute Precision, Recall, and Centroid Error.

## 8. Limitations
- Background subtraction is sensitive to sudden lighting changes.
- Centroid tracking can fail during object occlusion.
- Classical methods struggle with highly non-rigid objects compared to modern deep learning segmenters.
