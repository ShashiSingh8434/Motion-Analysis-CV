import numpy as np
from src.preprocessing import convert_to_grayscale, apply_gaussian_blur

def test_convert_to_grayscale():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = convert_to_grayscale(frame)
    assert len(gray.shape) == 2
    assert gray.shape == (100, 100)

def test_apply_gaussian_blur():
    frame = np.zeros((100, 100), dtype=np.uint8)
    blurred = apply_gaussian_blur(frame, ksize=(5, 5))
    assert blurred.shape == frame.shape
