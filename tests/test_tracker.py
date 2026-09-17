import numpy as np
from src.centroid_tracker import CentroidTracker

def test_centroid_registration():
    tracker = CentroidTracker(max_disappeared=5, max_distance=50)
    tracker.update([(10, 10, 20, 20)], [(20, 20)])
    
    assert len(tracker.objects) == 1
    assert 1 in tracker.objects

def test_centroid_deregistration():
    tracker = CentroidTracker(max_disappeared=2, max_distance=50)
    tracker.update([(10, 10, 20, 20)], [(20, 20)])
    assert len(tracker.objects) == 1
    
    # Frame 1 empty
    tracker.update([], [])
    assert len(tracker.objects) == 1
    
    # Frame 2 empty
    tracker.update([], [])
    assert len(tracker.objects) == 1
    
    # Frame 3 empty - should deregister
    tracker.update([], [])
    assert len(tracker.objects) == 0

def test_centroid_matching():
    tracker = CentroidTracker(max_disappeared=5, max_distance=50)
    tracker.update([(10, 10, 20, 20)], [(20, 20)])
    
    # Move object slightly
    tracker.update([(12, 12, 20, 20)], [(22, 22)])
    assert len(tracker.objects) == 1
    
    # Check if centroid updated
    assert tracker.objects[1][0] == (22, 22)
