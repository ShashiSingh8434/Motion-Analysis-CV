from src.motion_analyzer import MotionAnalyzer

def test_compute_direction():
    analyzer = MotionAnalyzer()
    
    assert analyzer.compute_direction(0, 0) == "STATIONARY"
    assert analyzer.compute_direction(1, 0) == "STATIONARY"
    assert analyzer.compute_direction(5, 0) == "RIGHT"
    assert analyzer.compute_direction(-5, 0) == "LEFT"
    assert analyzer.compute_direction(0, 5) == "DOWN"
    assert analyzer.compute_direction(0, -5) == "UP"
    assert analyzer.compute_direction(5, -5) == "UP-RIGHT"

def test_analyze_trajectory():
    analyzer = MotionAnalyzer()
    history = [(0, 0), (3, 4), (6, 8)]
    
    stats = analyzer.analyze_trajectory(history)
    
    assert stats["distance_travelled"] == 10.0
    assert stats["avg_pixel_velocity"] == 5.0
    assert stats["max_pixel_velocity"] == 5.0
    assert stats["direction"] == "DOWN-RIGHT"
