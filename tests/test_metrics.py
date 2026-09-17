from src.metrics import bb_iou

def test_bb_iou_identical():
    boxA = [10, 10, 20, 20]
    boxB = [10, 10, 20, 20]
    assert bb_iou(boxA, boxB) == 1.0

def test_bb_iou_no_overlap():
    boxA = [10, 10, 20, 20]
    boxB = [50, 50, 20, 20]
    assert bb_iou(boxA, boxB) == 0.0

def test_bb_iou_partial_overlap():
    boxA = [10, 10, 20, 20]
    boxB = [20, 20, 20, 20]
    # Intersection is [20, 20, 10, 10] Area = 100
    # Union is 400 + 400 - 100 = 700
    assert abs(bb_iou(boxA, boxB) - (100 / 700.0)) < 1e-5
