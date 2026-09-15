import pytest

from safetyvision.postprocess import filter_detections


def test_nms_preserves_different_classes():
    rows = [[0, 0, 10, 10, 0.9, 0], [1, 1, 10, 10, 0.8, 0], [0, 0, 10, 10, 0.7, 1]]
    assert len(filter_detections(rows)) == 2
    assert filter_detections([]) == []
    with pytest.raises(ValueError):
        filter_detections([[0, 0, 1, 1, 0.9, 0.5]])
