from safetyvision.drift import psi, status


def test_drift_status():
    assert status(0.05) == "stable"
    assert status(0.15) == "warning"
    assert status(0.30) == "critical"


def test_identical_distribution():
    assert psi([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0
