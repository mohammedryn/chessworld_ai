import numpy as np
import pytest
from src.change_detector import ChangeDetector


def test_first_frame_always_passes():
    detector = ChangeDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.has_changed(frame) is True


def test_identical_frames_return_false():
    detector = ChangeDetector()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detector.has_changed(frame)          # prime with first frame
    assert detector.has_changed(frame) is False


def test_different_frames_return_true():
    # Optical flow needs texture — uniform frames have no gradient so flow is always zero.
    # Use seeded random frames to simulate realistic pixel variance.
    detector = ChangeDetector()
    rng = np.random.default_rng(42)
    frame_a = rng.integers(0, 128, (480, 640, 3), dtype=np.uint8)
    frame_b = rng.integers(128, 256, (480, 640, 3), dtype=np.uint8)
    detector.has_changed(frame_a)
    assert detector.has_changed(frame_b) is True


def test_reset_clears_state():
    detector = ChangeDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.has_changed(frame)
    detector.reset()
    assert detector.has_changed(frame) is True  # first frame again
