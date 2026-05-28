import numpy as np
import pytest
from src.board_detector import BoardDetector

CORNER_TOLERANCE = 10  # pixels


def test_detect_returns_four_corners(synthetic_board_frame):
    frame, expected_pts = synthetic_board_frame
    detector = BoardDetector()
    corners = detector.detect(frame)
    assert corners is not None
    assert corners.shape == (4, 2)


def test_detected_corners_match_drawn_quad(synthetic_board_frame):
    frame, expected_pts = synthetic_board_frame
    detector = BoardDetector()
    corners = detector.detect(frame)
    assert corners is not None
    for detected in corners:
        dists = np.linalg.norm(expected_pts - detected, axis=1)
        assert dists.min() < CORNER_TOLERANCE, (
            f"Corner {detected} not near any expected corner: {expected_pts}"
        )


def test_detect_returns_none_for_blank_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = BoardDetector()
    assert detector.detect(frame) is None


def test_homography_produces_correct_output_size(synthetic_board_frame):
    frame, _ = synthetic_board_frame
    detector = BoardDetector(warp_size=640)
    corners = detector.detect(frame)
    assert corners is not None
    H = detector.get_homography(corners)
    warped = detector.warp(frame, H)
    assert warped.shape == (640, 640, 3)


def test_homography_lock_reuses_cached_H(synthetic_board_frame):
    frame, _ = synthetic_board_frame
    detector = BoardDetector(lock_frames=60)
    H1 = detector.get_locked_homography(frame)
    H2 = detector.get_locked_homography(frame)
    assert H1 is not None
    assert np.allclose(H1, H2)
