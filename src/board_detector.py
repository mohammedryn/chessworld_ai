import cv2
import numpy as np
from typing import Optional

WARP_SIZE = 640


class BoardDetector:
    def __init__(self, warp_size: int = WARP_SIZE, lock_frames: int = 60):
        self.warp_size = warp_size
        self.lock_frames = lock_frames
        self._H: Optional[np.ndarray] = None
        self._lock_counter: int = 0

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return 4 corners (TL, TR, BR, BL) as float32 (4,2) array, or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = 0.08 * frame_area
        max_area = 0.95 * frame_area  # reject contours that span nearly the whole image
        best_quad = None
        best_area = 0.0

        # Try simple threshold first (works well for high-contrast boards)
        for thresh_method in ['simple', 'adaptive']:
            if thresh_method == 'simple':
                _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            else:
                thresh = cv2.adaptiveThreshold(
                    blurred, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2,
                )

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) != 4:
                    continue
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / h if h > 0 else 0
                if not (0.6 < aspect < 1.6):
                    continue
                if area > best_area:
                    best_area = area
                    best_quad = approx.reshape(4, 2).astype(np.float32)

            if best_quad is not None:
                break

        if best_quad is None:
            return None
        return self._sort_corners(best_quad)

    def _sort_corners(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # TL: smallest x+y
        rect[2] = pts[np.argmax(s)]   # BR: largest x+y
        diff = np.diff(pts, axis=1).ravel()
        rect[1] = pts[np.argmin(diff)]  # TR: smallest y-x
        rect[3] = pts[np.argmax(diff)]  # BL: largest y-x
        return rect

    def get_homography(self, corners: np.ndarray) -> np.ndarray:
        dst = np.array([
            [0, 0],
            [self.warp_size - 1, 0],
            [self.warp_size - 1, self.warp_size - 1],
            [0, self.warp_size - 1],
        ], dtype=np.float32)
        return cv2.getPerspectiveTransform(corners, dst)

    def warp(self, frame: np.ndarray, H: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, H, (self.warp_size, self.warp_size))

    def get_locked_homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return cached H, recomputing only every lock_frames frames."""
        self._lock_counter += 1
        if self._H is None or self._lock_counter >= self.lock_frames:
            corners = self.detect(frame)
            if corners is not None:
                self._H = self.get_homography(corners)
                self._lock_counter = 0
        return self._H
