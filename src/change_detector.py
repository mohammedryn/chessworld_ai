import cv2
import numpy as np

MOTION_THRESHOLD = 5.0


class ChangeDetector:
    def __init__(self, threshold: float = MOTION_THRESHOLD):
        self.threshold = threshold
        self._prev_gray: np.ndarray | None = None

    def has_changed(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return True

        # Use frame differencing to detect change
        diff = cv2.absdiff(self._prev_gray, gray)
        changed = float(diff.mean()) > self.threshold
        self._prev_gray = gray
        return changed

    def reset(self):
        self._prev_gray = None
