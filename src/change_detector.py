import cv2
import numpy as np

MOTION_THRESHOLD = 0.5


class ChangeDetector:
    def __init__(self, threshold: float = MOTION_THRESHOLD):
        self.threshold = threshold
        self._prev_gray: np.ndarray | None = None

    def has_changed(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return True

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        changed = float(magnitude.mean()) > self.threshold
        self._prev_gray = gray
        return changed

    def reset(self):
        self._prev_gray = None
