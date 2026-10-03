import cv2
import numpy as np

MOTION_THRESHOLD = 0.5
# Farneback optical flow cost scales with pixel count. On high-res CCTV streams
# (3072px) full-res flow costs ~1.2s/frame — the pipeline bottleneck. Motion
# detection doesn't need that resolution, so frames wider than this are downscaled
# for the flow computation only. Frames at or below it (640px tests, 720px
# assignment videos) are untouched, so their behaviour is bit-identical.
MAX_FLOW_WIDTH = 960


class ChangeDetector:
    def __init__(self, threshold: float = MOTION_THRESHOLD):
        self.threshold = threshold
        self._prev_gray: np.ndarray | None = None

    def has_changed(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = 1.0
        if w > MAX_FLOW_WIDTH:
            scale = MAX_FLOW_WIDTH / w
            gray = cv2.resize(gray, (MAX_FLOW_WIDTH, max(1, int(h * scale))))

        if self._prev_gray is None:
            self._prev_gray = gray
            return True

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        # Rescale to full-res-equivalent magnitude so threshold is resolution-independent.
        changed = float(magnitude.mean()) / scale > self.threshold
        self._prev_gray = gray
        return changed

    def reset(self):
        self._prev_gray = None
