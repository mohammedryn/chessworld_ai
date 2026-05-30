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
        min_area = 0.05 * frame_area
        max_area = 0.95 * frame_area
        best_quad = None
        best_area = 0.0

        for thresh_method in ['simple', 'adaptive']:
            if thresh_method == 'simple':
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

                # Try progressively looser epsilon until we get 4 corners
                peri = cv2.arcLength(contour, True)
                quad = None
                for eps_factor in [0.02, 0.04, 0.06, 0.08, 0.10]:
                    approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
                    if len(approx) == 4:
                        quad = approx.reshape(4, 2).astype(np.float32)
                        break

                # Fallback: convex hull → fit to 4-point rectangle
                if quad is None:
                    hull = cv2.convexHull(contour)
                    hull_peri = cv2.arcLength(hull, True)
                    for eps_factor in [0.02, 0.05, 0.08, 0.12]:
                        approx = cv2.approxPolyDP(hull, eps_factor * hull_peri, True)
                        if len(approx) == 4:
                            quad = approx.reshape(4, 2).astype(np.float32)
                            break

                if quad is None:
                    continue

                x, y, w, h = cv2.boundingRect(quad)
                aspect = w / h if h > 0 else 0
                if not (0.5 < aspect < 2.0):
                    continue
                if area > best_area:
                    best_area = area
                    best_quad = quad

            if best_quad is not None:
                break

        if best_quad is None:
            # Fallback: green-board detection (for CCTV cameras with green/white boards)
            best_quad = self._detect_green_board(frame)

        if best_quad is None:
            return None
        return self._sort_corners(best_quad)

    def _detect_green_board(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect chess board by segmenting green squares (for green/white board style)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (36, 40, 40), (90, 255, 255))

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = 0.03 * frame_area

        big = [c for c in contours if cv2.contourArea(c) > min_area]
        if not big:
            return None

        # Union of top green regions → convex hull → 4-corner quad
        all_pts = np.vstack(big[:20])
        hull = cv2.convexHull(all_pts)
        hull_peri = cv2.arcLength(hull, True)
        for eps in [0.01, 0.02, 0.04, 0.06]:
            approx = cv2.approxPolyDP(hull, eps * hull_peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                x, y, w, h = cv2.boundingRect(quad)
                aspect = w / h if h > 0 else 0
                if 0.5 < aspect < 2.0:
                    return quad
        return None

    @staticmethod
    def detect_inner_board(warped: np.ndarray) -> tuple[int, int, float, float]:
        """Auto-detect actual chess square boundaries in a warped 640×640 image.

        Returns (top_offset, left_offset, square_h, square_w) using row/column
        variance analysis — chess squares have high local variance; uniform
        borders/tablecloth have low variance.
        """
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        def find_range(profile: np.ndarray) -> tuple[int, int]:
            try:
                import scipy.ndimage as ndi
                smoothed = ndi.uniform_filter1d(profile.astype(float), size=20)
            except ImportError:
                kernel = np.ones(20) / 20
                smoothed = np.convolve(profile.astype(float), kernel, mode='same')
            threshold = np.percentile(smoothed, 50)
            active = np.where(smoothed > threshold)[0]
            if len(active) < 16:
                return 0, len(profile) - 1
            return int(active[0]), int(active[-1])

        row_vars = np.array([np.var(gray[y, :]) for y in range(640)])
        col_vars = np.array([np.var(gray[:, x]) for x in range(640)])

        top, bottom = find_range(row_vars)
        left, right = find_range(col_vars)

        board_h = max(bottom - top, 1)
        board_w = max(right - left, 1)
        sq_h = board_h / 8.0
        sq_w = board_w / 8.0

        return top, left, sq_h, sq_w

    def _sort_corners(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # TL
        rect[2] = pts[np.argmax(s)]   # BR
        diff = np.diff(pts, axis=1).ravel()
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
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
