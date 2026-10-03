"""Per-square occupancy detector for fixed overhead IP-cam views.

Replaces YOLO for calibrated videos where pieces appear as overhead blobs.
YOLO was trained on 45-degree oblique views and misses ~50% of pieces from
overhead, capping the state-machine match score at ~48/64.

This detector only needs to answer two questions per square:
  1. Occupied or empty?  (variance: pieces create texture vs flat background)
  2. White piece or black piece?  (brightness vs known empty-square background)

The state machine's _match_score only checks occupancy (not piece type), and
its internal chess.Board() tracks actual piece identity from legal moves. So
'P'/'p' as generic white/black codes are sufficient — the state machine does
the rest.

IMPORTANT: Pass actual board boundaries (top_offset, left_offset, sq_h, sq_w)
from BoardDetector.detect_inner_board() so the detector uses the same grid as
PieceDetector.  Assuming a uniform 80px grid causes mis-assignment.

Calibration: the first few frames show the starting position (all videos start
at move 1). Rows 2-5 are empty, rows 0-1 are black pieces, rows 6-7 are white.
"""

import cv2
import numpy as np
from typing import Optional

from .piece_detector import BoardState

# Fraction of each square's interior to sample (avoids border bleed from
# adjacent pieces). 0.35 = inner 35% of width/height.
INNER_FRAC = 0.35


def _crop(img2d: np.ndarray, row: int, col: int,
          top: float, left: float, sq_h: float, sq_w: float) -> np.ndarray:
    """Return the central INNER_FRAC portion of square (row, col)."""
    mh = sq_h * (1.0 - INNER_FRAC) / 2.0
    mw = sq_w * (1.0 - INNER_FRAC) / 2.0
    y0 = max(0, int(top + row * sq_h + mh))
    y1 = max(y0 + 1, int(top + (row + 1) * sq_h - mh))
    x0 = max(0, int(left + col * sq_w + mw))
    x1 = max(x0 + 1, int(left + (col + 1) * sq_w - mw))
    return img2d[y0:y1, x0:x1]


class SquareOccupancyDetector:
    """Overhead occupancy detector. Call calibrate() once on opening frames."""

    def __init__(self):
        self._var_threshold: Optional[float] = None
        self._empty_v: dict[tuple, float] = {}   # (row,col) -> expected V when empty
        self._calibrated: bool = False

    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        start_frames: list[np.ndarray],
        top: float = 0.0, left: float = 0.0,
        sq_h: float = 80.0, sq_w: float = 80.0,
    ) -> bool:
        """Learn variance threshold and per-square empty brightness.

        Uses the known starting position: rows 2-5 are empty, rows 0-1 are
        black pieces, rows 6-7 are white pieces.

        Args:
            start_frames: calibrated warped 640x640 frames from the opening.
            top/left/sq_h/sq_w: actual board boundaries from detect_inner_board().
        """
        empty_vars: list[float] = []
        piece_vars: list[float] = []
        bright_acc: dict[tuple, list] = {}

        for warped in start_frames:
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
            v_ch = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)

            for row in range(8):
                for col in range(8):
                    cg = _crop(gray, row, col, top, left, sq_h, sq_w)
                    if cg.size == 0:
                        continue
                    var = float(np.var(cg))

                    if 2 <= row <= 5:           # empty at game start
                        empty_vars.append(var)
                        key = (row, col)
                        cv = _crop(v_ch, row, col, top, left, sq_h, sq_w)
                        bright_acc.setdefault(key, []).append(float(np.mean(cv)))
                    elif row in (0, 1, 6, 7):   # pieces at game start
                        piece_vars.append(var)

        if not empty_vars or not piece_vars:
            return False

        # Threshold midway between 85th-pct empty and 15th-pct piece variance.
        self._var_threshold = (
            float(np.percentile(empty_vars, 85)) +
            float(np.percentile(piece_vars, 15))
        ) / 2.0

        for key, vals in bright_acc.items():
            self._empty_v[key] = float(np.mean(vals))

        self._calibrated = True
        return True

    # ------------------------------------------------------------------ #
    def detect(
        self,
        warped: np.ndarray,
        top: float = 0.0, left: float = 0.0,
        sq_h: float = 80.0, sq_w: float = 80.0,
    ) -> BoardState:
        """Return 8x8 BoardState with 'P'/'p' (white/black) or None per square.

        The piece code is intentionally generic ('P'/'p') — the state machine
        only needs occupancy + colour; actual piece identity is maintained by
        chess.Board() internally.

        Args:
            warped: calibrated 640x640 board image.
            top/left/sq_h/sq_w: actual board boundaries from detect_inner_board().
        """
        board: BoardState = np.full((8, 8), None, dtype=object)

        if not self._calibrated:
            return board

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
        v_ch = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)

        for row in range(8):
            for col in range(8):
                cg = _crop(gray, row, col, top, left, sq_h, sq_w)
                if cg.size == 0:
                    continue
                var = float(np.var(cg))

                if var <= self._var_threshold:
                    continue  # empty

                # Piece present — determine colour vs expected empty background.
                cv = _crop(v_ch, row, col, top, left, sq_h, sq_w)
                brightness = float(np.mean(cv))

                # Use expected empty brightness for same column, middle rank.
                bg = self._empty_v.get(
                    (3, col),
                    self._empty_v.get((4, col),
                    self._empty_v.get((row, col), 128.0)),
                )

                code = 'P' if brightness >= bg else 'p'
                conf = min(var / self._var_threshold, 1.0)
                board[row][col] = (code, conf)

        return board

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
