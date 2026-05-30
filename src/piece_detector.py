import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional

SQUARE_SIZE = 80  # 640 / 8

PIECE_CODES: dict[str, str] = {
    "white-king": "K", "white-queen": "Q", "white-rook": "R",
    "white-bishop": "B", "white-knight": "N", "white-pawn": "P",
    "black-king": "k", "black-queen": "q", "black-rook": "r",
    "black-bishop": "b", "black-knight": "n", "black-pawn": "p",
}

# 8x8 array where each cell is (piece_code: str, confidence: float) or None
BoardState = np.ndarray


class PieceDetector:
    def __init__(self, model_path: str, confidence: float = 0.25):
        self.model = YOLO(model_path)
        self.confidence = confidence

    @staticmethod
    def center_to_square(
        cx: float, cy: float,
        square_size: float = SQUARE_SIZE,
        border_margin: float = 0.0,
        top_offset: float = 0.0, left_offset: float = 0.0,
        sq_h: float = 0.0, sq_w: float = 0.0,
    ) -> tuple[int, int]:
        """Map detection center (x, y) in warped 640x640 image to (row, col) on 8x8 grid.

        If sq_h/sq_w are non-zero, use those (from auto-detected inner board area).
        Otherwise fall back to symmetric border_margin approach.
        """
        if sq_h > 0 and sq_w > 0:
            col = max(0, min(int((cx - left_offset) / sq_w), 7))
            row = max(0, min(int((cy - top_offset) / sq_h), 7))
        else:
            cx_adj = cx - border_margin
            cy_adj = cy - border_margin
            active_size = 640.0 - 2.0 * border_margin
            adj_sq = active_size / 8.0
            col = max(0, min(int(cx_adj // adj_sq), 7))
            row = max(0, min(int(cy_adj // adj_sq), 7))
        return row, col

    def detect(
        self, warped: np.ndarray,
        border_margin: float = 0.0,
        top_offset: float = 0.0, left_offset: float = 0.0,
        sq_h: float = 0.0, sq_w: float = 0.0,
    ) -> BoardState:
        """Run YOLO on warped 640x640 board image; return 8x8 BoardState."""
        board: BoardState = np.full((8, 8), None, dtype=object)
        results = self.model(warped, conf=self.confidence, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_name: str = results.names[int(box.cls[0])]
            piece_code = PIECE_CODES.get(cls_name)
            if piece_code is None:
                continue
            cx = (x1 + x2) / 2
            cy = y1 * 0.3 + y2 * 0.7
            row, col = PieceDetector.center_to_square(
                cx, cy, border_margin=border_margin,
                top_offset=top_offset, left_offset=left_offset,
                sq_h=sq_h, sq_w=sq_w,
            )
            if board[row][col] is None or conf > board[row][col][1]:
                board[row][col] = (piece_code, conf)

        return board

    def detect_per_square(
        self, warped: np.ndarray,
        border_margin: float = 0.0,
        top_offset: float = 0.0, left_offset: float = 0.0,
        sq_h: float = 0.0, sq_w: float = 0.0,
    ) -> BoardState:
        """Per-square zoomed detection — more robust for extreme oblique angles."""
        board: BoardState = np.full((8, 8), None, dtype=object)

        actual_sq_h = sq_h if sq_h > 0 else 80.0
        actual_sq_w = sq_w if sq_w > 0 else 80.0
        actual_top = top_offset if sq_h > 0 else 0.0
        actual_left = left_offset if sq_w > 0 else 0.0

        for row in range(8):
            for col in range(8):
                y1 = int(actual_top + row * actual_sq_h)
                y2 = int(actual_top + (row + 1) * actual_sq_h)
                x1 = int(actual_left + col * actual_sq_w)
                x2 = int(actual_left + (col + 1) * actual_sq_w)

                y1, y2 = max(0, y1), min(639, y2)
                x1, x2 = max(0, x1), min(639, x2)
                if y2 <= y1 or x2 <= x1:
                    continue

                sq_crop = warped[y1:y2, x1:x2]
                if sq_crop.size == 0:
                    continue

                sq_zoom = cv2.resize(sq_crop, (320, 320))
                results = self.model(sq_zoom, conf=self.confidence * 0.7, verbose=False)[0]

                if results.boxes:
                    best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
                    conf = float(best_box.conf[0])
                    cls_name = results.names[int(best_box.cls[0])]
                    piece_code = PIECE_CODES.get(cls_name)
                    if piece_code is not None:
                        board[row][col] = (piece_code, conf)

        return board

    def mean_confidence(self, board: BoardState) -> float:
        confs = [cell[1] for row in board for cell in row if cell is not None]
        return float(np.mean(confs)) if confs else 0.0
