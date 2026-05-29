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
    def center_to_square(cx: float, cy: float, square_size: float = SQUARE_SIZE, border_margin: float = 0.0) -> tuple[int, int]:
        """Map detection center (x, y) in warped 640x640 image to (row, col) on 8x8 grid."""
        cx_adj = cx - border_margin
        cy_adj = cy - border_margin
        
        active_size = 640.0 - 2.0 * border_margin
        adj_square_size = active_size / 8.0
        
        col = max(0, min(int(cx_adj // adj_square_size), 7))
        row = max(0, min(int(cy_adj // adj_square_size), 7))
        return row, col

    def detect(self, warped: np.ndarray, border_margin: float = 0.0) -> BoardState:
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
            row, col = PieceDetector.center_to_square(cx, cy, border_margin=border_margin)
            if board[row][col] is None or conf > board[row][col][1]:
                board[row][col] = (piece_code, conf)

        return board

    def mean_confidence(self, board: BoardState) -> float:
        confs = [cell[1] for row in board for cell in row if cell is not None]
        return float(np.mean(confs)) if confs else 0.0
