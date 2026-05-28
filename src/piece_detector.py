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
    def __init__(self, model_path: str, confidence: float = 0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

    @staticmethod
    def center_to_square(cx: float, cy: float, square_size: int = SQUARE_SIZE) -> tuple[int, int]:
        """Map detection center (x, y) in warped 640x640 image to (row, col) on 8x8 grid."""
        col = min(int(cx // square_size), 7)
        row = min(int(cy // square_size), 7)
        return row, col

    def detect(self, warped: np.ndarray) -> BoardState:
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
            cy = (y1 + y2) / 2
            row, col = PieceDetector.center_to_square(cx, cy)
            if board[row][col] is None or conf > board[row][col][1]:
                board[row][col] = (piece_code, conf)

        return board

    def mean_confidence(self, board: BoardState) -> float:
        confs = [cell[1] for row in board for cell in row if cell is not None]
        return float(np.mean(confs)) if confs else 0.0
