import chess
import cv2
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from .board_detector import BoardDetector
from .change_detector import ChangeDetector
from .hand_detector import HandDetector
from .piece_detector import PieceDetector, BoardState, PIECE_CODES
from .pgn_writer import PGNWriter
from .state_machine import BoardStateMachine

FRAME_SKIP = 3
LOW_CONF_WARN = 0.4
MIN_MOVES_WARN = 5
BOARD_NOT_FOUND_FRAMES = 900   # ~30s at 30fps / FRAME_SKIP
BOARD_LOST_FRAMES = 150        # ~5s at 30fps / FRAME_SKIP

logger = logging.getLogger(__name__)


class BoardNotFoundError(Exception):
    pass


class ChessVisionPipeline:
    def __init__(self, piece_model_path: str, log_path: Optional[str] = None):
        self.board_detector = BoardDetector()
        self.piece_detector = PieceDetector(piece_model_path)
        self.hand_detector = HandDetector()

        if log_path:
            logging.basicConfig(
                filename=log_path, level=logging.INFO,
                format="%(message)s",
            )

    def _log(self, event: str, **kwargs):
        logger.info(json.dumps({"event": event, **kwargs}))

    def process_video(
        self,
        video_path: str,
        output_path: str,
        demo: bool = False,
        save_demo: Optional[str] = None,
    ) -> str:
        # Fresh stateful objects per video — cheap to create
        # min_frame_gap=60: 2s at 30fps — eliminates sub-2s noise without filtering real moves
        state_machine = BoardStateMachine(min_frame_gap=60)
        change_detector = ChangeDetector()

        # Run Auto-Calibration to detect board rotation and border margin
        print(f"Auto-calibrating board rotation and grid boundaries...")
        rotation, border_margin = self._calibrate_board(video_path)
        print(f"Locked configuration: rotation={rotation}°, border_margin={border_margin}px")
        self._log("calibration", rotation=rotation, border_margin=border_margin)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(
                f"Cannot open video: {video_path}. Check the file path and codec."
            )

        raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Phone videos often have corrupt frame count metadata — cap at 2 hours
        MAX_SANE_FRAMES = int(fps * 7200)
        total_frames = raw_frame_count if 0 < raw_frame_count < MAX_SANE_FRAMES else None

        pgn_writer = PGNWriter()
        demo_writer = self._make_demo_writer(cap, save_demo, fps) if save_demo else None

        H: Optional[np.ndarray] = None
        corners: Optional[np.ndarray] = None
        board_found = False
        frames_without_board = 0
        orientation_set = False
        conf_samples: list[float] = []
        frame_idx = 0

        with tqdm(total=total_frames, desc=Path(video_path).name, unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                pbar.update(1)

                if frame_idx % FRAME_SKIP != 0:
                    continue

                if not change_detector.has_changed(frame):
                    continue

                # Recompute corners every change-detected frame for accuracy.
                # Homography locking caused stale warps when the first frame's
                # board detection was imprecise.
                corners = self.board_detector.detect(frame)
                if corners is None:
                    frames_without_board += 1
                    if not board_found and frames_without_board > BOARD_NOT_FOUND_FRAMES:
                        raise BoardNotFoundError(
                            f"Board not found in first 30s of '{video_path}'. "
                            "Ensure the full board with white border is visible."
                        )
                    if board_found and frames_without_board > BOARD_LOST_FRAMES:
                        self._log("board_lost_extended", frame=frame_idx)
                    continue

                board_found = True
                frames_without_board = 0
                H = self.board_detector.get_homography(corners)
                warped = self.board_detector.warp(frame, H)

                # Apply auto-calibration rotation to align pieces vertically
                if rotation == 90:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    warped = cv2.rotate(warped, cv2.ROTATE_180)
                elif rotation == 270:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Hand detection disabled: MediaPipe fires on players' visible arms
                # regardless of polygon shrinking, blocking ~100% of frames.
                # The sliding window vote + optical flow gate provide sufficient
                # robustness against mid-move noise without explicit hand detection.

                # Auto-detect inner board boundaries only when the warp includes
                # substantial non-board area (offset > 30px or sq < 65px).
                # For well-calibrated warps the standard 80px grid is more accurate.
                top_off, left_off, sq_h, sq_w = BoardDetector.detect_inner_board(warped)
                use_inner = top_off > 30 or left_off > 30 or sq_h < 65 or sq_w < 65
                board_state: BoardState = self.piece_detector.detect(
                    warped,
                    border_margin=border_margin,
                    top_offset=top_off if use_inner else 0.0,
                    left_offset=left_off if use_inner else 0.0,
                    sq_h=sq_h if use_inner else 0.0,
                    sq_w=sq_w if use_inner else 0.0,
                )
                conf_samples.append(self.piece_detector.mean_confidence(board_state))

                if not orientation_set:
                    # Rotation calibration already aligns the board horizontally/vertically.
                    # We set flipped to False, as White is verified at the bottom.
                    state_machine.set_orientation(flipped=False)
                    orientation_set = True

                move = state_machine.update(board_state, frame_idx=frame_idx)
                if move is not None:
                    comment = None
                    if move.promotion and move.promotion != chess.QUEEN:
                        comment = "promoted to Queen by default"
                    pgn_writer.add_move(move, comment=comment)
                    self._log("move", frame=frame_idx, uci=move.uci())
                    n = pgn_writer.move_count()
                    side = "White" if n % 2 == 1 else "Black"
                    full = (n + 1) // 2
                    print(f"  Move {n:>2}  ({side:5})  {move.uci():<8}  [frame {frame_idx}]")

                if demo or save_demo:
                    composite = self._render_demo(frame, warped, board_state, pgn_writer, move)
                    if demo:
                        cv2.imshow("ChessVision PGN", composite)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    if demo_writer:
                        demo_writer.write(composite)

        cap.release()
        if demo_writer:
            demo_writer.release()
        if demo:
            cv2.destroyAllWindows()

        n_moves = pgn_writer.move_count()
        if n_moves < MIN_MOVES_WARN:
            print(f"WARNING: Only {n_moves} moves detected in '{video_path}'. Run with --debug.")

        if conf_samples and float(np.mean(conf_samples)) < LOW_CONF_WARN:
            print("WARNING: Low average YOLO confidence. Model may not suit this piece style.")

        pgn_writer.save(output_path)
        return pgn_writer.to_string()

    def close(self):
        """Release MediaPipe resources. Call after all videos are processed."""
        self.hand_detector.close()

    def _calibrate_board(self, video_path: str) -> tuple[int, float]:
        """Automatically detect the board's rotation (0, 90, 180, 270) and border margin (0-30)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, 0.0

        # Accumulate up to 5 frames where board corners are detected
        warped_frames = []
        frame_idx = 0
        while len(warped_frames) < 5 and frame_idx < 300:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 3 != 0:
                continue

            corners = self.board_detector.detect(frame)
            if corners is not None:
                H = self.board_detector.get_homography(corners)
                warped = self.board_detector.warp(frame, H)
                warped_frames.append(warped)

        cap.release()

        if not warped_frames:
            return 0, 0.0

        # Create standard starting chess board layout
        starting_board = chess.Board()

        best_rotation = 0
        best_margin = 0.0
        best_score = -1

        rotations = [0, 90, 180, 270]
        margins = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0]

        # Lower confidence and perspective height mapping (alpha=0.3) for extremely robust calibration
        calibration_conf = 0.25
        alpha = 0.3

        for rot in rotations:
            # Collect piece detections for all collected warped frames under this rotation
            all_frame_detections = []
            for warped in warped_frames:
                warped_rot = warped.copy()
                if rot == 90:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_180)
                elif rot == 270:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

                results = self.piece_detector.model(warped_rot, conf=calibration_conf, verbose=False)[0]
                detections = []
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_name = results.names[int(box.cls[0])]
                    piece_code = PIECE_CODES.get(cls_name)
                    if piece_code is not None:
                        cx = (x1 + x2) / 2
                        cy_mapped = y1 * alpha + y2 * (1.0 - alpha)
                        detections.append((piece_code, cx, cy_mapped, conf))
                all_frame_detections.append(detections)

            # Search border margins for this rotation
            for margin in margins:
                total_matches = 0
                total_pieces = 0
                for detections in all_frame_detections:
                    board_state = np.full((8, 8), None, dtype=object)
                    for piece_code, cx, cy, conf in detections:
                        row, col = PieceDetector.center_to_square(cx, cy, border_margin=margin)
                        if board_state[row][col] is None or conf > board_state[row][col][1]:
                            board_state[row][col] = (piece_code, conf)

                    # Compute match score against standard starting position
                    matches = 0
                    for r in range(8):
                        for c in range(8):
                            sq = chess.square(c, 7 - r)
                            piece = starting_board.piece_at(sq)
                            expected = piece.symbol() if piece else None
                            detected = board_state[r][c][0] if board_state[r][c] is not None else None
                            if expected == detected:
                                matches += 1
                    total_matches += matches
                    total_pieces += len(detections)

                avg_matches = total_matches / len(warped_frames)
                avg_pieces = total_pieces / len(warped_frames)

                # Penalize configurations with too few detections
                score = avg_matches
                if avg_pieces < 8:
                    score -= 20.0

                if score > best_score:
                    best_score = score
                    best_rotation = rot
                    best_margin = margin

        return best_rotation, best_margin


    @staticmethod
    def _detect_orientation(board_state: BoardState) -> bool:
        """Return True if board is flipped (black at bottom of warped image)."""
        top_white = sum(
            1 for r in range(4) for c in range(8)
            if board_state[r][c] is not None and board_state[r][c][0].isupper()
        )
        bottom_white = sum(
            1 for r in range(4, 8) for c in range(8)
            if board_state[r][c] is not None and board_state[r][c][0].isupper()
        )
        return top_white > bottom_white

    @staticmethod
    def _make_demo_writer(
        cap: cv2.VideoCapture,
        save_path: str,
        fps: float,
    ) -> cv2.VideoWriter:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        panel1_w = int(w * (640 / h)) if h > 0 else w
        composite_w = panel1_w + 640 + 400
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(save_path, fourcc, fps, (composite_w, 640))

    def _render_demo(
        self,
        frame: np.ndarray,
        warped: np.ndarray,
        board_state: BoardState,
        pgn_writer: PGNWriter,
        last_move,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = 640 / h if h > 0 else 1.0
        panel1 = cv2.resize(frame, (max(1, int(w * scale)), 640))

        panel2 = warped.copy()

        panel3 = np.zeros((640, 400, 3), dtype=np.uint8)
        pgn_text = pgn_writer.to_string()
        lines = [ln for ln in pgn_text.replace("\n\n", "\n").strip().split("\n") if ln]
        for i, line in enumerate(lines[-14:]):
            cv2.putText(
                panel3, line[:48], (8, 32 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

        if last_move:
            cv2.putText(
                panel3, f"MOVE: {last_move}", (8, 620),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        composite = np.hstack([panel1, panel2, panel3])
        return composite
