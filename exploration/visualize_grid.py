"""Visualize WHERE the pipeline's 8x8 grid lands vs where YOLO detects pieces.

For each video: calibrate exactly like the pipeline, warp a frame, draw the grid
that center_to_square() uses, run YOLO, and mark each detection's center + the
square it gets assigned to. If pieces sit centered in cells -> geometry is fine.
If pieces straddle grid lines / land in cells they don't belong to -> the warp is
the problem (not piece recognition).

Also prints a recall proxy: how many pieces YOLO detects per frame (good detection
count + bad grid alignment == geometry-limited, not appearance-limited).

    python exploration/visualize_grid.py videos/chessworld.mp4 videos/chess2.mp4 videos/chess3.mp4
"""
import sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.pipeline import ChessVisionPipeline
from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

BASE = Path('D:/chessworldai_assignment/chessvision-pgn')
OUT = BASE / 'exploration' / '_grid'
OUT.mkdir(parents=True, exist_ok=True)


def grid_lines(border_margin, top_off, left_off, sq_h, sq_w, use_inner):
    """Return the 9 x and 9 y grid coordinates center_to_square implies."""
    if use_inner:
        xs = [left_off + i * sq_w for i in range(9)]
        ys = [top_off + i * sq_h for i in range(9)]
    else:
        active = 640.0 - 2 * border_margin
        adj = active / 8.0
        xs = [border_margin + i * adj for i in range(9)]
        ys = [border_margin + i * adj for i in range(9)]
    return xs, ys


def main():
    pipe = ChessVisionPipeline(piece_model_path="models/piece_detector.pt", confidence=0.25)
    bd, pd = pipe.board_detector, pipe.piece_detector

    for vp in sys.argv[1:]:
        stem = Path(vp).stem
        rotation, margin = pipe._calibrate_board(vp)

        cap = cv2.VideoCapture(vp)
        n = 0
        while cap.grab():
            n += 1
        print(f"\n{stem}: rotation={rotation} margin={margin} ({n} frames)")

        for tag, frac in (("start", 0.04), ("mid", 0.55)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
            ret, frame = cap.read()
            if not ret:
                continue
            corners = bd.detect(frame)
            if corners is None:
                print(f"  {tag}: NO BOARD DETECTED")
                continue
            H = bd.get_homography(corners)
            warped = bd.warp(frame, H)
            if rotation == 90:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                warped = cv2.rotate(warped, cv2.ROTATE_180)
            elif rotation == 270:
                warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

            top_off, left_off, sq_h, sq_w = BoardDetector.detect_inner_board(warped)
            use_inner = top_off > 30 or left_off > 30 or sq_h < 65 or sq_w < 65

            vis = warped.copy()
            xs, ys = grid_lines(margin, top_off, left_off, sq_h, sq_w, use_inner)
            for x in xs:
                cv2.line(vis, (int(x), int(ys[0])), (int(x), int(ys[-1])), (0, 0, 255), 1)
            for y in ys:
                cv2.line(vis, (int(xs[0]), int(y)), (int(xs[-1]), int(y)), (0, 0, 255), 1)

            results = pd.model(warped, conf=pd.confidence, verbose=False)[0]
            n_det = 0
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = results.names[int(box.cls[0])]
                if PIECE_CODES.get(cls) is None:
                    continue
                n_det += 1
                cx = (x1 + x2) / 2
                cy = y1 * 0.3 + y2 * 0.7
                row, col = PieceDetector.center_to_square(
                    cx, cy, border_margin=margin,
                    top_offset=top_off if use_inner else 0.0,
                    left_offset=left_off if use_inner else 0.0,
                    sq_h=sq_h if use_inner else 0.0,
                    sq_w=sq_w if use_inner else 0.0,
                )
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.circle(vis, (int(cx), int(cy)), 4, (255, 0, 255), -1)
                cv2.putText(vis, f"{row}{col}", (int(cx) - 10, int(cy) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            cv2.imwrite(str(OUT / f"{stem}_{tag}.jpg"), vis)
            print(f"  {tag}: {n_det} pieces detected, use_inner={use_inner}")
        cap.release()
    print(f"\nsaved annotated grids to {OUT}")


if __name__ == "__main__":
    main()
