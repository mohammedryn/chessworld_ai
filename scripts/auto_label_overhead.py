"""Auto-label overhead camera frames from the KNOWN starting position.

For overhead IP-cam videos that start at move 1, the first few seconds show
all 32 pieces in their exact starting positions. This script:
  1. Extracts ~N warped frames from the opening (before any move occurs)
  2. Uses the known chess starting position to generate PERFECT YOLO labels
     (maps each piece's chess square to pixel coordinates in the calibrated warp)
  3. Saves images + labels ready for Roboflow / direct retraining

No manual labeling needed: the board position is definitively known. The key
improvement over the CCTV frame extractor: labels are GENERATED from chess
knowledge, not predicted by a failing model. After fine-tuning on these overhead
frames, the model will recognise pieces at this camera angle.

    python scripts/auto_label_overhead.py --video videos/chessworld.mp4
    python scripts/auto_label_overhead.py --video videos/chess3.mp4 --count 60
    python scripts/auto_label_overhead.py --video videos/chess2.mp4 --count 60
"""
import sys, json, argparse
from pathlib import Path
import cv2, numpy as np, chess

sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector

BASE = Path('D:/chessworldai_assignment/chessvision-pgn')

# Model class index lookup (from model.names)
PIECE_CODE_TO_NAME = {
    'K': 'white-king', 'Q': 'white-queen', 'R': 'white-rook',
    'B': 'white-bishop', 'N': 'white-knight', 'P': 'white-pawn',
    'k': 'black-king',  'q': 'black-queen',  'r': 'black-rook',
    'b': 'black-bishop','n': 'black-knight', 'p': 'black-pawn',
}
NAME_TO_IDX = {
    'black-bishop': 0, 'black-king': 1, 'black-knight': 2, 'black-pawn': 3,
    'black-queen': 4, 'black-rook': 5, 'white-bishop': 6, 'white-king': 7,
    'white-knight': 8, 'white-pawn': 9, 'white-queen': 10, 'white-rook': 11,
}


def rotate(img, r):
    if r == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if r == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def board_to_label_grid() -> dict[tuple, tuple]:
    """Map chess squares → (row, col) in the rotated warp.

    The pipeline applies its calibration rotation to the WARP IMAGE so that
    white ends up at the bottom — matching the standard chess orientation.
    The label coordinates are for the ALREADY-ROTATED image, so we use the
    standard mapping directly (no rotation correction needed here).

      row = 7 - rank  →  rank 8 (black back) = row 0 (image top)
                          rank 1 (white back) = row 7 (image bottom)
      col = file      →  a-file = col 0 (image left)
    """
    mapping = {}
    for sq in chess.SQUARES:
        row = 7 - chess.square_rank(sq)
        col = chess.square_file(sq)
        mapping[sq] = (row, col)
    return mapping


def sq_to_pixel(row: int, col: int, sq_h: float, sq_w: float,
                top: float, left: float) -> tuple[float, float]:
    """Center pixel of square (row, col) in the 640x640 warp."""
    cx = left + (col + 0.5) * sq_w
    cy = top + (row + 0.5) * sq_h
    return cx, cy


def generate_labels(board: chess.Board, sq_map: dict, sq_h: float, sq_w: float,
                    top: float, left: float, img_size: int = 640) -> list[str]:
    """YOLO label lines for all pieces on the board."""
    lines = []
    # From overhead, pieces appear as discs covering ~50% of each square
    bw = sq_w * 0.55 / img_size
    bh = sq_h * 0.55 / img_size
    for sq, (row, col) in sq_map.items():
        piece = board.piece_at(sq)
        if piece is None:
            continue
        code = piece.symbol()
        name = PIECE_CODE_TO_NAME.get(code)
        if name is None:
            continue
        cls_idx = NAME_TO_IDX[name]
        cx, cy = sq_to_pixel(row, col, sq_h, sq_w, top, left)
        cx_n = cx / img_size
        cy_n = cy / img_size
        lines.append(f"{cls_idx} {cx_n:.6f} {cy_n:.6f} {bw:.6f} {bh:.6f}")
    return lines


def detect_board_bounds(warped: np.ndarray) -> tuple[float, float, float, float]:
    """Return grid boundaries for the calibrated warp.

    The calibrated perspective warp maps detected board corners to the full
    640×640 image, so the board fills the image with a uniform 8×8 grid of
    80px cells. We use this directly rather than trying to detect inner
    boundaries (detect_inner_board is unreliable for overhead views — dark
    pieces from above have low row variance and are incorrectly excluded).
    """
    return 0.0, 0.0, 80.0, 80.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--count", type=int, default=80,
                    help="Frames to extract from the opening (default 80).")
    ap.add_argument("--opening-secs", type=float, default=20.0,
                    help="Seconds from start to sample (default 20s).")
    args = ap.parse_args()

    stem = Path(args.video).stem
    cfg_path = BASE / "calibration" / f"{stem}.json"
    if not cfg_path.exists():
        print(f"No calibration for {stem}. Run scripts/calibrate_camera.py first.")
        return

    cfg = json.loads(cfg_path.read_text())
    corners = np.array(cfg["corners"], np.float32)
    H = BoardDetector().get_homography(corners)
    k1, k2, rot = cfg["k1"], cfg.get("k2", 0.0), cfg["rotation"]
    sq_map = board_to_label_grid()

    out = BASE / "training_frames" / f"{stem}_overhead"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total = 0
    while cap.grab():
        total += 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Sample only from the opening window (before any move happens)
    open_end = min(int(fps * args.opening_secs), total)
    step = max(1, open_end // args.count)

    maps = BoardDetector.undistort_maps(
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        k1, k2) if (k1 or k2) else None

    starting_board = chess.Board()
    saved = 0
    for fi in range(0, open_end, step):
        if saved >= args.count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        src = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR) if maps else frame
        warped = rotate(cv2.warpPerspective(src, H, (640, 640)), rot)

        top, left, sq_h, sq_w = detect_board_bounds(warped)
        labels = generate_labels(starting_board, sq_map, sq_h, sq_w, top, left)
        if not labels:
            continue

        name = f"{stem}_open_f{fi:06d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), warped)
        (out / "labels" / f"{name}.txt").write_text("\n".join(labels) + "\n")
        saved += 1

    cap.release()
    print(f"{stem}: {saved} overhead frames auto-labeled → {out}")
    print(f"  Labels generated from the KNOWN starting position — no manual work needed.")
    print(f"  Board bounds used: top={top:.0f} left={left:.0f} sq_h={sq_h:.1f} sq_w={sq_w:.1f}")
    print(f"\nNext: merge into models/combined_dataset/ and run scripts/train_combined.py")
    print(f"  Or upload to Roboflow + verify labels before training.")

    # Quick visual check: save a few annotated frames
    dbg = BASE / "exploration" / "_autolabel_overhead"
    dbg.mkdir(parents=True, exist_ok=True)
    cap2 = cv2.VideoCapture(args.video)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(open_end * 0.5))
    ret, frame = cap2.read(); cap2.release()
    if ret:
        src = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR) if maps else frame
        warped = rotate(cv2.warpPerspective(src, H, (640, 640)), rot)
        vis = warped.copy()
        for sq, (row, col) in sq_map.items():
            if starting_board.piece_at(sq) is None:
                continue
            cx, cy = sq_to_pixel(row, col, sq_h, sq_w, top, left)
            bw_px = sq_w * 0.72 / 2
            bh_px = sq_h * 0.82 / 2
            code = starting_board.piece_at(sq).symbol()
            color = (0, 200, 0) if code.isupper() else (200, 100, 0)
            cv2.rectangle(vis,
                          (int(cx - bw_px), int(cy - bh_px)),
                          (int(cx + bw_px), int(cy + bh_px)), color, 1)
            cv2.putText(vis, code, (int(cx) - 5, int(cy) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.imwrite(str(dbg / f"{stem}_labels.jpg"), vis)
        print(f"\n  Debug visualization: {dbg / f'{stem}_labels.jpg'}")


if __name__ == "__main__":
    main()
