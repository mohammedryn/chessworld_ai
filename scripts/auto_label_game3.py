"""Auto-label game3 frames using the known-correct PGN.

Strategy: sample game3 at regular intervals, estimate which move we're at
based on timing (frame_idx / frames_per_move), generate YOLO labels from
that board state, then copy into the combined training dataset.
"""
import sys, cv2, chess, chess.pgn, shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.board_detector import BoardDetector

BASE = Path('D:/chessworldai_assignment/chessvision-pgn')

# Class order must match data.yaml exactly
PIECE_TO_IDX = {
    'black-bishop': 0, 'black-king': 1, 'black-knight': 2, 'black-pawn': 3,
    'black-queen': 4, 'black-rook': 5,
    'white-bishop': 6, 'white-king': 7, 'white-knight': 8, 'white-pawn': 9,
    'white-queen': 10, 'white-rook': 11,
}
SYM_TO_NAME = {
    'k': 'black-king',   'q': 'black-queen',  'r': 'black-rook',
    'b': 'black-bishop', 'n': 'black-knight',  'p': 'black-pawn',
    'K': 'white-king',   'Q': 'white-queen',   'R': 'white-rook',
    'B': 'white-bishop', 'N': 'white-knight',  'P': 'white-pawn',
}

def board_to_yolo_labels(board: chess.Board) -> list[str]:
    labels = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        col = chess.square_file(sq)      # 0=a … 7=h
        row = 7 - chess.square_rank(sq)  # 0=rank8 … 7=rank1
        cx = (col + 0.5) / 8.0
        cy = (row + 0.5) / 8.0
        w, h = 0.10, 0.12
        cls_idx = PIECE_TO_IDX[SYM_TO_NAME[piece.symbol()]]
        labels.append(f'{cls_idx} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}')
    return labels


def main():
    # Load game3 PGN
    with open(BASE / 'output/game3.pgn') as f:
        game = chess.pgn.read_game(f)
    moves = list(game.mainline_moves())
    print(f'Loaded {len(moves)} moves from game3.pgn')

    # Build list of board states: index i = state AFTER move i-1 (i=0 is start)
    states: list[chess.Board] = []
    b = chess.Board()
    states.append(b.copy())
    for m in moves:
        b.push(m)
        states.append(b.copy())

    cap = cv2.VideoCapture(str(BASE / 'videos/game3.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Corrupt metadata workaround: read actual frame count (game3 is short)
    actual_frames = 0
    while cap.grab():
        actual_frames += 1
    print(f'game3.mp4: {actual_frames} actual frames @ {fps:.1f} fps')

    start_offset = int(fps * 3)  # skip first 3 seconds pre-game setup

    bd = BoardDetector()
    out_dir = BASE / 'training_frames' / 'game3_autolabeled'
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)
    (out_dir / 'labels').mkdir(parents=True, exist_ok=True)

    # Build sample list, then map sample index → move state proportionally
    sample_interval = 90
    sample_positions = list(range(start_offset, actual_frames, sample_interval))
    n_samples = len(sample_positions)
    saved = 0

    for i, fi in enumerate(sample_positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        corners = bd.detect(frame)
        if corners is None:
            continue

        H = bd.get_homography(corners)
        warped = bd.warp(frame, H)
        # game3 uses rotation=0° so no cv2.rotate needed

        # Map sample index proportionally across known board states
        move_idx = min(int(i * len(states) / n_samples), len(states) - 1)
        board_state = states[move_idx]

        labels = board_to_yolo_labels(board_state)
        if not labels:
            continue

        stem = f'game3_f{fi:05d}'
        cv2.imwrite(str(out_dir / 'images' / f'{stem}.jpg'), warped)
        (out_dir / 'labels' / f'{stem}.txt').write_text('\n'.join(labels))
        saved += 1

    cap.release()
    print(f'Generated {saved} labeled frames in {out_dir}')

    # Copy into combined training dataset
    train_img_dir = BASE / 'models/combined_dataset/train/images'
    train_lbl_dir = BASE / 'models/combined_dataset/train/labels'

    copied = 0
    for img in (out_dir / 'images').glob('*.jpg'):
        lbl = out_dir / 'labels' / (img.stem + '.txt')
        if lbl.exists():
            shutil.copy(img, train_img_dir / img.name)
            shutil.copy(lbl, train_lbl_dir / lbl.name)
            copied += 1

    print(f'Copied {copied} image+label pairs into combined training dataset')
    print(f'New training set size: {len(list(train_img_dir.glob("*.jpg")))} images')


if __name__ == '__main__':
    main()
