"""Extract calibrated training frames + model-assisted pre-labels for retraining.

For a fixed-camera video with a calibration, this samples frames, applies the
EXACT calibrated preprocessing the pipeline uses at inference (undistort + locked
corners + rotation), runs YOLO to propose boxes, and writes YOLO-format labels.

Upload images + labels to Roboflow: the model's guesses appear as pre-labels, so
you CORRECT boxes (~5x faster) instead of drawing from scratch. The conf is set
low to favour recall (catch faint pieces); you delete the false positives, which
is much faster than adding missed pieces.

    python scripts/prelabel.py --video videos/chess3.mp4 --count 120 --conf 0.12

Output:
    training_frames/<stem>_cal/images/*.jpg
    training_frames/<stem>_cal/labels/*.txt   (YOLO: cls cx cy w h, normalised)
    training_frames/<stem>_cal/classes.txt
"""
import sys, json, argparse
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

BASE = Path('D:/chessworldai_assignment/chessvision-pgn')
# Fixed 12-class order (matches PIECE_CODES insertion order: white then black).
CLASSES = list(PIECE_CODES.keys())
CODE_TO_IDX = {PIECE_CODES[name]: i for i, name in enumerate(CLASSES)}


def rotate(img, r):
    if r == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if r == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.12,
                    help="Low conf favours recall; you delete false positives in Roboflow.")
    args = ap.parse_args()

    stem = Path(args.video).stem
    cfg_path = BASE / "calibration" / f"{stem}.json"
    if not cfg_path.exists():
        print(f"No calibration for {stem}. Run scripts/calibrate_camera.py first.")
        return
    cfg = json.loads(cfg_path.read_text())
    H = BoardDetector().get_homography(np.array(cfg["corners"], np.float32))
    k1, k2, rot = cfg["k1"], cfg.get("k2", 0.0), cfg["rotation"]

    out = BASE / "training_frames" / f"{stem}_cal"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n")

    pd = PieceDetector("models/piece_detector.pt")
    cap = cv2.VideoCapture(args.video)
    total = 0
    while cap.grab():
        total += 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start = int(fps * 3)
    step = max(1, (total - start) // args.count)

    saved = boxes = 0
    maps = BoardDetector.undistort_maps(
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        k1, k2) if (k1 or k2) else None

    for fi in range(start, total, step):
        if saved >= args.count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        src = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR) if maps else frame
        warped = rotate(cv2.warpPerspective(src, H, (640, 640)), rot)

        res = pd.model(warped, conf=args.conf, verbose=False)[0]
        lines = []
        for box in res.boxes:
            code = PIECE_CODES.get(res.names[int(box.cls[0])])
            if code is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / 640
            cy = (y1 + y2) / 2 / 640
            bw = (x2 - x1) / 640
            bh = (y2 - y1) / 640
            lines.append(f"{CODE_TO_IDX[code]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        name = f"{stem}_f{fi:06d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), warped)
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
        saved += 1
        boxes += len(lines)

    cap.release()
    print(f"{stem}: {saved} frames, {boxes} pre-labelled boxes "
          f"(~{boxes/max(saved,1):.1f}/frame) -> {out}")
    print("Upload images/ + labels/ to Roboflow (YOLOv8), fix boxes, export, "
          "merge into models/combined_dataset/, run scripts/train_combined.py")


if __name__ == "__main__":
    main()
