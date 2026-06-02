"""Extract warped board frames from ChessWorld AI CCTV footage for labeling.

Pulls frames at regular intervals, runs the SAME board detection + warp +
rotation that the pipeline uses, and saves the 640x640 warped images.

These are the exact images YOLO sees — so labeling THESE in Roboflow and
retraining teaches the model ChessWorld AI's specific camera angle and
piece appearance. This is the real fix for CCTV noise.

Usage:
    python scripts/extract_cctv_frames.py --video videos/chess3.mp4 --count 120

Then upload training_frames/cctv/ to Roboflow, label the pieces, export
as YOLOv8, merge into models/combined_dataset/, and run train_combined.py.
"""
import sys, argparse, cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.board_detector import BoardDetector

BASE = Path('D:/chessworldai_assignment/chessvision-pgn')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to CCTV video")
    parser.add_argument("--count", type=int, default=120,
                        help="Number of frames to extract (default: 120)")
    parser.add_argument("--rotation", type=int, default=270,
                        choices=[0, 90, 180, 270],
                        help="Rotation to apply (default: 270 — matches CCTV calibration)")
    parser.add_argument("--out", default="training_frames/cctv",
                        help="Output directory (default: training_frames/cctv)")
    args = parser.parse_args()

    bd = BoardDetector()
    out_dir = BASE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}")
        return

    # Count actual frames (CCTV metadata is often corrupt)
    actual = 0
    while cap.grab():
        actual += 1
    print(f"{Path(args.video).name}: {actual} actual frames")

    # Sample evenly across the whole video, skipping first 3s of setup
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start = int(fps * 3)
    step = max(1, (actual - start) // args.count)

    saved = 0
    for fi in range(start, actual, step):
        if saved >= args.count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        corners = bd.detect(frame)
        if corners is None:
            continue

        H = bd.get_homography(corners)
        warped = bd.warp(frame, H)

        if args.rotation == 90:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotation == 180:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        elif args.rotation == 270:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        stem = f"{Path(args.video).stem}_f{fi:06d}"
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), warped)
        saved += 1

    cap.release()
    print(f"Saved {saved} warped frames to {out_dir}")
    print("\nNext steps:")
    print("  1. Upload these images to Roboflow (free account)")
    print("  2. Label every chess piece using the 12-class schema")
    print("  3. Export as YOLOv8, merge into models/combined_dataset/")
    print("  4. Run: python scripts/train_combined.py")


if __name__ == "__main__":
    main()
