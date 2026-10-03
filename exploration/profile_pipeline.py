"""Time each pipeline component on real chess3 frames to find the bottleneck."""
import sys, time, json
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.board_detector import BoardDetector
from src.change_detector import ChangeDetector
from src.piece_detector import PieceDetector

cap = cv2.VideoCapture("videos/chess3.mp4")
frames = []
for _ in range(120):
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f)
cap.release()
h, w = frames[0].shape[:2]
print(f"frame size: {w}x{h}, {len(frames)} frames")

cd = ChangeDetector()
pd = PieceDetector("models/piece_detector.pt")
print("YOLO device:", pd.model.device)

cfg = json.loads(Path("calibration/chess3.json").read_text())
corners = np.array(cfg["corners"], dtype=np.float32)
H = BoardDetector().get_homography(corners)
maps = BoardDetector.undistort_maps(h, w, cfg["k1"], cfg["k2"])

def timeit(fn, n=40):
    fn()  # warmup
    t = time.perf_counter()
    for i in range(n):
        fn(i)
    return (time.perf_counter() - t) / n * 1000  # ms

import inspect
def wrap(fn):
    return (lambda i=0: fn(i)) if len(inspect.signature(fn).parameters) else (lambda i=0: fn())

t_flow = timeit(lambda i=0: cd.has_changed(frames[i % len(frames)]))
t_remap = timeit(lambda i=0: cv2.remap(frames[i % len(frames)], maps[0], maps[1], cv2.INTER_LINEAR))
t_warp = timeit(lambda i=0: cv2.warpPerspective(frames[i % len(frames)], H, (640, 640)))
warped = cv2.warpPerspective(cv2.remap(frames[0], maps[0], maps[1], cv2.INTER_LINEAR), H, (640, 640))
t_yolo = timeit(lambda i=0: pd.detect(warped))
t_detect = timeit(lambda i=0: BoardDetector().detect(frames[i % len(frames)]))

print(f"\nPer-call timings (ms):")
print(f"  optical flow (Farneback, full {w}x{h}): {t_flow:7.1f}")
print(f"  board contour detect (non-cal path)    : {t_detect:7.1f}")
print(f"  remap undistort (full res)             : {t_remap:7.1f}")
print(f"  warpPerspective -> 640                 : {t_warp:7.1f}")
print(f"  YOLO detect on 640                     : {t_yolo:7.1f}")

# Downscale optical-flow test
small = cv2.resize(frames[0], (w // 3, h // 3))
cd2 = ChangeDetector()
t_flow_small = timeit(lambda i=0: cd2.has_changed(cv2.resize(frames[i % len(frames)], (w // 3, h // 3))))
print(f"  optical flow on 1/3-scale (+resize)    : {t_flow_small:7.1f}")
