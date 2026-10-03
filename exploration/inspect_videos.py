"""Dump raw frames + metadata from videos for visual inspection."""
import sys, cv2
from pathlib import Path

BASE = Path('D:/chessworldai_assignment/chessvision-pgn')
OUT = BASE / 'exploration' / '_inspect'
OUT.mkdir(parents=True, exist_ok=True)

for vp in sys.argv[1:]:
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        print(f"{vp}: CANNOT OPEN")
        continue
    # count actual frames (metadata often corrupt)
    n = 0
    while cap.grab():
        n += 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stem = Path(vp).stem
    print(f"{stem}: {w}x{h}, {fps:.1f}fps, {n} frames ({n/fps/60:.1f} min)")
    for frac in (0.30, 0.55, 0.80):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(OUT / f"{stem}_{int(frac*100)}.jpg"), frame)
    cap.release()
print(f"saved to {OUT}")
