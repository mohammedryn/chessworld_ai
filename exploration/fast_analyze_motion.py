import cv2
import numpy as np
from pathlib import Path

def fast_analyze():
    video_path = "D:/chessworldai_assignment/chessvision-pgn/videos/game1.mp4"
    if not Path(video_path).exists():
        video_path = "D:/chessworldai_assignment/videos/game1.mp4"
    if not Path(video_path).exists():
        with open("D:/chessworldai_assignment/chessvision-pgn/scratch/motion_events.txt", "w") as f:
            f.write("Video not found!\n")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total frames: {total_frames}")

    frame_step = 10  # process every 10th frame (3 frames per second)
    
    ret, prev_frame = cap.read()
    if not ret:
        with open("D:/chessworldai_assignment/chessvision-pgn/scratch/motion_events.txt", "w") as f:
            f.write("Failed to read video!\n")
        return

    # Resize to 160x120 and grayscale to make it super fast
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (160, 120))
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    motion_signals = []
    timestamps = []
    
    frame_idx = 1
    processed_count = 1

    while True:
        # Fast-forward frame_step - 1 frames
        for _ in range(frame_step - 1):
            cap.grab()
            frame_idx += 1
            
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        processed_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        diff = cv2.absdiff(gray, prev_gray)
        mean_diff = np.mean(diff)

        motion_signals.append(mean_diff)
        timestamps.append(frame_idx / fps)
        
        prev_gray = gray

    cap.release()

    # Smooth the signal slightly using a 3-element rolling window
    smoothed = np.convolve(motion_signals, np.ones(3)/3, mode='same')

    # Detect peaks/intervals of motion
    # Let's find the dynamic range of motion
    min_val = np.min(smoothed)
    max_val = np.max(smoothed)
    std_val = np.std(smoothed)
    mean_val = np.mean(smoothed)
    
    # Threshold is average motion plus a fraction of std deviation
    threshold = mean_val + 0.3 * std_val

    in_motion = False
    motion_start_idx = 0
    events = []

    for idx, val in enumerate(smoothed):
        t = timestamps[idx]
        if val > threshold and not in_motion:
            in_motion = True
            motion_start_idx = idx
        elif val < threshold and in_motion:
            in_motion = False
            duration = t - timestamps[motion_start_idx]
            if duration >= 0.5 and duration <= 8.0:
                events.append({
                    "start": timestamps[motion_start_idx],
                    "end": t,
                    "duration": duration,
                    "peak": float(np.max(smoothed[motion_start_idx:idx+1]))
                })

    # Write output to file
    out_path = "D:/chessworldai_assignment/chessvision-pgn/scratch/motion_events.txt"
    with open(out_path, "w") as f:
        f.write(f"Processed {processed_count} frames at step={frame_step} ({fps} FPS).\n")
        f.write(f"Motion Signal Stats: Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}, Std={std_val:.4f}, Threshold={threshold:.4f}\n")
        f.write(f"Detected {len(events)} candidate motion events:\n\n")
        for i, ev in enumerate(events):
            f.write(f"Move candidate {i+1}: {ev['start']:.2f}s -> {ev['end']:.2f}s (duration: {ev['duration']:.2f}s, peak={ev['peak']:.2f})\n")

    print(f"Analysis complete. Wrote {len(events)} events to {out_path}.")

if __name__ == "__main__":
    fast_analyze()
