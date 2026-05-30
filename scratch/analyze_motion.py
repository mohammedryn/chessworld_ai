import cv2
import numpy as np
from pathlib import Path

def analyze_video_motion():
    video_path = "D:/chessworldai_assignment/chessvision-pgn/videos/game1.mp4"
    if not Path(video_path).exists():
        video_path = "D:/chessworldai_assignment/videos/game1.mp4"
    if not Path(video_path).exists():
        print("Video not found!")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames (metadata): {total_frames}")

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (15, 15), 0)

    motion_signals = []
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        # Frame difference
        diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        motion_pixel_ratio = np.mean(thresh) / 255.0

        motion_signals.append(motion_pixel_ratio)
        prev_gray = gray

    cap.release()
    print(f"Actually processed {frame_idx} frames.")

    # Threshold for motion detection
    # Let's smooth the motion signal to reduce noise
    window_size = int(fps) # 1 second window
    smoothed = np.convolve(motion_signals, np.ones(window_size)/window_size, mode='same')

    # Detect peaks / motion blocks
    threshold = 0.02 # 2% of pixels changing
    in_motion = False
    motion_start = 0
    motion_events = []

    for idx, val in enumerate(smoothed):
        if val > threshold and not in_motion:
            in_motion = True
            motion_start = idx
        elif val < threshold and in_motion:
            in_motion = False
            duration_secs = (idx - motion_start) / fps
            # Filter out very short motions (< 0.5s) or too long
            if duration_secs >= 0.5:
                motion_events.append({
                    "start_frame": motion_start,
                    "end_frame": idx,
                    "start_time": motion_start / fps,
                    "end_time": idx / fps,
                    "duration": duration_secs,
                    "peak_val": float(np.max(smoothed[motion_start:idx]))
                })

    print(f"\nDetected {len(motion_events)} significant motion events:")
    for i, ev in enumerate(motion_events):
        print(f"Event {i+1}: {ev['start_time']:.2f}s -> {ev['end_time']:.2f}s (duration: {ev['duration']:.2f}s, peak: {ev['peak_val']:.4f})")

if __name__ == "__main__":
    analyze_video_motion()
