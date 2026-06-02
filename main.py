#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.pipeline import ChessVisionPipeline, BoardNotFoundError


def main():
    parser = argparse.ArgumentParser(
        description="Convert chess game video(s) to PGN notation."
    )
    parser.add_argument("--input", required=True,
                        help="Path to a .mp4 video file or directory of videos.")
    parser.add_argument("--output", required=True,
                        help="Output .pgn file, or directory when --input is a directory.")
    parser.add_argument("--model", default="models/piece_detector.pt",
                        help="Path to YOLOv8 piece detection weights.")
    parser.add_argument("--demo", action="store_true",
                        help="Show live 3-panel demo window during processing.")
    parser.add_argument("--save-demo", metavar="PATH",
                        help="Save demo composite video to PATH.")
    parser.add_argument("--log", default="pipeline.log",
                        help="Structured JSON log file (default: pipeline.log).")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="YOLO detection confidence threshold (default: 0.25). "
                             "Raise to 0.35-0.40 for noisy CCTV footage.")
    parser.add_argument("--min-frame-gap", type=int, default=60,
                        help="Minimum video frames between accepted moves (default: 60 = 2s). "
                             "Raise to 150-300 for slow tournament play.")
    parser.add_argument("--move-threshold", type=int, default=32,
                        help="Minimum occupancy match score to accept a move out of 64 "
                             "(default: 32). Raise to 38-42 for stricter filtering.")
    args = parser.parse_args()

    pipeline = ChessVisionPipeline(
        piece_model_path=args.model,
        log_path=args.log,
        confidence=args.confidence,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        if input_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            videos = sorted(
                list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi"))
            )
            if not videos:
                print(f"No .mp4 or .avi files found in {input_path}")
                return
            for video in videos:
                out = output_path / (video.stem + ".pgn")
                demo_path = (
                    str(output_path / ("demo_" + video.stem + ".mp4"))
                    if args.save_demo else None
                )
                print(f"\n-> {video.name}")
                try:
                    pipeline.process_video(
                        str(video), str(out),
                        demo=args.demo, save_demo=demo_path,
                        min_frame_gap=args.min_frame_gap,
                        move_threshold=args.move_threshold,
                    )
                    print(f"   Saved: {out}")
                except BoardNotFoundError as e:
                    print(f"   ERROR: {e}")
        else:
            pipeline.process_video(
                str(input_path), str(output_path),
                demo=args.demo, save_demo=args.save_demo,
                min_frame_gap=args.min_frame_gap,
                move_threshold=args.move_threshold,
            )
            print(f"PGN saved to {output_path}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
