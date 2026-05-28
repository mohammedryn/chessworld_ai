"""Download pre-trained chess piece detection model from Roboflow.

Usage:
    python scripts/download_models.py --api-key YOUR_ROBOFLOW_API_KEY

Get a free API key at: https://roboflow.com (sign up -> Settings -> API Keys)
Model: https://universe.roboflow.com/chess-pieces-detection-pn0jv/chess-pieces-new-version
"""
import argparse
import shutil
from pathlib import Path
from roboflow import Roboflow


def download(api_key: str):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("chess-pieces-detection-pn0jv").project("chess-pieces-new-version")
    version = project.version(2)
    version.download("yolov8", location="models/roboflow_download")

    weights_src = Path("models/roboflow_download/weights/best.pt")
    weights_dst = Path("models/piece_detector.pt")
    weights_dst.parent.mkdir(exist_ok=True)
    shutil.copy(weights_src, weights_dst)
    print(f"Model saved to {weights_dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    args = parser.parse_args()
    download(args.api_key)
