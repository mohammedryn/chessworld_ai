"""Download chess piece training dataset from Roboflow and train YOLOv8n.

Usage:
    # Step 1: Download dataset
    python scripts/download_models.py --api-key YOUR_ROBOFLOW_API_KEY

    # Step 2: Train the model (run separately or add --train flag)
    python scripts/download_models.py --api-key YOUR_ROBOFLOW_API_KEY --train

Get a free API key at: https://roboflow.com (sign up -> Settings -> API Keys)
Dataset: https://universe.roboflow.com/chesscam/chesscam-dh33p/dataset/4
"""
import argparse
import shutil
import zipfile
import io
from pathlib import Path

import requests
import yaml


def download_dataset(api_key: str) -> Path:
    """Download chesscam-dh33p v4 dataset to models/dataset/."""
    print("Fetching dataset export link...")
    r = requests.get(
        f"https://api.roboflow.com/chesscam/chesscam-dh33p/4/yolov8?api_key={api_key}",
        timeout=15,
    )
    r.raise_for_status()
    link = r.json().get("export", {}).get("link", "")
    if not link:
        raise RuntimeError("No export link returned — try again in a few seconds.")

    print("Downloading dataset (~190 MB)...")
    resp = requests.get(link, timeout=300, stream=True)
    resp.raise_for_status()
    data = resp.content
    if data[:2] != b"PK":
        raise RuntimeError(f"Download returned non-ZIP content (HTTP {resp.status_code}).")

    dataset_dir = Path("models/dataset")
    shutil.rmtree(dataset_dir, ignore_errors=True)
    dataset_dir.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dataset_dir)

    # Fix data.yaml paths to absolute
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    abs_base = dataset_dir.resolve()
    d["train"] = str(abs_base / "train" / "images")
    d["val"] = str(abs_base / "valid" / "images")
    d["test"] = str(abs_base / "test" / "images")
    with open(yaml_path, "w") as f:
        yaml.dump(d, f)

    print(f"Dataset saved to {dataset_dir}/")
    return yaml_path


def train(yaml_path: Path):
    """Fine-tune YOLOv8n on the chess piece dataset."""
    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training YOLOv8n on {'GPU' if device == 0 else 'CPU'}...")

    model = YOLO(str(Path("models/yolov8n.pt")))
    results = model.train(
        data=str(yaml_path),
        epochs=30,
        imgsz=640,
        batch=16,
        device=device,
        project="models/training",
        name="chess_v1",
        exist_ok=True,
        verbose=False,
        patience=10,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    dst = Path("models/piece_detector.pt")
    shutil.copy(best, dst)
    print(f"Model saved to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--train", action="store_true", help="Also train the model after download")
    args = parser.parse_args()

    yaml_path = download_dataset(args.api_key)
    if args.train:
        train(yaml_path)
