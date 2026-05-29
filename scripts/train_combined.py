from ultralytics import YOLO
import torch, shutil
from pathlib import Path


def main():
    torch.cuda.empty_cache()
    BASE = Path('D:/chessworldai_assignment/chessvision-pgn')

    model = YOLO('yolov8n.pt')
    results = model.train(
        data=str(BASE / 'models/combined_dataset/data.yaml'),
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        project=str(BASE / 'models/training'),
        name='chess_combined_v1',
        exist_ok=True,
        verbose=False,
        patience=10,
        workers=4,
    )

    best = Path(results.save_dir) / 'weights' / 'best.pt'
    dst = BASE / 'models' / 'piece_detector.pt'
    shutil.copy(best, dst)
    print(f'Saved to {dst}')
    print(f'Best mAP50: {results.results_dict.get("metrics/mAP50(B)", "N/A")}')


if __name__ == '__main__':
    main()
