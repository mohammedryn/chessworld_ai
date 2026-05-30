from ultralytics import YOLO
import torch, shutil
from pathlib import Path


def main():
    torch.cuda.empty_cache()
    BASE = Path('D:/chessworldai_assignment/chessvision-pgn')

    # Fine-tune from existing trained model (faster convergence than yolov8n.pt)
    model = YOLO(str(BASE / 'models' / 'piece_detector.pt'))
    results = model.train(
        data=str(BASE / 'models/combined_dataset/data.yaml'),
        epochs=40,
        imgsz=640,
        batch=8,
        device=0,
        project=str(BASE / 'models/training'),
        name='chess_augmented_v1',
        exist_ok=True,
        verbose=False,
        patience=15,
        workers=4,
        # Stronger augmentations for angle robustness
        perspective=0.004,
        degrees=20,
        shear=10,
        scale=0.6,
        flipud=0.1,
    )

    best = Path(results.save_dir) / 'weights' / 'best.pt'
    dst = BASE / 'models' / 'piece_detector.pt'
    shutil.copy(best, dst)
    print(f'Saved to {dst}')
    print(f'Best mAP50: {results.results_dict.get("metrics/mAP50(B)", "N/A")}')


if __name__ == '__main__':
    main()
