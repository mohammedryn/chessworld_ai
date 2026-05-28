# ChessVision PGN

Convert over-the-board chess game videos into PGN notation using computer vision.

## Pipeline Overview

```
Video → Optical Flow Gate → Hand Occlusion Filter → Board Detection (Homography)
     → YOLOv8 Piece Detection → Sliding Window Vote → python-chess Move Validation → PGN
```

## Dependencies

- Python 3.10+
- CUDA 12.x (NVIDIA GPU recommended — tested on RTX 4050 6GB)
- See `requirements.txt` for full Python package list

## Setup

```bash
# 1. Install PyTorch with CUDA support (must be before other deps)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install remaining dependencies
pip install -r requirements.txt

# 3. Download the pre-trained chess piece detection model
#    Get a free API key at https://roboflow.com -> Settings -> API Keys
python scripts/download_models.py --api-key YOUR_API_KEY
```

## Usage

```bash
# Single video -> PGN
python main.py --input game1.mp4 --output output/game1.pgn

# Batch: process a folder of videos
python main.py --input videos/ --output output/

# Real-time 3-panel demo (original video | warped board | live PGN)
python main.py --input game1.mp4 --output output/game1.pgn --demo

# Demo mode + save composite video for submission
python main.py --input game1.mp4 --output output/game1.pgn --demo --save-demo output/demo.mp4
```

## Detection / Model Pipeline

| Stage | Method | Why |
| --- | --- | --- |
| Frame selection | Farneback dense optical flow | Skip ~95% of static frames — only process frames where something moved |
| Hand occlusion | MediaPipe Hands | Discard frames while a hand is over the board (mid-move) |
| Board detection | OpenCV contour detection → largest quadrilateral → perspective homography | White border is distinctive; homography locked for 60 frames for efficiency |
| Piece detection | YOLOv8s pre-trained on Roboflow Chess Pieces dataset | ~95% mAP on standard Staunton plastic pieces, CUDA-accelerated |
| Move detection | 3-frame sliding window plurality vote + python-chess legal move filter | Eliminates single-frame glitches; chess rules remove false positives |

## Tests

```bash
pytest -v
```

24 unit tests covering: optical flow gate, board corner detection, piece grid mapping, board state machine (normal moves, castling, en passant, majority vote noise), and PGN assembly.

## Output

- `output/gameN.pgn` — PGN notation for each game
- `output/demo_gameN.mp4` — 3-panel composite demo video (original | warped | live PGN)
- `pipeline.log` — structured JSON event log (one entry per detected move or error)
