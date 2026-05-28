import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
from typing import Optional

# MediaPipe 0.10+ uses the Tasks API; the legacy mp.solutions.hands is no longer available.
# A hand landmarker model file is required.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "hand_landmarker.task"
)


def _ensure_model(model_path: str) -> str:
    """Download the hand landmarker model if not already present."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    return model_path


class HandDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ):
        if model_path is None:
            model_path = os.path.normpath(_DEFAULT_MODEL_PATH)

        resolved = _ensure_model(model_path)

        VisionRunningMode = mp.tasks.vision.RunningMode
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=resolved),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def has_occlusion(
        self,
        frame: np.ndarray,
        board_corners: Optional[np.ndarray],
    ) -> bool:
        """Return True if any hand landmark falls inside the board quad."""
        if board_corners is None:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return False

        h, w = frame.shape[:2]
        poly = board_corners.reshape((-1, 1, 2)).astype(np.float32)

        for hand_landmarks in results.hand_landmarks:
            for lm in hand_landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                if cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0:
                    return True
        return False

    def close(self):
        self._landmarker.close()
