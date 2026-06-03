"""MediaPipe Holistic wrapper — extracts raw 2D hand landmarks from a frame."""

import logging
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from grippers.capture.camera import Frame

logger = logging.getLogger(__name__)

_HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


class HandDetector:
    """Thin wrapper around MediaPipe Holistic for hand landmark extraction."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._conf_threshold = min_detection_confidence

    # ------------------------------------------------------------------
    def detect(self, frame: Frame) -> dict[str, Optional[np.ndarray]]:
        """
        Returns dict with keys "left" and "right".
        Each value is either None (not detected) or np.ndarray of shape (21, 3):
          columns: x_norm, y_norm, z_relative (MediaPipe's raw z, not metric).
        """
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(rgb)

        output: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}

        for side, landmarks in (
            ("left", results.left_hand_landmarks),
            ("right", results.right_hand_landmarks),
        ):
            if landmarks is None:
                continue
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                dtype=np.float32,
            )
            output[side] = pts

        return output

    def draw_landmarks(self, image: np.ndarray, detected: dict) -> np.ndarray:
        """Draw hand landmarks onto a copy of image for visualisation."""
        annotated = image.copy()
        mp_draw = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles

        for side, pts in detected.items():
            if pts is None:
                continue
            # Reconstruct a NormalizedLandmarkList for drawing
            lm_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
            for x, y, z in pts:
                lm = lm_list.landmark.add()
                lm.x, lm.y, lm.z = float(x), float(y), float(z)
            mp_draw.draw_landmarks(
                annotated,
                lm_list,
                _HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
        return annotated

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
