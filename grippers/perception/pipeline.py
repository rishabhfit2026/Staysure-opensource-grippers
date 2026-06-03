"""PerceptionPipeline — wires detector + depth lifter into per-frame HandPose list."""

import logging
import time
from collections import deque
from typing import Optional

import numpy as np

from grippers.capture.camera import Frame
from grippers.perception.detector import HandDetector
from grippers.perception.depth import DepthLifter
from grippers.perception.pose import HandPose

logger = logging.getLogger(__name__)

_VELOCITY_WINDOW = 5  # frames for finite-difference velocity


class PerceptionPipeline:
    """
    Full perception pipeline: Frame → list[HandPose].

    Usage:
        pipeline = PerceptionPipeline()
        poses = pipeline.process(frame)
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        use_depth: bool = True,
        focal_px: float = 600.0,
        device: Optional[str] = None,
    ):
        self._detector = HandDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._lifter = DepthLifter(device=device) if use_depth else None
        self._focal_px = focal_px
        self._use_depth = use_depth

        # Per-hand wrist position history for velocity estimation
        self._wrist_history: dict[str, deque] = {
            "left": deque(maxlen=_VELOCITY_WINDOW),
            "right": deque(maxlen=_VELOCITY_WINDOW),
        }
        self._last_ts: dict[str, float] = {"left": 0.0, "right": 0.0}

    # ------------------------------------------------------------------
    def process(self, frame: Frame) -> list[HandPose]:
        detected = self._detector.detect(frame)

        depth_map = None
        if self._use_depth and self._lifter is not None:
            if any(v is not None for v in detected.values()):
                depth_map = self._lifter.estimate(frame.image)

        poses: list[HandPose] = []
        for side in ("left", "right"):
            lm_2d = detected.get(side)
            if lm_2d is None:
                continue

            if depth_map is not None and self._lifter is not None:
                landmarks_4 = self._lifter.lift_landmarks(
                    lm_2d,
                    depth_map,
                    frame.image.shape,
                    focal_px=self._focal_px,
                )
            else:
                # No depth — fill z with MediaPipe's relative z, confidence=1
                landmarks_4 = np.zeros((21, 4), dtype=np.float32)
                landmarks_4[:, :3] = lm_2d
                landmarks_4[:, 3] = 1.0

            wrist_vel = self._compute_velocity(side, landmarks_4[0, :3], frame.timestamp)
            grip_aperture = self._compute_grip_aperture(landmarks_4)

            pose = HandPose(
                timestamp=frame.timestamp,
                hand=side,
                landmarks=landmarks_4,
                wrist_velocity=wrist_vel,
                grip_aperture=grip_aperture,
                valid=True,
            )
            poses.append(pose)

        return poses

    def draw_overlay(self, frame: Frame, poses: list[HandPose]) -> np.ndarray:
        """Return annotated image with landmarks drawn."""
        detected = {p.hand: p.landmarks[:, :3] for p in poses}
        return self._detector.draw_landmarks(frame.image, detected)

    # ------------------------------------------------------------------
    def _compute_velocity(self, side: str, wrist_pos: np.ndarray, ts: float) -> np.ndarray:
        history = self._wrist_history[side]
        history.append((ts, wrist_pos.copy()))

        if len(history) < 2:
            return np.zeros(3, dtype=np.float32)

        t0, p0 = history[0]
        t1, p1 = history[-1]
        dt = t1 - t0
        if dt < 1e-6:
            return np.zeros(3, dtype=np.float32)
        return ((p1 - p0) / dt).astype(np.float32)

    def _compute_grip_aperture(self, landmarks_4: np.ndarray) -> float:
        thumb_tip = landmarks_4[HandPose.THUMB_TIP_IDX, :3]
        index_tip = landmarks_4[HandPose.INDEX_TIP_IDX, :3]
        wrist = landmarks_4[HandPose.WRIST_IDX, :3]
        hand_span = np.linalg.norm(landmarks_4[9, :3] - wrist) + 1e-6
        aperture = np.linalg.norm(thumb_tip - index_tip) / hand_span
        return float(np.clip(aperture, 0.0, 2.0))

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "PerceptionPipeline":
        return self

    def __exit__(self, *_) -> None:
        self.close()
