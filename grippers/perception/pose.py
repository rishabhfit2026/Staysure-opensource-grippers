"""Core data contract: HandPose — the structured output of the perception pipeline."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class HandPose:
    timestamp: float
    hand: Literal["left", "right"]

    # (21, 4) — columns: x_norm, y_norm, z_lifted_m, confidence
    # x, y are normalised [0,1] relative to frame size
    # z_lifted_m is metric depth from Depth-Anything-V2
    landmarks: np.ndarray

    # (3,) m/s — finite-difference velocity of wrist keypoint
    wrist_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Normalised distance between thumb tip (4) and index tip (8)
    grip_aperture: float = 0.0

    # False when confidence is below threshold or hand is occluded
    valid: bool = True

    # ------------------------------------------------------------------
    LANDMARK_NAMES = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]

    THUMB_TIP_IDX = 4
    INDEX_TIP_IDX = 8
    WRIST_IDX = 0

    def wrist_position(self) -> np.ndarray:
        return self.landmarks[self.WRIST_IDX, :3]

    def thumb_tip(self) -> np.ndarray:
        return self.landmarks[self.THUMB_TIP_IDX, :3]

    def index_tip(self) -> np.ndarray:
        return self.landmarks[self.INDEX_TIP_IDX, :3]

    def mean_confidence(self) -> float:
        return float(self.landmarks[:, 3].mean())

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "hand": self.hand,
            "landmarks": self.landmarks,
            "wrist_velocity": self.wrist_velocity,
            "grip_aperture": self.grip_aperture,
            "valid": self.valid,
        }
