"""Unit tests for perception layer — no camera or GPU required."""

import numpy as np
import pytest

from grippers.perception.pose import HandPose
from grippers.perception.depth import DepthLifter


def _fake_landmarks_2d() -> np.ndarray:
    rng = np.random.default_rng(42)
    pts = rng.uniform(0.2, 0.8, (21, 3)).astype(np.float32)
    pts[:, 2] *= 0.1  # small z
    return pts


def test_handpose_fields():
    lm = np.zeros((21, 4), dtype=np.float32)
    pose = HandPose(timestamp=1.0, hand="right", landmarks=lm)
    assert pose.valid is True
    assert pose.grip_aperture == 0.0
    assert pose.wrist_position().shape == (3,)
    assert pose.mean_confidence() == 0.0


def test_handpose_to_dict():
    lm = np.zeros((21, 4), dtype=np.float32)
    pose = HandPose(timestamp=2.5, hand="left", landmarks=lm)
    d = pose.to_dict()
    assert d["hand"] == "left"
    assert d["timestamp"] == 2.5
    assert d["landmarks"].shape == (21, 4)


def test_depth_lifter_no_gpu():
    """DepthLifter.lift_landmarks should work without loading the neural model."""
    lifter = DepthLifter(device="cpu")

    lm_2d = _fake_landmarks_2d()
    depth_map = np.random.default_rng(0).uniform(0.0, 1.0, (480, 640)).astype(np.float32)

    result = lifter.lift_landmarks(lm_2d, depth_map, (480, 640))
    assert result.shape == (21, 4)
    assert np.all(result[:, 3] == 1.0), "confidence should be 1.0"
    assert np.all(result[:, 2] > 0), "z should be positive"
    assert np.allclose(result[:, 0], lm_2d[:, 0]), "x should be preserved"
    assert np.allclose(result[:, 1], lm_2d[:, 1]), "y should be preserved"


def test_grip_aperture_range():
    lm = np.zeros((21, 4), dtype=np.float32)
    # wrist at origin, middle MCP at (0, 0.1, 0), thumb and index tips separated
    lm[9] = [0.0, 0.1, 0.0, 1.0]
    lm[4] = [0.05, 0.05, 0.0, 1.0]  # thumb tip
    lm[8] = [0.15, 0.05, 0.0, 1.0]  # index tip
    pose = HandPose(timestamp=0.0, hand="right", landmarks=lm)
    # We only test that it doesn't crash and stays in reasonable range
    assert 0.0 <= pose.grip_aperture <= 2.0 or pose.grip_aperture == 0.0
