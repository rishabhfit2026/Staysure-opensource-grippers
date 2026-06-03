"""Unit tests for HDF5 session writer and reader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from grippers.capture.camera import Frame
from grippers.data.session import SessionWriter, SessionReader
from grippers.perception.pose import HandPose


def _make_pose(ts: float, side: str = "right") -> HandPose:
    lm = np.random.default_rng(int(ts * 1000)).uniform(0, 1, (21, 4)).astype(np.float32)
    return HandPose(
        timestamp=ts,
        hand=side,
        landmarks=lm,
        wrist_velocity=np.zeros(3, dtype=np.float32),
        grip_aperture=0.3,
        valid=True,
    )


def _make_frame(ts: float, idx: int) -> Frame:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (idx % 255, 100, 50)
    return Frame(image=img, timestamp=ts, index=idx)


def test_write_and_read_poses():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir) / "test_session"

        poses_written = [_make_pose(float(i) * 0.033, "right") for i in range(10)]

        with SessionWriter(session_path, fps=30.0) as writer:
            for pose in poses_written:
                writer.write([pose])

        with SessionReader(session_path) as reader:
            poses_read = list(reader.poses("right"))

        assert len(poses_read) == 10
        for orig, read in zip(poses_written, poses_read):
            assert abs(orig.timestamp - read.timestamp) < 1e-5
            assert orig.hand == read.hand
            np.testing.assert_allclose(orig.landmarks, read.landmarks, atol=1e-5)
            assert abs(orig.grip_aperture - read.grip_aperture) < 1e-5


def test_write_and_read_frames():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir) / "test_session"

        frames = [_make_frame(float(i) * 0.033, i) for i in range(5)]
        poses = [_make_pose(float(i) * 0.033) for i in range(5)]

        with SessionWriter(session_path, fps=30.0) as writer:
            for pose, frame in zip(poses, frames):
                writer.write([pose], frame)

        with SessionReader(session_path) as reader:
            frames_read = list(reader.frames())

        assert len(frames_read) == 5
        for (ts, img) in frames_read:
            assert img.shape == (480, 640, 3)


def test_meta_stored():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir) / "test_session"
        with SessionWriter(session_path, fps=25.0, camera_source="test_cam") as writer:
            writer.write([_make_pose(0.0)])

        with SessionReader(session_path) as reader:
            meta = reader.meta
            assert meta["fps"] == 25.0
            assert meta["camera_source"] == "test_cam"
            assert "start_time" in meta
            assert "grippers_version" in meta


def test_empty_poses_no_crash():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir) / "test_session"
        frame = _make_frame(0.0, 0)
        with SessionWriter(session_path) as writer:
            writer.write([], frame)  # No poses, just a frame

        with SessionReader(session_path) as reader:
            frames = list(reader.frames())
            assert len(frames) == 1
