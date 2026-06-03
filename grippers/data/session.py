"""HDF5-based session writer and reader for recorded hand-pose sessions."""

import logging
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import h5py
import numpy as np

from grippers.capture.camera import Frame
from grippers.perception.pose import HandPose
from grippers.data.schema import (
    FLUSH_EVERY, FRAMES_GROUP, JPEG_QUALITY, META_GROUP, POSE_FIELDS, POSES_GROUP
)
import grippers

logger = logging.getLogger(__name__)


class SessionWriter:
    """Write a stream of HandPose objects + frames to an HDF5 file incrementally."""

    def __init__(self, path: str | Path, fps: float = 30.0, camera_source: str = ""):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._hdf5_path = self._path / "session.h5"

        self._f = h5py.File(self._hdf5_path, "w")
        self._frame_count = 0

        # Meta
        meta = self._f.create_group(META_GROUP)
        meta.attrs["fps"] = fps
        meta.attrs["camera_source"] = camera_source
        meta.attrs["start_time"] = time.time()
        meta.attrs["grippers_version"] = grippers.__version__

        # Pose datasets (extendable)
        self._pose_ds: dict[str, dict[str, h5py.Dataset]] = {}
        for side in ("left", "right"):
            grp = self._f.create_group(f"{POSES_GROUP}/{side}")
            self._pose_ds[side] = {}
            for field, (dtype, shape) in POSE_FIELDS.items():
                ds = grp.create_dataset(
                    field,
                    shape=(0,) + shape,
                    maxshape=(None,) + shape,
                    dtype=dtype,
                    chunks=(256,) + shape,
                    compression="gzip",
                    compression_opts=4,
                )
                self._pose_ds[side][field] = ds

        # Frame dataset — variable-length bytes
        vlen = h5py.special_dtype(vlen=np.dtype("uint8"))
        self._frame_ds = self._f.create_dataset(
            f"{FRAMES_GROUP}/jpeg",
            shape=(0,),
            maxshape=(None,),
            dtype=vlen,
        )
        self._frame_ts_ds = self._f.create_dataset(
            f"{FRAMES_GROUP}/timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype="f8",
        )

    # ------------------------------------------------------------------
    def write(self, poses: list[HandPose], frame: Optional[Frame] = None) -> None:
        for pose in poses:
            side = pose.hand
            ds = self._pose_ds[side]
            n = ds["timestamp"].shape[0]

            for field in POSE_FIELDS:
                ds[field].resize(n + 1, axis=0)
                value = getattr(pose, field)
                if field == "hand":
                    continue
                ds[field][n] = value

        if frame is not None:
            n = self._frame_ds.shape[0]
            self._frame_ds.resize(n + 1, axis=0)
            self._frame_ts_ds.resize(n + 1, axis=0)
            _, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            self._frame_ds[n] = encoded.flatten()
            self._frame_ts_ds[n] = frame.timestamp
            self._frame_count += 1

        if self._frame_count % FLUSH_EVERY == 0:
            self._f.flush()

    def close(self) -> None:
        self._f.flush()
        self._f.close()
        logger.info("Session written to %s (%d frames)", self._hdf5_path, self._frame_count)

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------

class SessionReader:
    """Iterate over HandPose objects stored in a session HDF5 file."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        hdf5_path = self._path / "session.h5"
        if not hdf5_path.exists():
            raise FileNotFoundError(f"No session.h5 found in {self._path}")
        self._f = h5py.File(hdf5_path, "r")

    # ------------------------------------------------------------------
    def poses(self, side: str = "right") -> Iterator[HandPose]:
        grp = self._f[f"{POSES_GROUP}/{side}"]
        n = grp["timestamp"].shape[0]
        for i in range(n):
            yield HandPose(
                timestamp=float(grp["timestamp"][i]),
                hand=side,
                landmarks=grp["landmarks"][i],
                wrist_velocity=grp["wrist_velocity"][i],
                grip_aperture=float(grp["grip_aperture"][i]),
                valid=bool(grp["valid"][i]),
            )

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yields (timestamp, bgr_image) tuples."""
        grp = self._f[FRAMES_GROUP]
        n = grp["jpeg"].shape[0]
        for i in range(n):
            ts = float(grp["timestamp"][i])
            buf = np.frombuffer(bytes(grp["jpeg"][i]), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            yield ts, img

    @property
    def meta(self) -> dict:
        return dict(self._f[META_GROUP].attrs)

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "SessionReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()
