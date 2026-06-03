"""Camera ingestion — USB, RTSP, and file sources with a threaded frame buffer."""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    index: int


class Camera:
    """Threaded camera reader. Keeps a small queue; drops oldest frame on overflow."""

    def __init__(self, source: Union[int, str], buffer_size: int = 4):
        self._source = source
        self._buffer: queue.Queue[Frame] = queue.Queue(maxsize=buffer_size)
        self._index = 0
        self._dropped = 0
        self._stop_event = threading.Event()

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source!r}")

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def _reader(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                if isinstance(self._source, str):
                    logger.info("End of video file reached.")
                else:
                    logger.warning("Camera read failed (ret=False). Retrying.")
                time.sleep(0.01)
                continue

            f = Frame(image=frame, timestamp=time.monotonic(), index=self._index)
            self._index += 1

            if self._buffer.full():
                try:
                    self._buffer.get_nowait()
                    self._dropped += 1
                    logger.debug("Dropped frame %d (buffer full, total dropped=%d)", f.index, self._dropped)
                except queue.Empty:
                    pass

            self._buffer.put(f)

    # ------------------------------------------------------------------
    def read(self, timeout: float = 1.0) -> Frame:
        """Block until a frame is available, then return it."""
        try:
            return self._buffer.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("No frame received within timeout.")

    def is_open(self) -> bool:
        return self._cap.isOpened() and not self._stop_event.is_set()

    def release(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._cap.release()
        logger.info("Camera released. Total dropped frames: %d", self._dropped)

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def resolution(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h
