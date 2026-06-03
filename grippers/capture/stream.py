"""FrameBuffer — sliding window of recent frames with drop tracking."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from grippers.capture.camera import Frame


@dataclass
class BufferStats:
    total_received: int = 0
    total_dropped: int = 0
    total_processed: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def drop_rate(self) -> float:
        if self.total_received == 0:
            return 0.0
        return self.total_dropped / self.total_received

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def fps(self) -> float:
        elapsed = self.elapsed
        if elapsed < 1e-6:
            return 0.0
        return self.total_processed / elapsed


class FrameBuffer:
    """
    Sliding window buffer of the last N frames.
    Useful for temporal models that need access to recent history.
    """

    def __init__(self, maxlen: int = 30):
        self._buf: deque[Frame] = deque(maxlen=maxlen)
        self.stats = BufferStats()

    def push(self, frame: Frame) -> None:
        self.stats.total_received += 1
        if len(self._buf) == self._buf.maxlen:
            self.stats.total_dropped += 1
        self._buf.append(frame)

    def latest(self) -> Optional[Frame]:
        return self._buf[-1] if self._buf else None

    def window(self, n: int) -> list[Frame]:
        """Return the last n frames (oldest first)."""
        frames = list(self._buf)
        return frames[-n:] if len(frames) >= n else frames

    def mark_processed(self) -> None:
        self.stats.total_processed += 1

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)
