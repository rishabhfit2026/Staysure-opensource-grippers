"""CLI: record a hand-motion session from camera."""

import logging
import signal
import sys
import time
from pathlib import Path

import click
import cv2
import yaml

from grippers.capture.camera import Camera
from grippers.data.session import SessionWriter
from grippers.perception.pipeline import PerceptionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--source", default=0, show_default=True, help="Camera index, RTSP URL, or video file path.")
@click.option("--output", default="sessions/session_001", show_default=True, help="Output directory for the session.")
@click.option("--duration", default=0, show_default=True, help="Recording duration in seconds (0 = unlimited).")
@click.option("--no-depth", is_flag=True, default=False, help="Disable depth estimation (faster, CPU-only).")
@click.option("--config", default=None, help="Path to YAML config file.")
@click.option("--show", is_flag=True, default=False, help="Show live camera window with overlay.")
def main(source, output, duration, no_depth, config, show):
    """Record a hand-motion session. Press Ctrl+C to stop."""

    cfg = {}
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)

    # Resolve source type
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # Keep as string (file/RTSP)

    stop = False

    def _handler(sig, frame):
        nonlocal stop
        stop = True
        logger.info("Stopping recording …")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    with Camera(source) as cam, \
         PerceptionPipeline(use_depth=not no_depth) as pipeline, \
         SessionWriter(output, fps=cam.fps, camera_source=str(source)) as writer:

        logger.info("Recording to %s  |  source=%s  |  depth=%s", output, source, not no_depth)
        logger.info("Press Ctrl+C to stop.")

        start = time.monotonic()
        frame_count = 0
        fps_clock = time.monotonic()
        fps_frames = 0

        while not stop:
            if duration > 0 and (time.monotonic() - start) >= duration:
                logger.info("Duration reached (%ds).", duration)
                break

            try:
                frame = cam.read(timeout=2.0)
            except TimeoutError:
                logger.warning("Frame timeout — camera may have disconnected.")
                break

            poses = pipeline.process(frame)
            writer.write(poses, frame)
            frame_count += 1
            fps_frames += 1

            if show:
                overlay = pipeline.draw_overlay(frame, poses)
                elapsed = time.monotonic() - fps_clock
                if elapsed >= 1.0:
                    fps = fps_frames / elapsed
                    fps_frames = 0
                    fps_clock = time.monotonic()
                    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Grippers — Recording", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if show:
            cv2.destroyAllWindows()

        logger.info("Session complete. %d frames recorded.", frame_count)


if __name__ == "__main__":
    main()
