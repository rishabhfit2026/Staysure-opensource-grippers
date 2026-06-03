"""CLI: replay a recorded session with landmark overlay."""

import logging
import time

import click
import cv2
import numpy as np

from grippers.data.session import SessionReader
from grippers.perception.detector import HandDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--session", required=True, help="Path to session directory.")
@click.option("--speed", default=1.0, show_default=True, help="Playback speed multiplier.")
@click.option("--side", default="both", type=click.Choice(["left", "right", "both"]), show_default=True)
def main(session, speed, side):
    """Replay a recorded session with hand landmark overlay."""

    with SessionReader(session) as reader:
        meta = reader.meta
        fps = float(meta.get("fps", 30.0))
        logger.info("Session meta: %s", meta)

        detector = HandDetector()
        frame_delay = 1.0 / (fps * speed)

        sides = ["left", "right"] if side == "both" else [side]

        # Build a pose lookup by frame timestamp
        pose_by_ts: dict[float, list] = {}
        for s in sides:
            try:
                for pose in reader.poses(s):
                    pose_by_ts.setdefault(pose.timestamp, []).append(pose)
            except KeyError:
                logger.warning("No poses found for side: %s", s)

        logger.info("Press 'q' to quit.")
        prev_ts = None

        for ts, img in reader.frames():
            t0 = time.monotonic()

            # Draw poses if available for this timestamp
            poses = pose_by_ts.get(ts, [])
            if poses:
                detected = {p.hand: p.landmarks[:, :3] for p in poses}
                img = detector.draw_landmarks(img, detected)

                for p in poses:
                    label = f"{p.hand.upper()}  aperture={p.grip_aperture:.2f}"
                    cv2.putText(img, label, (10, 60 if p.hand == "left" else 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Grippers — Replay", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elapsed = time.monotonic() - t0
            wait = frame_delay - elapsed
            if wait > 0:
                time.sleep(wait)

        cv2.destroyAllWindows()
        detector.close()
        logger.info("Playback complete.")


if __name__ == "__main__":
    main()
