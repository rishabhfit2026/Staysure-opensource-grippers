"""Depth-Anything-V2 wrapper — lifts 2D landmarks to metric 3D using monocular depth."""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_DEFAULT_FOCAL_PX = 600.0  # Reasonable default for 640×480; override via camera intrinsics


class DepthLifter:
    """
    Loads Depth-Anything-V2-Small and uses its output to assign metric z values
    to 2D hand landmarks.

    The model produces relative depth (larger = farther). We rescale to approximate
    metric depth using a scale factor estimated from hand size heuristic or provided
    camera intrinsics.
    """

    def __init__(self, device: Optional[str] = None):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None  # Lazy load

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading Depth-Anything-V2-Small on %s …", self._device)
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=_MODEL_ID,
                device=0 if self._device == "cuda" else -1,
            )
            logger.info("Depth model loaded.")
        except Exception as e:
            logger.error("Failed to load depth model: %s", e)
            raise

    # ------------------------------------------------------------------
    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on a BGR image.
        Returns a (H, W) float32 depth map normalised to [0, 1].
        Larger value = farther from camera.
        """
        self._load()
        from PIL import Image as PILImage
        rgb = image_bgr[:, :, ::-1]
        pil_img = PILImage.fromarray(rgb.astype(np.uint8))
        result = self._pipe(pil_img)
        depth = np.array(result["depth"], dtype=np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        return depth

    def lift_landmarks(
        self,
        landmarks_2d: np.ndarray,
        depth_map: np.ndarray,
        frame_shape: tuple[int, int],
        focal_px: float = _DEFAULT_FOCAL_PX,
        hand_scale_m: float = 0.18,
    ) -> np.ndarray:
        """
        Given (21, 3) landmarks in normalised [0,1] coords and a (H,W) depth map,
        return (21, 4) array: x_norm, y_norm, z_metric_m, confidence.

        z is estimated by:
          1. Sample depth_map at each landmark pixel
          2. Compute median depth across all 21 landmarks
          3. Scale so the wrist-to-middle-finger distance ≈ hand_scale_m
        """
        h, w = frame_shape[:2]
        result = np.zeros((21, 4), dtype=np.float32)
        result[:, 0] = landmarks_2d[:, 0]
        result[:, 1] = landmarks_2d[:, 1]
        result[:, 3] = 1.0  # confidence placeholder

        # Sample depth at each landmark position
        px = np.clip((landmarks_2d[:, 0] * w).astype(int), 0, w - 1)
        py = np.clip((landmarks_2d[:, 1] * h).astype(int), 0, h - 1)
        raw_z = depth_map[py, px]  # (21,) in [0,1]

        # Anchor: scale using span of wrist (idx 0) to middle MCP (idx 9)
        # Expected real distance ~hand_scale_m / 2
        wrist = landmarks_2d[0, :2]
        middle_mcp = landmarks_2d[9, :2]
        pixel_span = np.linalg.norm((wrist - middle_mcp) * np.array([w, h]))

        if pixel_span > 1e-3:
            # pinhole: Z = f * real_size / pixel_size
            z_anchor = focal_px * (hand_scale_m / 2.0) / pixel_span
        else:
            z_anchor = 0.5  # fallback 50 cm

        # Normalise raw_z so median aligns with z_anchor
        median_raw = float(np.median(raw_z)) + 1e-6
        scale = z_anchor / median_raw
        result[:, 2] = raw_z * scale

        return result
