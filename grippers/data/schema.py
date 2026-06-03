"""HDF5 schema constants for session storage."""

# HDF5 group structure:
#   /meta          — scalar attributes: fps, camera_source, start_time, grippers_version
#   /poses/left/   — per-field datasets for left hand
#   /poses/right/  — per-field datasets for right hand
#   /frames/       — compressed JPEG bytes stored as variable-length uint8

POSES_GROUP = "poses"
FRAMES_GROUP = "frames"
META_GROUP = "meta"

POSE_FIELDS = {
    "timestamp":       ("f8",  ()),       # float64 scalar
    "landmarks":       ("f4",  (21, 4)),  # float32 (21, 4)
    "wrist_velocity":  ("f4",  (3,)),     # float32 (3,)
    "grip_aperture":   ("f4",  ()),       # float32 scalar
    "valid":           ("?",   ()),       # bool scalar
}

FLUSH_EVERY = 100  # frames between HDF5 flushes
JPEG_QUALITY = 80
