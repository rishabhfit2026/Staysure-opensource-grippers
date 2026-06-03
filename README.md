# Staysure OpenSource Grippers

**Vision-Based Humanoid Manipulation Learning System**

A zero-hardware-overhead framework that turns any head-mounted camera into a full hand-motion capture pipeline — then transfers the learned manipulation skills directly onto **humanoid robots** (simulation-first; Unitree G1/H1 and compatible platforms).

---

## The Core Idea

You strap a phone or action camera to your head, do your work with your hands, and walk away. The system handles the rest:

1. **Capture** — continuous video from any monocular camera mounted on a helmet, headband, or tripod
2. **Track** — real-time hand landmark detection and 3D pose estimation via neural networks
3. **Encode** — every grip, pinch, rotate, and slide is compressed into a transferable skill vector
4. **Transfer** — skill vectors are replayed on any robot gripper through an inverse-kinematics adapter

No gloves. No motion-capture suits. No specialised hardware beyond a camera you already own.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  DATA COLLECTION LAYER                  │
│  Phone / Action Cam  ──►  Video Stream  ──►  Frame Buffer│
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                  PERCEPTION LAYER                        │
│  Hand Detector (YOLOv8 / MediaPipe Holistic)             │
│  3D Landmark Estimator  ──►  Wrist + 21 keypoints        │
│  Depth Estimator (monocular depth / stereo optional)     │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                  LEARNING LAYER                          │
│  Temporal Encoder (Transformer / TCN)                    │
│  Skill Segmenter  ──►  Primitive action library          │
│  Imitation Learning (ACT / Diffusion Policy)             │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                  TRANSFER LAYER                          │
│  IK Solver  ──►  Gripper joint angles                    │
│  Robot Adapter (ROS2 / gRPC / USB serial)                │
│  Simulation Validator (Isaac Sim / MuJoCo)               │
└─────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Status |
|---|---|
| Monocular hand tracking (CPU-capable) | Planned |
| 21-keypoint 3D pose with depth estimation | Planned |
| Temporal skill segmentation | Planned |
| Skill library export / import | Planned |
| ROS2 gripper adapter | Planned |
| Simulation replay (MuJoCo) | Planned |
| Web UI for annotation and review | Planned |
| Multi-hand / bimanual support | Planned |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/rishabhfit2026/Staysure-opensource-grippers
cd Staysure-opensource-grippers

# 2. Install
pip install -e ".[dev]"

# 3. Record a session (webcam or RTSP phone stream)
python -m grippers.scripts.record --source 0 --output sessions/session_001 --show

# 4. Replay with landmark overlay
python -m grippers.scripts.visualize --session sessions/session_001

# 5. Run tests
pytest tests/
```

---

## Roadmap

### Phase 1 — Perception Foundation
- [ ] Camera ingestion module (RTSP, USB, file)
- [ ] MediaPipe Holistic integration for baseline hand tracking
- [ ] 3D landmark lifting with monocular depth (DPT / Depth-Anything)
- [ ] Data recording pipeline (HDF5 / MCAP)

### Phase 2 — Learning Engine
- [ ] Temporal Convolutional Network for motion encoding
- [ ] Unsupervised skill segmentation (change-point detection)
- [ ] ACT (Action Chunking with Transformers) policy training
- [ ] Diffusion Policy training pipeline

### Phase 3 — Transfer & Deployment
- [ ] Analytical IK solver for 2-finger / 3-finger grippers
- [ ] ROS2 node publishing joint trajectories
- [ ] MuJoCo simulation environment for skill validation
- [ ] Hardware abstraction layer (any URDF-described gripper)

### Phase 4 — Scale & Intelligence
- [ ] Multi-session skill library with semantic search
- [ ] Few-shot adaptation to new gripper morphologies
- [ ] On-device inference (Jetson / RK3588)
- [ ] Web dashboard for session review and labelling

---

## Tech Stack

| Domain | Library / Framework |
|---|---|
| Hand Tracking | MediaPipe, YOLOv8-Pose |
| Depth Estimation | Depth-Anything-V2, ZoeDepth |
| Policy Learning | ACT, Diffusion Policy, LeRobot |
| Simulation | MuJoCo, Isaac Sim |
| Robot Middleware | ROS2 Humble |
| Data Format | HDF5, MCAP |
| Training | PyTorch 2.x, Lightning |
| Serving | FastAPI, gRPC |

---

## Repository Layout

```
grippers/
├── capture/          # Camera ingestion, streaming, buffering
├── perception/       # Hand detection, 3D pose, depth
├── learning/         # Skill encoding, policy training
├── transfer/         # IK, robot adapters, simulation
├── data/             # Dataset utilities, HDF5 schema
├── configs/          # YAML config files
└── scripts/          # CLI entry points

sessions/             # Recorded sessions (gitignored)
models/               # Trained checkpoints (gitignored)
tests/
docs/
```

---

## Contributing

This project is fully open source. See [AGENTS.md](AGENTS.md) for how the AI agents that drive development are structured and what their goals are.

Pull requests, issues, and discussion are welcome.

---

## License

Apache 2.0 — free to use, modify, and deploy in commercial and research contexts.
