# Video → 3D Scene Reconstruction

A feed-forward pipeline that takes a short indoor phone video and produces a geometrically coherent 3D scene — with optional open-vocabulary semantic labels.

Built for the [Humanoid](https://humanoid.com) Perception & Spatial AI internship challenge.

---

## Demo

```
python run.py examples/room.mp4 --semantic
```

<!-- Replace with your actual output screenshots/GIFs once you have run the pipeline -->
> *Example outputs go here — point cloud render, depth maps, semantic overlay*

---

## What it does

| Stage | What happens |
|---|---|
| **1. Frame extraction** | Samples N frames from the video using farthest-point sampling on appearance thumbnails — maximises visual diversity, skips near-duplicate frames where the camera barely moved |
| **2. Feed-forward reconstruction** | VGGT-Omega / VGGT-1B predicts depth maps, camera poses, and dense 3D point maps in a single forward pass — no COLMAP, no iterative optimisation |
| **3. Point cloud post-processing** | Open3D merges per-frame point maps, filters by confidence, and removes statistical outliers |
| **4. Semantic labeling** *(optional)* | SAM 2.1 segments each frame; OpenCLIP encodes every segment; labels are fused into 3D by multi-view voting with a z-buffer occlusion test |
| **5. Visualization** | Rerun opens an interactive browser viewer showing the RGB frames, depth maps, camera trajectories, and 3D scene |
| **6. Open-vocabulary query** *(optional)* | `query.py` searches the reconstructed scene with *any* natural-language phrase — not just the label set — and renders a 3D relevancy heatmap |

---

## Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- **macOS** (Apple Silicon M1–M4) **or** **Linux with an NVIDIA GPU**

### Step 1 — Create the conda environment

```bash
conda env create -f environment.yml
conda activate video-to-3d
```

### Step 2 — Install PyTorch and model packages

```bash
bash install.sh
```

This script auto-detects your hardware and installs:
- The correct PyTorch build (CUDA 12.1 / MPS / CPU)
- VGGT-1B (CVPR 2025 Best Paper)
- VGGT-Omega (CVPR 2026 Oral) — CUDA only
- SAM 2.1

### Step 3 — (Optional) Pre-download model weights

```bash
python scripts/download_models.py
```

This downloads ~5 GB of model weights from HuggingFace so the first run starts immediately. If you skip this step the weights are downloaded automatically on first use.

---

## Usage

```bash
# Basic reconstruction (geometry only)
python run.py path/to/video.mp4

# With semantic labels
python run.py path/to/video.mp4 --semantic

# More frames for a denser cloud (recommended on CUDA)
python run.py path/to/video.mp4 --frames 80 --semantic

# Custom label vocabulary
python run.py path/to/video.mp4 --semantic \
    --labels "chair,table,sofa,door,window,lamp,monitor"

# Generate a surface mesh as well
python run.py path/to/video.mp4 --mesh

# Force a specific device
python run.py path/to/video.mp4 --device cuda   # Linux workstation
python run.py path/to/video.mp4 --device mps    # MacBook
python run.py path/to/video.mp4 --device cpu    # no GPU

# Just save files, skip the interactive viewer
python run.py path/to/video.mp4 --no-viewer
```

### Search the scene in natural language

Any run made with `--semantic` saves per-segment CLIP features alongside the
geometry, so the reconstructed scene becomes *queryable* — with arbitrary
phrases, not just the labels used at reconstruction time:

```bash
python query.py outputs/room_20260612_010453 "office chair"
python query.py outputs/room_20260612_010453 "something to sit on" "coffee mug"
```

Each query takes milliseconds (no re-reconstruction) and produces a 3D
relevancy heatmap — shown in the Rerun viewer next to the RGB cloud and saved
as `query_<phrase>.ply`. Relevancy is computed LERF-style against canonical
negative prompts, so the heatmap highlights *how query-like* each region is
rather than raw similarity.

### All flags

| Flag | Default | Description |
|---|---|---|
| `video` | — | Input video path (MP4, MOV, AVI) |
| `--output` | `outputs/<video>_<timestamp>` | Output directory — auto-named so runs never overwrite each other |
| `--frames` | `50` | Frames to sample (auto-capped at 20 on MPS; 80–100 recommended on CUDA) |
| `--model` | `auto` | `vggt-omega` on CUDA, `vggt` elsewhere |
| `--device` | auto | Force `cuda` / `mps` / `cpu` |
| `--conf-percentile` | `20` | Drop the least-confident X% of points — scene-adaptive, no per-scene tuning needed |
| `--semantic` | off | Enable SAM2 + CLIP semantic labeling |
| `--labels` | 20 indoor classes | Custom comma-separated label list |
| `--mesh` | off | Generate a coloured surface mesh |
| `--mesh-method` | `tsdf` | `tsdf`: volumetric fusion of the depth maps (robust, coloured) · `poisson`: surface fit on the merged cloud |
| `--no-viewer` | off | Skip Rerun, just save `.ply` files |
| `--no-turntable` | off | Skip turntable GIF generation |

---

## Outputs

Each run saves to its own timestamped directory so results are never overwritten:

```
outputs/
└── room_20260611_143022/       # <video>_<timestamp>
    ├── frames/                 # extracted PNG frames
    ├── scene.ply               # RGB point cloud       ← open in MeshLab
    ├── scene_semantic.ply      # semantic cloud        ← if --semantic
    ├── scene_features.npz      # CLIP features         ← if --semantic; used by query.py
    ├── query_<phrase>.ply      # query heatmap         ← written by query.py
    ├── scene_mesh.ply          # surface mesh          ← if --mesh
    ├── turntable.gif           # 360° orbit render     ← auto-generated
    ├── demo.rrd                # Rerun session         ← rerun <path>/demo.rrd
    └── run_info.json           # parameters + metrics for this run
```

`run_info.json` records the full parameter set, point count, bounding box volume, average point density, and per-stage timings — useful for comparing runs after code or parameter changes.

### Viewing the outputs

| Tool | How |
|---|---|
| **Rerun** (interactive) | Opened automatically; or `rerun outputs/demo.rrd` |
| **MeshLab** (free) | `File → Open → scene.ply` |
| **CloudCompare** (free) | Drag & drop `scene.ply` |
| **Open3D** (Python) | `import open3d as o3d; o3d.visualization.draw([o3d.io.read_point_cloud("outputs/scene.ply")])` |

The **Rerun viewer** shows everything at once:
- *Top panel*: fly through the 3D point cloud and mesh (if `--mesh`); camera frustums mark where each frame was taken
- *Bottom left*: RGB frame — scrub the timeline to walk through the video
- *Bottom middle*: Depth map — colour = distance from camera
- *Bottom right*: Semantic overlay (if `--semantic` was used)

---

## Hardware notes

| Hardware | Model used | Typical time (50 frames) |
|---|---|---|
| NVIDIA RTX 3090 / 4090 | VGGT-Omega (CVPR 2026) | ~20–30 s |
| NVIDIA A100 | VGGT-Omega | ~15 s |
| MacBook M4 Pro (MPS) | VGGT-1B (CVPR 2025) | ~3–6 min |
| CPU only | VGGT-1B | ~20–40 min |

If you run out of memory, reduce `--frames` (try `30`) or lower `--image-size 384`.

---

## Design choices

### Why VGGT / VGGT-Omega?

Classical pipelines (COLMAP → MVS → mesh) are robust but slow and brittle on textureless surfaces. **VGGT** (Wang et al., CVPR 2025 Best Paper) is a 1B-parameter transformer that estimates camera poses, depth maps, and a dense 3D point cloud from a set of images **in a single feed-forward pass** — no feature matching, no bundle adjustment, no iterative optimisation.

**VGGT-Omega** (Wang et al., CVPR 2026 Oral) is its direct successor, with improved architecture and better depth quality. It is selected automatically on CUDA hardware; the original VGGT-1B is used on Apple Silicon (confirmed MPS-compatible, float32 only).

### Why SAM 2.1 + OpenCLIP for semantics?

Rather than training a dedicated semantic model, we use two powerful off-the-shelf models:

- **SAM 2.1** (Meta) segments every object in each frame automatically, with no prompts.
- **OpenCLIP** (ViT-B/32, OpenAI weights) embeds each crop and compares it against text embeddings for candidate labels via cosine similarity.

Labels are assigned per-segment in 2D, then **fused into 3D by multi-view voting**: every point is projected into every frame, a z-buffer test against that frame's depth map rejects occluded observations (so points can't inherit labels from surfaces in front of them), and the per-point label is the majority across all views that genuinely saw it. Geometry-semantic alignment comes for free — the 3D positions, the depth maps used for the occlusion test, and the camera poses all come from the same forward pass.

The approach is inspired by [Ov3R](https://arxiv.org/abs/2507.22052) (Gong et al., 2025) but independently implemented, runs on Apple Silicon, and supports an arbitrary open-vocabulary label set at runtime.

### Why open-vocabulary 3D querying?

For a robot, a reconstruction is most useful when it can be *asked things*: "where is the mug?", "something to sit on". Rather than committing to a fixed label set, the pipeline keeps the CLIP feature of every SAM2 segment plus a sparse table of which 3D points were visible in which segments (reusing the occlusion-tested visibility from label fusion). A point's semantic feature is the mean of its segments' features — and since scoring is linear (`text · mean(f) = mean(text · f)`), queries are evaluated segment-side: one small matrix product plus a sparse average. That's **~30× less memory than a dense per-point feature field, with identical results**, and any phrase can be searched in milliseconds without re-running reconstruction. Relevancy follows [LERF](https://arxiv.org/abs/2303.09553) (Kerr et al., ICCV 2023): query similarity is contrasted against canonical negatives ("object", "stuff", …) for crisp, calibrated heatmaps.

### Why TSDF fusion for the mesh?

The pipeline already produces exactly what volumetric fusion wants: per-frame metric depth maps with known camera poses. **TSDF fusion** (the KinectFusion approach, standard in robotics perception) integrates every depth map into a signed-distance voxel grid, so per-frame depth noise *averages out* instead of accumulating, and the mesh — extracted via marching cubes — gets its colours directly from the video frames. Voxel size adapts to the scene (~256 voxels across the diagonal). This is more principled than fitting a surface to the merged point cloud after the fact; Poisson reconstruction remains available via `--mesh-method poisson`.

### Why Rerun for visualisation?

[Rerun](https://rerun.io) is a spatial data visualiser designed for robotics and perception. In a single browser window it shows video frames, depth maps, camera trajectories, and point clouds all time-linked — exactly the kind of multi-modal output this pipeline produces. Sessions are saved as `.rrd` files that can be shared and reopened without re-running the pipeline.

---

## Project structure

```
humanoid-video-to-3D/
├── run.py                  # main entry point
├── query.py                # natural-language 3D scene search
├── environment.yml         # conda environment (all deps except PyTorch)
├── install.sh              # platform-aware PyTorch + model installer
├── pipeline/
│   ├── extract_frames.py   # video → PNG frames
│   ├── reconstruct.py      # VGGT / VGGT-Omega inference → point maps
│   ├── postprocess.py      # Open3D cleanup + TSDF / Poisson mesh
│   ├── semantics.py        # SAM2 + CLIP → multi-view fused 3D labels
│   └── openvocab.py        # CLIP feature bundle + query scoring
├── tests/                  # unit tests (geometry, fusion, query scoring)
├── viz/
│   ├── viewer.py           # Rerun visualization
│   └── turntable.py        # 360° GIF renderer (called automatically by run.py)
├── utils/
│   └── device.py           # hardware detection (CUDA / MPS / CPU)
├── scripts/
│   ├── download_models.py  # pre-fetch HuggingFace checkpoints
│   └── turntable.py        # standalone GIF renderer: python scripts/turntable.py <ply> <gif>
└── examples/               # sample inputs
```

---

## References

- **VGGT**: Wang et al., *Visual Geometry Grounded Transformer*, CVPR 2025 Best Paper. [Paper](https://arxiv.org/abs/2503.11651) · [Code](https://github.com/facebookresearch/vggt)
- **VGGT-Omega**: Wang et al., *VGGT Omega*, CVPR 2026 Oral. [Paper](https://arxiv.org/abs/2605.15195) · [Code](https://github.com/facebookresearch/vggt-omega)
- **SAM 2.1**: Ravi et al., *SAM 2*, Meta 2024. [Code](https://github.com/facebookresearch/sam2)
- **OpenCLIP**: Cherti et al., CVPR 2023. [Code](https://github.com/mlfoundations/open_clip)
- **Ov3R**: Gong et al., *Open-Vocabulary Semantic 3D Reconstruction from RGB Videos*, arXiv 2025. [Paper](https://arxiv.org/abs/2507.22052)
- **Open3D**: Zhou et al., arXiv 2018. [Docs](http://www.open3d.org)
- **Rerun**: [rerun.io](https://rerun.io)
