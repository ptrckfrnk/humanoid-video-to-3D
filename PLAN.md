# PLAN — Humanoid Internship Challenge: Video to 3D

## Goal
Build a public GitHub repo that takes a short indoor phone video and reconstructs a 3D scene.
Apply for Perception & Spatial AI internship at Humanoid (London).

## Must-Have (from job description)
- [ ] Accept video input (phone-captured, short indoor scene)
- [ ] Generate a 3D representation of the scene
- [ ] Geometrically coherent and consistent reconstruction
- [ ] Clear README with run instructions
- [ ] Example input + output included

## Nice-to-Have
- [ ] Semantic labels in 3D (e.g. tables, chairs) — text-queryable
- [ ] Geometry-semantic alignment

## Hardware Constraint
MacBook Pro M4 Pro — Apple Silicon MPS only, no CUDA.

## Key Criteria (what judges care about)
1. Simplicity and usability
2. **Creativity** — "not a standard solution"
3. Quality of reconstruction
4. Compelling presentation of results
5. Geometry-semantic coherence

---

## Research Findings (June 2026)

### Feed-Forward Reconstruction Methods

| Method | Venue | GitHub | MPS? | Notes |
|---|---|---|---|---|
| VGGT | CVPR 2025 Best Paper | facebookresearch/vggt | ✅ (float32) | N images → cameras + depth + point cloud in one pass |
| VGGT-Omega | CVPR 2026 Oral | facebookresearch/vggt-omega | ✅ (likely) | Improved VGGT, released May 2026 |
| FastVGGT | ICLR 2026 | mystorm16/FastVGGT | Unknown | 4× faster VGGT, training-free pruning |
| InfiniteVGGT | 2026 | AutoLab-SAI-SJTU/InfiniteVGGT | Unknown | Infinite streaming with rolling memory |
| StreamVGGT | ICLR 2026 | wzzheng/StreamVGGT | Unknown | Streaming 4D, causal temporal attention |
| SLAM3R | CVPR 2025 Highlight | PKU-VCL-3DV/SLAM3R | ❌ (CUDA) | Real-time 20+ FPS dense video reconstruction |
| Fast3R | CVPR 2025 | facebookresearch/fast3r | Unknown | 1000+ images in one forward pass |
| MonST3R | ICLR 2025 | junyi42/monst3r | Unknown | DUSt3R for dynamic/moving scenes |

### Semantic 3D Methods

| Method | Venue | Key idea | MPS? |
|---|---|---|---|
| Ov3R | arXiv Jul 2025 | CLIP3R: CLIP into SLAM3R backbone; 2D-3D OVS semantic lifting | Likely ❌ (SLAM3R base) |
| OVSeg3R | arXiv Sep 2025 | 2D→3D open-vocab instance seg via reconstruction | Unknown |
| SAM2 + CLIP lifting | — | Per-frame SAM2 masks + CLIP text labels → lifted to 3D via depth/pose | ✅ MPS |

### 3D Gaussian Splatting on Apple Silicon
| Tool | MPS? | Notes |
|---|---|---|
| gsplat (official) | ❌ CUDA-only | Main library, not usable |
| OpenSplat | ✅ Apple Metal | Production-grade, CPU/GPU |
| msplat | ✅ Metal | ~70s for full scene on M4 Max |
| gsplat-mps | ⚠️ Old fork | v0.1.3, limited |

---

## Chosen Pipeline (DECISION PENDING USER APPROVAL)

### Proposed: VGGT-Omega + SAM2/CLIP Semantic Lifting

**Stage 1 — Frame extraction**
- FFmpeg: extract N evenly-spaced frames from video (e.g. 50-100 frames)

**Stage 2 — Feed-forward 3D reconstruction**
- **VGGT-Omega** (CVPR 2026 Oral): N images → dense point maps + camera poses + depth
- Runs on MPS (float32); no SfM/COLMAP needed
- One model, one forward pass, seconds of inference

**Stage 3 — Point cloud post-processing**
- Open3D: merge point maps, remove outliers, optional Poisson mesh reconstruction

**Stage 4 — Semantic lifting (optional but impressive)**
- **SAM2**: per-frame instance segmentation masks (MPS)
- **OpenCLIP**: embed mask crops → text-queryable labels (MPS)
- Project labels onto 3D point cloud via depth + camera poses from Stage 2

**Stage 5 — Export & visualization**
- PLY point cloud (universal, opens in MeshLab, CloudCompare)
- Open3D interactive viewer
- Optional: WebGL/Three.js viewer in notebook for demo

### Why this is creative (not standard)
- VGGT-Omega is 3 weeks old as of June 2026 — no published pipeline wraps it like this
- SAM2 + CLIP semantic lifting is the closest thing to Ov3R's approach but independently
  implemented on a different backbone, without CUDA
- End-to-end from phone video to labelled 3D in a single `python run.py video.mp4`

### Apple Silicon compatibility
- VGGT-Omega: MPS ✅ (float32, ~5GB model)
- Open3D: CPU ✅ (pure Python/C++)
- SAM2: MPS ✅
- OpenCLIP: MPS ✅
- FFmpeg: CPU ✅

---

## Repo Structure

```
humanoid-video-to-3D/
├── README.md              # instructions, example outputs, design note
├── PLAN.md                # this file
├── pyproject.toml
├── run.py                 # main entry: python run.py input.mp4 [--semantic]
├── pipeline/
│   ├── __init__.py
│   ├── extract_frames.py  # FFmpeg frame extraction
│   ├── reconstruct.py     # VGGT-Omega inference → point cloud
│   ├── postprocess.py     # Open3D cleanup + mesh
│   └── semantics.py       # SAM2 + CLIP → 3D labels (optional)
├── viz/
│   └── viewer.py          # Open3D / notebook viewer
├── examples/
│   ├── input.mp4
│   └── output/
│       ├── scene.ply
│       ├── scene_semantic.ply
│       └── renders/
└── notebooks/
    └── demo.ipynb         # walkthrough with visualizations
```

---

## Status
- [x] Research phase complete
- [x] User approved pipeline
- [x] Environment setup (environment.yml + install.sh)
- [x] Stage 1: Frame extraction (pipeline/extract_frames.py)
- [x] Stage 2: VGGT / VGGT-Omega integration (pipeline/reconstruct.py)
- [x] Stage 3: Open3D post-processing (pipeline/postprocess.py)
- [x] Stage 4: Semantic lifting SAM2 + CLIP (pipeline/semantics.py)
- [x] Stage 5: Rerun visualization (viz/viewer.py)
- [x] README with design choices
- [ ] **NEXT: Test on actual video — record a room and run the pipeline**
- [ ] Add real example outputs to README (screenshots / GIF)
- [ ] Git init + push to GitHub
