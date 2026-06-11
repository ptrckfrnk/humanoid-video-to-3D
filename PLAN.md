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
MacBook Pro M4 Pro — Apple Silicon MPS only, no CUDA on local machine.
Cloud GPU (RunPod / Lambda Labs A100) available for high-fidelity runs.

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
- VGGT-1B: MPS ✅ (float32, ~20 frame limit due to O(n²) attention)
- VGGT-Omega: CUDA only ❌ MPS — requires cloud GPU
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
- [x] Stage 3: Open3D post-processing + ground-plane alignment (pipeline/postprocess.py)
- [x] Stage 4: Semantic lifting SAM2 + CLIP (pipeline/semantics.py)
- [x] Stage 5: Rerun visualization with named semantic legend (viz/viewer.py)
- [x] README with design choices
- [x] Public repo pushed to GitHub (ptrckfrnk/humanoid-video-to-3D)
- [ ] Real example videos shot and run through pipeline
- [ ] High-fidelity outputs via cloud GPU (VGGT-Omega, 80 frames)
- [ ] Turntable GIF of each scene → embedded in README
- [ ] README examples section: 3 scenes, side-by-side input/output layout

---

## Improvement Roadmap

Priority matrix for the pre-submission polish sprint.
**P1** = must do, directly affects deliverable quality.
**P2** = high leverage, do after P1.
**P3** = if time allows, adds impressiveness.

---

### A. Video Capture  *(no code, free, highest ROI)*

| # | Priority | Task | Notes |
|---|---|---|---|
| A1 | **P1** | Shoot 3 example scenes using correct technique | See technique guide below |
| A2 | **P1** | Scene selection: desk/lab bench, bookshelf corner, robot arm or mechanical object | Dense texture + interesting 3D structure |
| A3 | **P1** | Camera motion: slow orbital arc (~180° over 6–8 sec), not linear walk | Baseline between views is critical for VGGT depth accuracy |
| A4 | **P1** | Lock focus + exposure before recording; diffuse even lighting, no direct sun/reflections | Motion blur and overexposure break multi-view consistency |
| A5 | P2 | Replace the existing example video with the best new one | Current example may not showcase the pipeline well |

**Capture technique checklist (for each video):**
- Lock focus and exposure before starting
- Move slowly: 6–8 seconds for a 180° arc
- Keep the subject centered; don't tilt up/down mid-shot
- Avoid: mirrors, glass surfaces, plain white walls, fast movement
- Good scenes: desk with objects, shelves, lab equipment, corner of a furnished room

---

### B. Pipeline / Architecture Improvements  *(code changes)*

| # | Priority | Task | Notes |
|---|---|---|---|
| B1 | **P1** | Tune `--conf` threshold per scene (try 1.0–2.5) | Default 1.5 may be too aggressive or too loose depending on scene |
| B2 | P2 | Smarter frame sampling: skip near-duplicate frames using SSIM or optical flow | Uniform sampling wastes budget on near-identical frames |
| ~~B3~~ | ~~P2~~ | ~~Better mesh post-processing: increase Poisson depth 9→11, add density trimming~~ | ✅ Done: depth→10, density trimming added |
| B4 | P2 | Texture projection onto mesh (Open3D `create_from_point_cloud_poisson` + UV map) | Makes mesh outputs look photorealistic, not just geometry |
| B5 | P3 | Sliding window + ICP alignment for long videos on MPS | Enables 60–100 frame coverage without CUDA; complex but high technical value |

---

### C. Compute / Cloud GPU  *(rent ~$3–5 of GPU time)*

| # | Priority | Task | Notes |
|---|---|---|---|
| C1 | **P1** | Set up cloud GPU environment (RunPod or Lambda Labs, A100 or 4090) | One-time setup; ~30 min |
| C2 | **P1** | Run each of the 3 example videos with `--model vggt-omega --frames 80 --device cuda --mesh` | VGGT-Omega + 80 frames = significantly denser, higher-quality output |
| C3 | **P1** | Download outputs (.ply, .rrd) locally | These become the example outputs in the README |
| C4 | P2 | Also run `--semantic` on at least one scene on the GPU | Semantics is faster on CUDA; test on all 3 if budget allows |

---

### D. Presentation  *(what judges actually see first)*

| # | Priority | Task | Notes |
|---|---|---|---|
| ~~D1~~ | ~~**P1**~~ | ~~Turntable GIF for each of the 3 scenes → embed in README~~ | ✅ Done: auto-generated after every run.py call (viz/turntable.py, --no-turntable to skip) |
| D2 | **P1** | README examples section: 3 scenes, one image per row (input frame / point cloud / semantic overlay) | Judges want to see what it looks like before they run anything |
| D3 | **P1** | Check in the .rrd file for at least one scene (or link to GitHub Release asset) | Anyone with `rerun examples/office.rrd` can explore interactively — impressive |
| D4 | P2 | Add a GIF of the Rerun viewer being navigated (screen recording) | Shows the live interactive experience, not just a static export |
| D5 | P2 | Add metrics callout in README: point count, inference time, model name, # frames | Concrete numbers signal rigour |
| D6 | P3 | Short Loom / screen-record of end-to-end run (terminal → Rerun viewer) | Strong for portfolio; optional for submission |

---

### Turntable GIF — how to use (D1 ✅)

Auto-generated after every `python run.py video.mp4` call → `outputs/turntable.gif`.
Skip with `--no-turntable`. Run standalone: `python scripts/turntable.py outputs/scene.ply outputs/turntable.gif`
