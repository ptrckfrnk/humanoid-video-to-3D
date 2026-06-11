# PLAN — Humanoid Internship Challenge: Video to 3D

*Sprint tracker and research log.*
*Challenge spec, judging criteria, and approach rationale → [CLAUDE.md](CLAUDE.md)*
*Public documentation, design choices, usage → [README.md](README.md)*

---

## Research Landscape (June 2026)

Captured at decision time. The chosen approach is VGGT-1B/Omega + SAM2+CLIP — see README § Design choices for rationale.

### Feed-Forward Reconstruction Methods

| Method | Venue | GitHub | MPS? | Notes |
|---|---|---|---|---|
| **VGGT** | CVPR 2025 Best Paper | facebookresearch/vggt | ✅ (float32) | N images → cameras + depth + point cloud in one pass |
| **VGGT-Omega** | CVPR 2026 Oral | facebookresearch/vggt-omega | ❌ CUDA only | Improved VGGT, released May 2026; ~3 weeks old at submission |
| FastVGGT | ICLR 2026 | mystorm16/FastVGGT | Unknown | 4× faster VGGT, training-free pruning |
| InfiniteVGGT | 2026 | AutoLab-SAI-SJTU/InfiniteVGGT | Unknown | Infinite streaming with rolling memory |
| StreamVGGT | ICLR 2026 | wzzheng/StreamVGGT | Unknown | Streaming 4D, causal temporal attention |
| SLAM3R | CVPR 2025 Highlight | PKU-VCL-3DV/SLAM3R | ❌ (CUDA) | Real-time 20+ FPS dense video reconstruction |
| Fast3R | CVPR 2025 | facebookresearch/fast3r | Unknown | 1000+ images in one forward pass |
| MonST3R | ICLR 2025 | junyi42/monst3r | Unknown | DUSt3R for dynamic/moving scenes |

### Semantic 3D Methods

| Method | Venue | Key idea | MPS? |
|---|---|---|---|
| Ov3R | arXiv Jul 2025 | CLIP into SLAM3R backbone; 2D→3D open-vocab semantic lifting | ❌ (SLAM3R base) |
| OVSeg3R | arXiv Sep 2025 | 2D→3D open-vocab instance seg via reconstruction | Unknown |
| **SAM2 + CLIP lifting** | — | Per-frame SAM2 masks + CLIP text labels → lifted to 3D via depth/pose | ✅ MPS |

### 3D Gaussian Splatting on Apple Silicon

| Tool | MPS? | Notes |
|---|---|---|
| gsplat (official) | ❌ CUDA-only | Main library, not usable |
| OpenSplat | ✅ Apple Metal | Production-grade, CPU/GPU |
| msplat | ✅ Metal | ~70s for full scene on M4 Max |

---

## Status

- [x] Research phase complete
- [x] Pipeline approved and implemented
- [x] Environment setup (environment.yml + install.sh)
- [x] Stage 1: Frame extraction — farthest-point sampling on thumbnails (pipeline/extract_frames.py)
- [x] Stage 2: VGGT / VGGT-Omega inference (pipeline/reconstruct.py)
- [x] Stage 3: Open3D post-processing + ground-plane alignment + optional mesh (pipeline/postprocess.py)
- [x] Stage 4: SAM2 + OpenCLIP semantic lifting (pipeline/semantics.py)
- [x] Stage 5: Rerun viewer + named semantic legend (viz/viewer.py)
- [x] Turntable GIF auto-generated after every run (viz/turntable.py)
- [x] Timestamped output dirs — runs never overwrite each other
- [x] run_info.json: parameters + metrics per run
- [x] Mesh logged in Rerun viewer
- [x] Mesh pipeline smoke-tested on MPS (depth=6 works; depth≥7 segfaults — Open3D ARM bug)
- [x] Stage 3 progress output: per-step timing + point counts
- [x] Open3D verbosity silenced (eliminated blank-line spam)
- [x] Public repo pushed to GitHub (ptrckfrnk/humanoid-video-to-3D)
- [ ] Real example videos shot and processed
- [ ] High-fidelity outputs via cloud GPU (VGGT-Omega, 80 frames)
- [ ] Turntable GIFs + example outputs embedded in README
- [ ] README examples section: 3 scenes, side-by-side layout

---

## Improvement Roadmap

**P1** = must do before submission · **P2** = high leverage · **P3** = if time allows

---

### A. Video Capture  *(no code — highest ROI)*

| # | Priority | Task | Notes |
|---|---|---|---|
| A1 | **P1** | Shoot 3 example scenes | Desk/lab bench, bookshelf corner, robot arm or mechanical object |
| A2 | **P1** | Correct camera motion | Slow orbital arc ~180° over 6–8 s — not a linear walk |
| A3 | **P1** | Lock focus + exposure | Before recording; diffuse even lighting, no reflections or direct sun |
| A4 | **P1** | Avoid problematic surfaces | No mirrors, glass, plain white walls, or fast movement |
| A5 | P2 | Replace current example video | room.mov may not showcase the pipeline well |

**Capture checklist per scene:**
- Lock focus + exposure before starting
- Move slowly: 6–8 s for a 180° arc; keep subject centred
- No tilt up/down mid-shot
- Good scenes: cluttered desk, bookshelf, lab equipment, furnished room corner

---

### B. Pipeline Improvements  *(code changes)*

| # | Priority | Task | Notes |
|---|---|---|---|
| B1 | **P1** | Tune `--conf` per scene (try 1.0–2.5) | Default 1.5 may be too tight or too loose depending on scene |
| ~~B2~~ | ~~P2~~ | ~~Smarter frame sampling~~ | ✅ Done: farthest-point sampling on 64×64 thumbnails |
| ~~B3~~ | ~~P2~~ | ~~Better mesh post-processing~~ | ✅ Done: downsampling fixed, depth=6 smoke-tested on MPS |
| B4 | P2 | Texture projection onto mesh | Photorealistic mesh colours from video frames → UV map |
| B5 | P3 | Sliding window + ICP for long videos on MPS | 60–100 frame coverage without CUDA; complex but high value |
| ~~B6~~ | ~~P1~~ | ~~Fix Poisson segfault at depth≥7 on Apple Silicon~~ | ✅ Done: TSDF fusion is the default mesh path (sidesteps Poisson entirely); `--mesh-method poisson` kept as fallback |

---

### C. Compute / Cloud GPU  *(~$3–5 of GPU time)*

| # | Priority | Task | Notes |
|---|---|---|---|
| C1 | **P1** | Set up cloud GPU (RunPod or Lambda Labs, A100 or 4090) | One-time setup; ~30 min |
| C2 | **P1** | Run 3 example videos: `--model vggt-omega --frames 80 --device cuda --mesh` | VGGT-Omega + 80 frames = much denser, higher-quality output |
| C3 | **P1** | Download outputs (.ply, .rrd) for README | These become the submitted example outputs |
| C4 | P2 | Run `--semantic` on at least one scene on GPU | Semantics is faster on CUDA |

---

### D. Presentation  *(what judges see first)*

| # | Priority | Task | Notes |
|---|---|---|---|
| ~~D1~~ | ~~P1~~ | ~~Turntable GIF~~ | ✅ Done: auto-generated after every run (`--no-turntable` to skip) |
| D2 | **P1** | README examples section | 3 scenes × 1 row: input frame / point cloud / semantic overlay |
| D3 | **P1** | Check in a .rrd file (or GitHub Release) | `rerun examples/office.rrd` lets judges explore interactively |
| D4 | P2 | GIF of Rerun viewer being navigated | Shows live interactive experience |
| D5 | P2 | Metrics callout in README | Point count, inference time, model name, frame count |
| D6 | P3 | Short screen-record of end-to-end run | Terminal → Rerun viewer; strong for portfolio |
