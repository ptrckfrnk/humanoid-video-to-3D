# CLAUDE.md — Project Context

*Loaded automatically by Claude Code at the start of every session.*
*Sprint status and backlog → [PLAN.md](PLAN.md) · Public documentation → [README.md](README.md)*

---

## What this project is

A job application for the **Perception & Spatial AI internship at Humanoid (London)** — a robotics company Patrick really wants to work at. This is not a homework exercise. Every decision should be made with submission quality in mind.

---

## The challenge (verbatim)

> **Intern Challenge: From Video to 3D Reconstruction**
>
> Build a system that takes a short video (e.g. captured on a phone), of a small indoor area such as a small room, and reconstructs a 3D scene.
>
> The core goal is geometric reconstruction from video. Semantic understanding is welcome, but optional.
>
> At a minimum, your system should:
> - Generate a 3D representation of the scene from video input
> - Produce a reconstruction that is geometrically coherent and consistent
>
> Optional extensions:
> - Assign semantic labels in 3D (e.g. tables, chairs)
> - Ensure any semantic predictions are aligned with the underlying geometry
>
> There are no constraints on real-time performance. We're intentionally leaving the approach open — use any tools, models, frameworks, or agentic workflows you find effective.
>
> **What to submit:** working codebase · run instructions · example inputs/outputs · (optional) design note
>
> **What we care about:**
> 1. Simplicity and usability
> 2. **Creativity in approach**
> 3. Quality of 3D reconstruction
> 4. Clear, compelling presentation of results
> 5. Coherence between geometry and semantics
>
> *"We're not looking for standard solutions, we're looking for how you think. The strongest submissions are creative, original, and push beyond the obvious."*

---

## Why the current approach is creative

A vanilla COLMAP pipeline would score low. Our approach is deliberately cutting-edge:

- **VGGT-Omega** (CVPR 2026 Oral, ~3 weeks old at submission): single forward pass — no feature matching, no bundle adjustment, no iterative optimisation
- **SAM2.1 + OpenCLIP semantic lifting**: open-vocabulary 3D labels with no training required
- End-to-end: `python run.py video.mp4` → coloured point cloud + mesh + semantic labels + Rerun viewer

Full design rationale in README § Design choices.

---

## Hardware and model compatibility

| Hardware | Model | Frames | Notes |
|---|---|---|---|
| NVIDIA A100 / 4090 | VGGT-Omega (CVPR 2026) | 80–100 | Final quality run target |
| MacBook M4 Pro (MPS) | VGGT-1B (CVPR 2025) | ≤20 | O(n²) global attention limits frame count |
| CPU only | VGGT-1B | ≤20 | Slow but functional |

**MPS-specific constraints:**
- VGGT-1B: ✅ float32 only, ~20 frame limit
- SAM2: ✅ MPS
- OpenCLIP: ✅ MPS
- Open3D (Poisson mesh): ✅ depth=6 · ❌ depth≥7 segfaults (ARM build bug — see B6 in PLAN.md)
- VGGT-Omega: ❌ CUDA only — requires cloud GPU

---

## Known open issues

- **B6**: Poisson segfaults at depth≥7 on Apple Silicon — must fix before GPU run. Try `open3d==0.17.0` or fall back to `ball_pivoting`.
- **No real example outputs yet** — the GPU run (C1–C3 in PLAN.md) is the next critical step.
- **`room.mov`** is a quick smoke-test video only; not suitable for README examples.

---

## Decision checklist — before suggesting any change

1. Does this make the submission *more impressive* to a robotics company hiring for Perception & Spatial AI?
2. Does it improve quality, usability, or the README presentation?
3. Is it worth the time cost relative to shooting real videos + running on GPU (highest ROI right now)?
