# CLAUDE.md — Project Context

This file is read automatically by Claude Code at the start of every session.

---

## What this project is

A job application for a **Perception & Spatial AI internship at Humanoid (London)** — a robotics company Patrick really wants to work at. This is not a homework exercise. Every decision should be made with submission quality in mind.

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
> **What to submit:**
> - A working codebase
> - Clear instructions on how to run your system
> - Example input(s) and output(s)
> - (Optional) A short note explaining your design choices and tradeoffs
>
> **What we care about:**
> - Simplicity and usability of your solution
> - **Creativity in approach**
> - Quality of 3D scene reconstruction
> - Clear, compelling presentation of results
> - Coherence between geometry and semantics
>
> *Make something you're proud of.*

---

## What the judges are looking for

- **"We're not looking for standard solutions, we're looking for how you think."**
- **"The strongest submissions are creative, original, and push beyond the obvious."**

This means: a vanilla COLMAP pipeline would score low. The current approach (VGGT-Omega, a CVPR 2026 Oral paper, with SAM2+CLIP semantic lifting) is deliberately cutting-edge and non-standard.

---

## Current approach (why it's creative)

- **VGGT-1B / VGGT-Omega** (CVPR 2025 Best Paper / CVPR 2026 Oral): single forward pass, no COLMAP, no iterative optimisation
- **SAM2.1 + OpenCLIP semantic lifting**: open-vocabulary 3D labels without any training
- **End-to-end**: `python run.py video.mp4` → coloured point cloud + optional mesh + semantic labels + Rerun viewer
- VGGT-Omega is ~3 weeks old as of submission — no other published pipeline wraps it like this

---

## Hardware

- **Development**: MacBook Pro M4 Pro (MPS only, no CUDA)
- **Final quality run**: Cloud GPU (RunPod / Lambda Labs A100 or 4090) — not yet done

---

## Key constraints and known issues

- MPS: max ~20 frames due to O(n²) global attention in VGGT
- Poisson mesh: depth=6 works on MPS; depth≥7 segfaults (Open3D ARM bug, see B6 in PLAN.md)
- GPU run needed for: VGGT-Omega model, 80 frames, depth=9 mesh, final example outputs

---

## Before suggesting any change, ask:

1. Does this make the submission *more impressive* to a robotics company hiring for Perception & Spatial AI?
2. Does it improve quality, usability, or the README presentation?
3. Is it worth the time cost vs. shooting real videos and running on GPU (which has the highest ROI)?
