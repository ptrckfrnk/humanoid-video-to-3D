"""
Rerun 0.33-compatible interactive visualization.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │               3D Scene (main panel)                 │
  │   · Dense RGB point cloud                           │
  │   · Camera frustums (one per frame)                 │
  │   · Camera images pinned in 3D space                │
  ├──────────────────┬──────────────┬───────────────────┤
  │   RGB Frame      │  Depth Map   │  Semantic Labels  │
  │   (timeline)     │  (timeline)  │  (static)         │
  └──────────────────┴──────────────┴───────────────────┘
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.reconstruct import ReconstructionResult
from pipeline.postprocess import SceneResult


def launch_viewer(
    result:     ReconstructionResult,
    scene:      SceneResult,
    semantic:   "Optional[SemanticResult]",
    output_dir: Path,
) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    rrd_path = output_dir / "demo.rrd"

    rr.init("video-to-3d", spawn=True)

    # ── Blueprint ─────────────────────────────────────────────────────────────
    bottom_panels = [
        rrb.Spatial2DView(name="RGB Frame", origin="/camera/rgb"),
        rrb.Spatial2DView(name="Depth Map", origin="/camera/depth"),
    ]
    if semantic is not None:
        bottom_panels.append(
            rrb.Spatial2DView(name="Semantic", origin="/camera/semantic")
        )

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(name="3D Scene", origin="/world"),
            rrb.Horizontal(*bottom_panels),
            row_shares=[3, 1],
        ),
        rrb.SelectionPanel(expanded=False),
        rrb.TimePanel(expanded=True),
    )
    rr.send_blueprint(blueprint)

    S    = result.images.shape[0]
    H, W = result.images.shape[1:3]

    # Invert extrinsics: cam-from-world → world-from-cam (4x4)
    c2w = _invert_se3_batch(result.extrinsics)   # (S, 4, 4)

    # ── Per-frame data ────────────────────────────────────────────────────────
    print(f"  Logging {S} frames to Rerun...")
    for s in range(S):
        rr.set_time("frame", sequence=s)   # rerun 0.33 API

        # 2D panels
        rr.log("camera/rgb",   rr.Image(result.images[s]))
        rr.log("camera/depth", rr.DepthImage(result.depth[s], meter=1.0))

        # Camera pose in 3D world — mat3x3 + translation
        R = c2w[s, :3, :3]
        t = c2w[s, :3, 3]
        rr.log(
            f"world/cameras/cam_{s:04d}",
            rr.Transform3D(translation=t, mat3x3=R),
        )
        K = result.intrinsics[s]
        rr.log(
            f"world/cameras/cam_{s:04d}",
            rr.Pinhole(image_from_camera=K, width=W, height=H),
        )
        # Pin the RGB image to its frustum so it appears in the 3D view
        rr.log(f"world/cameras/cam_{s:04d}/image", rr.Image(result.images[s]))

    # ── RGB point cloud (not time-indexed — always visible) ───────────────────
    rr.set_time("frame", sequence=S - 1)

    pts  = np.asarray(scene.point_cloud.points, dtype=np.float32)
    cols = (np.asarray(scene.point_cloud.colors) * 255).astype(np.uint8)
    rr.log("world/scene/geometry", rr.Points3D(positions=pts, colors=cols, radii=0.003))

    # ── Semantic cloud ────────────────────────────────────────────────────────
    if semantic is not None:
        sem_pts  = np.asarray(semantic.colored_cloud.points, dtype=np.float32)
        sem_cols = (np.asarray(semantic.colored_cloud.colors) * 255).astype(np.uint8)
        rr.log(
            "world/scene/semantic",
            rr.Points3D(positions=sem_pts, colors=sem_cols, radii=0.003),
        )
        _log_semantic_overlay(result, semantic, S - 1)

    # Save session to file
    rr.save(str(rrd_path))

    print(f"\n  Rerun session saved → {rrd_path}")
    print(f"  Reopen later with:  rerun {rrd_path}")
    print("  Press Ctrl+C in this terminal to exit.\n")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _invert_se3_batch(extrinsics: np.ndarray) -> np.ndarray:
    """Invert (S, 3, 4) cam-from-world matrices → (S, 4, 4) world-from-cam."""
    S   = extrinsics.shape[0]
    c2w = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))
    R   = extrinsics[:, :3, :3]
    t   = extrinsics[:, :3, 3:]
    c2w[:, :3, :3] = R.transpose(0, 2, 1)
    c2w[:, :3, 3:] = -(R.transpose(0, 2, 1) @ t)
    return c2w


def _log_semantic_overlay(
    result:    ReconstructionResult,
    semantic:  "SemanticResult",
    frame_idx: int,
) -> None:
    """Log a colour-coded 2D semantic overlay for one frame."""
    import rerun as rr
    from pipeline.semantics import _project_points

    H, W     = result.images.shape[1:3]
    overlay  = result.images[frame_idx].copy()
    pts_all  = np.asarray(semantic.colored_cloud.points)
    cols_all = (np.asarray(semantic.colored_cloud.colors) * 255).astype(np.uint8)

    uvs, depths = _project_points(
        pts_all, result.extrinsics[frame_idx], result.intrinsics[frame_idx]
    )
    valid = (
        (depths > 0.05) &
        (uvs[:, 0] >= 0) & (uvs[:, 0] < W) &
        (uvs[:, 1] >= 0) & (uvs[:, 1] < H)
    )
    if valid.any():
        us = uvs[valid, 0].astype(int).clip(0, W - 1)
        vs = uvs[valid, 1].astype(int).clip(0, H - 1)
        overlay[vs, us] = cols_all[valid]

    rr.set_time("frame", sequence=frame_idx)
    rr.log("camera/semantic", rr.Image(overlay))
