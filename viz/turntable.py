"""
Turntable GIF renderer — orbits a camera around a .ply scene and stitches
the frames into a palette-optimised GIF via ffmpeg.

Rendering strategy:
  - Linux / headless: OffscreenRenderer (EGL, no display needed)
  - macOS:            Visualizer with capture_screen_image (window appears briefly)
"""

from __future__ import annotations
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

# Open3D ViewControl rotates by this many radians per "pixel" unit
_RAD_PER_PIXEL = 0.003


def render_turntable(
    ply_path: Path,
    output_path: Path,
    n_frames: int = 72,
    fps: int = 24,
    width: int = 800,
    height: int = 600,
) -> bool:
    """Render a turntable GIF. Returns True on success, False if ffmpeg missing."""
    # Probe file type silently — Open3D warns when a PLY has no triangles
    prev_level = o3d.utility.get_verbosity_level()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    geom = o3d.io.read_triangle_mesh(str(ply_path))
    o3d.utility.set_verbosity_level(prev_level)

    is_mesh = len(geom.triangles) > 0
    if not is_mesh:
        geom = o3d.io.read_point_cloud(str(ply_path))
        if not geom.has_points():
            print(f"  [turntable] could not load {ply_path}", file=sys.stderr)
            return False

    frames_dir = Path(tempfile.mkdtemp())

    # Try headless first (Linux/cloud GPU); fall back to windowed (macOS)
    try:
        _render_offscreen(geom, is_mesh, frames_dir, n_frames, width, height)
    except Exception as e:
        if any(k in str(e) for k in ("EGL", "Headless", "headless")):
            _render_windowed(geom, is_mesh, frames_dir, n_frames, width, height)
        else:
            raise

    return _stitch_gif(frames_dir, output_path, fps, width)


# ── Offscreen renderer (Linux / headless) ────────────────────────────────────

def _render_offscreen(
    geom, is_mesh: bool, frames_dir: Path, n_frames: int, width: int, height: int
) -> None:
    bbox   = geom.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())
    extent = np.asarray(bbox.get_extent())
    radius = float(np.linalg.norm(extent[[0, 2]])) * 0.9
    cam_y  = float(center[1] + extent[1] * 0.35)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.array([0.12, 0.12, 0.12, 1.0]))

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit" if is_mesh else "defaultUnlit"
    if not is_mesh:
        mat.point_size = 3.0
    renderer.scene.add_geometry("scene", geom, mat)
    renderer.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 75000)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.enable_indirect_light(True)

    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        eye = [
            float(center[0]) + radius * math.sin(angle),
            cam_y,
            float(center[2]) + radius * math.cos(angle),
        ]
        renderer.setup_camera(60.0, center.tolist(), eye, [0.0, 1.0, 0.0])
        o3d.io.write_image(
            str(frames_dir / f"frame_{i:04d}.png"),
            renderer.render_to_image(),
        )


# ── Windowed renderer (macOS) ─────────────────────────────────────────────────

def _render_windowed(
    geom, is_mesh: bool, frames_dir: Path, n_frames: int, width: int, height: int
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=width, height=height)
    vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.12, 0.12, 0.12])
    if is_mesh:
        opt.mesh_show_back_face = True
    else:
        opt.point_size = 2.0

    vis.reset_view_point(True)
    ctr = vis.get_view_control()

    # Let the initial frame settle before capturing
    for _ in range(10):
        vis.poll_events()
        vis.update_renderer()

    # Tilt slightly downward for a better angle on room-scale scenes
    ctr.rotate(0, -150)

    # Exact 360° orbit: 2π rad / _RAD_PER_PIXEL total units, split across frames
    step = (2 * math.pi / _RAD_PER_PIXEL) / n_frames

    for i in range(n_frames):
        ctr.rotate(step, 0)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(
            str(frames_dir / f"frame_{i:04d}.png"), do_render=True
        )

    vis.destroy_window()


# ── GIF stitching ─────────────────────────────────────────────────────────────

def _stitch_gif(frames_dir: Path, output_path: Path, fps: int, width: int) -> bool:
    pattern = str(frames_dir / "frame_%04d.png")
    palette = str(frames_dir / "palette.png")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
            "-vf", f"scale={width}:-1:flags=lanczos,palettegen",
            palette,
        ], check=True, capture_output=True)
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", pattern, "-i", palette,
            "-lavfi", f"scale={width}:-1:flags=lanczos [x]; [x][1:v] paletteuse",
            str(output_path),
        ], check=True, capture_output=True)
        return True
    except FileNotFoundError:
        return False
