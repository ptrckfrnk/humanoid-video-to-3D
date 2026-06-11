"""
Turntable GIF renderer — orbits a camera around a .ply scene and stitches
the frames into a palette-optimised GIF via ffmpeg.
"""

from __future__ import annotations
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d


def render_turntable(
    ply_path: Path,
    output_path: Path,
    n_frames: int = 72,
    fps: int = 24,
    width: int = 800,
    height: int = 600,
) -> bool:
    """Render a turntable GIF. Returns True on success, False if ffmpeg missing."""
    geom = o3d.io.read_triangle_mesh(str(ply_path))
    is_mesh = len(geom.triangles) > 0
    if not is_mesh:
        geom = o3d.io.read_point_cloud(str(ply_path))
        if not geom.has_points():
            print(f"  [turntable] could not load {ply_path}", file=sys.stderr)
            return False

    bbox   = geom.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())
    extent = np.asarray(bbox.get_extent())
    orbit_radius = float(np.linalg.norm(extent[[0, 2]])) * 0.9
    cam_y = float(center[1] + extent[1] * 0.35)

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

    frames_dir = Path(tempfile.mkdtemp())

    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        eye = [
            float(center[0]) + orbit_radius * math.sin(angle),
            cam_y,
            float(center[2]) + orbit_radius * math.cos(angle),
        ]
        renderer.setup_camera(60.0, center.tolist(), eye, [0.0, 1.0, 0.0])
        img = renderer.render_to_image()
        o3d.io.write_image(str(frames_dir / f"frame_{i:04d}.png"), img)

    return _stitch_gif(frames_dir, output_path, fps, width)


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
