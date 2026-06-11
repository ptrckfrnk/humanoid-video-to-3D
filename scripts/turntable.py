#!/usr/bin/env python3
"""
Render a turntable GIF from a .ply point cloud or mesh.

Usage:
    python scripts/turntable.py outputs/scene.ply outputs/turntable.gif
    python scripts/turntable.py outputs/scene_mesh.ply outputs/mesh.gif --frames 72 --fps 24
"""

import argparse
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
) -> None:
    geom = o3d.io.read_triangle_mesh(str(ply_path))
    is_mesh = len(geom.triangles) > 0
    if not is_mesh:
        geom = o3d.io.read_point_cloud(str(ply_path))
        if not geom.has_points():
            print(f"Error: could not load geometry from {ply_path}", file=sys.stderr)
            sys.exit(1)

    bbox = geom.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())
    extent = np.asarray(bbox.get_extent())
    orbit_radius = float(np.linalg.norm(extent[[0, 2]])) * 0.9
    cam_y = float(center[1] + extent[1] * 0.35)   # slightly above center (scene is Y-up)

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
    print(f"  Rendering {n_frames} frames...", flush=True)

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

    _stitch_gif(frames_dir, output_path, fps, width)
    print(f"  Done → {output_path}")


def _stitch_gif(frames_dir: Path, output_path: Path, fps: int, width: int) -> None:
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

    except FileNotFoundError:
        print("ffmpeg not found — install it with: conda install -c conda-forge ffmpeg", file=sys.stderr)
        print(f"Raw frames saved in: {frames_dir}", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a turntable GIF from a .ply file.")
    p.add_argument("ply", type=Path, help="Input .ply file (point cloud or mesh)")
    p.add_argument("output", type=Path, help="Output .gif path")
    p.add_argument("--frames", "-n", type=int, default=72,
                   help="Frames for one full orbit (default: 72)")
    p.add_argument("--fps", type=int, default=24,
                   help="GIF playback speed (default: 24)")
    p.add_argument("--width", type=int, default=800,
                   help="Output width in pixels (default: 800)")
    p.add_argument("--height", type=int, default=600,
                   help="Render window height in pixels (default: 600)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.ply.exists():
        print(f"Error: {args.ply} not found", file=sys.stderr)
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_turntable(args.ply, args.output, args.frames, args.fps, args.width, args.height)
