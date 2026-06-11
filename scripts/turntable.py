#!/usr/bin/env python3
"""
Render a turntable GIF from a .ply point cloud or mesh.

Usage:
    python scripts/turntable.py outputs/scene.ply outputs/turntable.gif
    python scripts/turntable.py outputs/scene_mesh.ply outputs/mesh.gif --frames 72 --fps 24
"""

import argparse
import sys
from pathlib import Path

from viz.turntable import render_turntable


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
    ok = render_turntable(args.ply, args.output, args.frames, args.fps, args.width, args.height)
    if not ok:
        print("ffmpeg not found — install with: conda install -c conda-forge ffmpeg", file=sys.stderr)
        sys.exit(1)
    print(f"Done → {args.output}")
