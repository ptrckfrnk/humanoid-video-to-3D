#!/usr/bin/env python3
"""
Video → 3D Scene Reconstruction
================================
Internship Challenge — Humanoid (London)

Examples
--------
Basic geometry:
    python run.py examples/room.mp4

With semantic labels:
    python run.py examples/room.mp4 --semantic

Custom labels, more frames, mesh output:
    python run.py examples/room.mp4 --frames 80 --semantic \\
        --labels "chair,table,sofa,door,window" --mesh

Force CUDA on a Linux workstation:
    python run.py examples/room.mp4 --device cuda --model vggt-omega

Skip the interactive viewer (just save files):
    python run.py examples/room.mp4 --no-viewer

Each run saves to its own timestamped directory (outputs/<video>_<timestamp>/)
so results are never overwritten. Compare runs with scripts/compare_runs.py.
"""

import argparse
import datetime
import json
import sys
import time
import warnings
from pathlib import Path

# VGGT uses the deprecated torch.cuda.amp.autocast API — nothing we can fix upstream
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

import numpy as np
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct a 3D scene from a phone video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", type=Path,
                   help="Path to input video (MP4, MOV, AVI, …)")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output directory (default: outputs/<video>_<timestamp>)")
    p.add_argument("--frames", "-n", type=int, default=50,
                   help="Number of frames to sample (50 is safe for M4 Pro; "
                        "use 80-100 on CUDA for denser clouds)")
    p.add_argument("--model", choices=["auto", "vggt", "vggt-omega"],
                   default="auto",
                   help="'auto': VGGT-Omega on CUDA, VGGT-1B otherwise")
    p.add_argument("--device", type=str, default=None,
                   help="Force hardware backend (cuda / mps / cpu)")
    p.add_argument("--conf", type=float, default=1.5,
                   help="Confidence threshold for point filtering "
                        "(higher → fewer but cleaner points)")
    p.add_argument("--semantic", action="store_true",
                   help="Run open-vocabulary semantic labeling (SAM2 + CLIP)")
    p.add_argument("--labels", type=str, default=None,
                   help="Custom label set, comma-separated. "
                        "Default: 20 common indoor categories")
    p.add_argument("--mesh", action="store_true",
                   help="Generate a coloured surface mesh (TSDF fusion of the "
                        "depth maps by default)")
    p.add_argument("--mesh-method", choices=["tsdf", "poisson"], default="tsdf",
                   help="tsdf: volumetric fusion of depth maps (robust, coloured); "
                        "poisson: surface fit on the merged point cloud")
    p.add_argument("--no-viewer", action="store_true",
                   help="Skip Rerun viewer (just save .ply files)")
    p.add_argument("--no-turntable", action="store_true",
                   help="Skip turntable GIF generation")
    p.add_argument("--image-size", type=int, default=518,
                   help="Resolution passed to VGGT preprocessing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_run_start = time.time()

    from utils.device import get_device, get_dtype
    from pipeline.extract_frames import extract_frames
    from pipeline.reconstruct import reconstruct
    from pipeline.postprocess import postprocess

    device = get_device(args.device)
    dtype  = get_dtype(device)

    # MPS (Apple Silicon) can't fit 50 frames in the global attention — cap at 20
    if device.type == "mps" and args.frames == 50:
        args.frames = 20
        console.print("[dim]Note: defaulting to 20 frames on MPS to fit in memory. "
                      "Use --frames N to override.[/dim]")

    if not args.video.exists():
        console.print(f"[red]Error:[/red] Video not found: {args.video}")
        sys.exit(1)

    # Each run gets its own directory — never overwrites previous results
    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path("outputs") / f"{args.video.stem}_{ts}"

    console.rule("[bold cyan]Video → 3D Scene Reconstruction[/bold cyan]")
    console.print(f"  Video   : [green]{args.video}[/green]")
    console.print(f"  Output  : [green]{args.output}[/green]")
    console.print(f"  Device  : [yellow]{device}[/yellow]  (dtype={dtype})")
    console.print(f"  Frames  : {args.frames}")
    console.print(f"  Model   : {args.model}")
    console.print(f"  Semantic: {'yes' if args.semantic else 'no'}")
    console.print()

    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(exist_ok=True)

    timings: dict[str, float] = {}

    # ── Stage 1: Extract frames ───────────────────────────────────────────────
    console.print("[bold]Stage 1 / 4[/bold]  Extracting frames...")
    t0 = time.time()
    frame_paths = extract_frames(args.video, args.frames, frames_dir)
    timings["extract_frames"] = round(time.time() - t0, 2)
    console.print(f"           → {len(frame_paths)} frames  ({timings['extract_frames']:.1f}s)")

    # ── Stage 2: 3D Reconstruction ────────────────────────────────────────────
    console.print("[bold]Stage 2 / 4[/bold]  Running 3D reconstruction...")
    t0 = time.time()
    try:
        result = reconstruct(
            frame_paths,
            device=device,
            dtype=dtype,
            model_name=args.model,
            image_size=args.image_size,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "invalid buffer size" in str(e).lower():
            console.print(f"[red]Out of memory.[/red]  Current: --frames {args.frames}")
            console.print("  Try reducing: --frames 15")
            sys.exit(1)
        raise
    timings["reconstruct"] = round(time.time() - t0, 2)
    console.print(f"           → done  ({timings['reconstruct']:.1f}s)")

    if args.mesh and len(frame_paths) < 15:
        console.print(
            f"[yellow]Note:[/yellow] --mesh works best with 15+ frames "
            f"(current: {len(frame_paths)}). Surface may be incomplete."
        )

    # ── Stage 3: Post-process ─────────────────────────────────────────────────
    console.print("[bold]Stage 3 / 4[/bold]  Building point cloud...")
    t0 = time.time()
    scene = postprocess(
        result,
        conf_threshold=args.conf,
        build_mesh=args.mesh,
        mesh_method=args.mesh_method,
        output_dir=args.output,
        console=console,
    )
    timings["postprocess"] = round(time.time() - t0, 2)
    n_pts = len(scene.point_cloud.points)
    console.print(f"           → {n_pts:,} points  ({timings['postprocess']:.1f}s)")

    # ── Stage 4: Semantics ────────────────────────────────────────────────────
    semantic = None
    if args.semantic:
        console.print("[bold]Stage 4 / 4[/bold]  Semantic labeling (SAM2 + CLIP)...")
        t0 = time.time()
        try:
            from pipeline.semantics import label_scene
            label_list = (
                [l.strip() for l in args.labels.split(",")]
                if args.labels else None
            )
            semantic = label_scene(result, scene, label_list, device)
            unique_labels = sorted(set(semantic.labels))
            timings["semantics"] = round(time.time() - t0, 2)
            console.print(f"           → {len(unique_labels)} classes  ({timings['semantics']:.1f}s)")
            from pipeline.semantics import LABEL_COLORS, _label_color
            console.print("  [bold]Class legend:[/bold]")
            for lbl in unique_labels:
                c = LABEL_COLORS.get(lbl, _label_color(lbl))
                console.print(f"    [rgb({c[0]},{c[1]},{c[2]})]■[/rgb({c[0]},{c[1]},{c[2]})] {lbl}")
        except ImportError as e:
            console.print(f"  [yellow]⚠  Semantics skipped:[/yellow] {e}")
            console.print("     Install SAM2 + OpenCLIP (see README) to enable.")
    else:
        console.print("[bold]Stage 4 / 4[/bold]  Semantic labeling "
                      "[dim]— skipped (add --semantic to enable)[/dim]")

    # ── Save outputs ──────────────────────────────────────────────────────────
    import open3d as o3d
    console.print()
    console.print("[bold]Saving outputs…[/bold]")

    ply_path = args.output / "scene.ply"
    o3d.io.write_point_cloud(str(ply_path), scene.point_cloud)
    console.print(f"  [green]✓[/green] Point cloud     → {ply_path}")

    if semantic is not None:
        sem_path = args.output / "scene_semantic.ply"
        o3d.io.write_point_cloud(str(sem_path), semantic.colored_cloud)
        console.print(f"  [green]✓[/green] Semantic cloud  → {sem_path}")

    if scene.mesh is not None:
        mesh_path = args.output / "scene_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), scene.mesh)
        console.print(f"  [green]✓[/green] Surface mesh    → {mesh_path}")

    # ── Save run metadata ─────────────────────────────────────────────────────
    timings["total"] = round(time.time() - t_run_start, 2)
    _save_run_info(args, device, len(frame_paths), scene, semantic, timings)
    console.print(f"  [green]✓[/green] Run metadata    → {args.output / 'run_info.json'}")

    # ── Turntable GIF ─────────────────────────────────────────────────────────
    gif_path = args.output / "turntable.gif"
    if not args.no_turntable:
        console.print()
        console.print("[bold]Rendering turntable GIF…[/bold]")
        console.print("  [dim](skip with --no-turntable)[/dim]")
        try:
            from viz.turntable import render_turntable
            source_ply = (args.output / "scene_semantic.ply") if semantic is not None \
                         else ply_path
            ok = render_turntable(source_ply, gif_path)
            if ok:
                console.print(f"  [green]✓[/green] Turntable GIF   → {gif_path}")
            else:
                console.print(
                    "  [yellow]⚠  ffmpeg not found[/yellow] — "
                    "install with: conda install -c conda-forge ffmpeg"
                )
        except Exception as e:
            console.print(f"  [yellow]⚠  Turntable skipped:[/yellow] {e}")

    # ── Visualize ─────────────────────────────────────────────────────────────
    if not args.no_viewer:
        console.print()
        console.print("[bold]Launching Rerun viewer…[/bold]")
        console.print("  [dim]Browser should open automatically. "
                      "Press Ctrl+C to stop.[/dim]")
        try:
            from viz.viewer import launch_viewer
            launch_viewer(result, scene, semantic, args.output)
        except ImportError:
            console.print(
                "  [yellow]⚠  rerun-sdk not installed.[/yellow]  "
                "Open the .ply files in MeshLab or CloudCompare."
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Done[/bold green]")
    console.print(f"  All outputs in [green]{args.output}/[/green]")
    console.print(f"  View in MeshLab: [dim]open {ply_path}[/dim]")
    if gif_path.exists():
        console.print(f"  Turntable GIF:  [dim]open {gif_path}[/dim]")
    if (args.output / "demo.rrd").exists():
        console.print(f"  Reopen viewer:  [dim]rerun {args.output}/demo.rrd[/dim]")
    console.print()


# ── Run metadata ──────────────────────────────────────────────────────────────

def _save_run_info(
    args,
    device,
    n_frames_extracted: int,
    scene,
    semantic,
    timings: dict,
) -> None:
    import open3d as o3d

    pts   = np.asarray(scene.point_cloud.points)
    bbox  = scene.point_cloud.get_axis_aligned_bounding_box()
    vol   = float(np.prod(np.asarray(bbox.get_extent())))

    # Average nearest-neighbour distance — proxy for point density.
    # Sample 2000 points so it stays fast on large clouds.
    sample_n   = min(len(pts), 2000)
    sample_idx = np.random.choice(len(pts), sample_n, replace=False)
    sample_pcd = o3d.geometry.PointCloud()
    sample_pcd.points = o3d.utility.Vector3dVector(pts[sample_idx])
    avg_nn = float(np.mean(sample_pcd.compute_nearest_neighbor_distance()))

    info = {
        "video":     str(args.video),
        "run_dir":   str(args.output),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "model":              args.model,
            "device":             str(device),
            "n_frames_requested": args.frames,
            "n_frames_extracted": n_frames_extracted,
            "conf_threshold":     args.conf,
            "image_size":         args.image_size,
            "semantic":           args.semantic,
            "mesh":               args.mesh,
            "mesh_method":        args.mesh_method if args.mesh else None,
        },
        "results": {
            "n_points":           len(pts),
            "bbox_volume_m3":     round(vol, 4),
            "avg_nn_dist_m":      round(avg_nn, 6),
            "n_semantic_classes": len(set(semantic.labels)) if semantic else None,
        },
        "timings_s": timings,
    }

    with open(args.output / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
