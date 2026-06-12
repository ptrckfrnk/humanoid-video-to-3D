#!/usr/bin/env python3
"""
Open-vocabulary 3D scene query
===============================
Search a reconstructed scene with natural language — any phrase, not just the
label set used at reconstruction time.

Examples
--------
    python query.py outputs/room_20260612_010453 "office chair"
    python query.py outputs/room_20260612_010453 "something to sit on" "coffee mug"
    python query.py outputs/room_20260612_010453 "keyboard" --no-viewer

Requires a run made with --semantic (it saves scene_features.npz: per-segment
CLIP features + the point→segment observation table). Each query produces a
relevancy heatmap over the point cloud — saved as query_<phrase>.ply and shown
in the Rerun viewer alongside the RGB cloud.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from rich.console import Console

from pipeline.openvocab import (
    CANONICAL_NEGATIVES,
    aggregate_point_scores,
    load_feature_bundle,
    relevancy_scores,
    score_colormap,
)

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query a reconstructed 3D scene with natural language.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dir", type=Path,
                   help="Output directory of a run made with --semantic")
    p.add_argument("queries", nargs="+",
                   help="One or more natural-language queries")
    p.add_argument("--temperature", type=float, default=10.0,
                   help="Relevancy softmax temperature (higher → sharper)")
    p.add_argument("--no-viewer", action="store_true",
                   help="Skip the Rerun viewer, just save .ply heatmaps")
    return p.parse_args()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main() -> None:
    args = parse_args()

    feat_path = args.run_dir / "scene_features.npz"
    if not feat_path.exists():
        console.print(f"[red]Error:[/red] {feat_path} not found.")
        console.print("Re-run the pipeline with --semantic to enable querying.")
        sys.exit(1)

    bundle = load_feature_bundle(feat_path)
    points     = bundle["points"]
    rgb        = bundle["colors"]
    mask_feats = bundle["mask_feats"]
    n_points   = len(points)

    console.rule("[bold cyan]Open-Vocabulary 3D Query[/bold cyan]")
    console.print(f"  Scene    : [green]{args.run_dir}[/green]")
    console.print(f"  Points   : {n_points:,}   Segments: {len(mask_feats):,}")
    console.print(f"  Queries  : {', '.join(repr(q) for q in args.queries)}")
    console.print()

    # ── Encode query + canonical negative prompts ─────────────────────────────
    import open_clip
    import torch
    from utils.device import get_device

    device = get_device(None)
    model_name = str(bundle["clip_model"])
    console.print(f"  Loading CLIP text encoder ({model_name})...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=str(bundle["clip_pretrained"])
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    prompts = list(args.queries) + CANONICAL_NEGATIVES
    with torch.inference_mode():
        tokens = tokenizer(prompts).to(device)
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.cpu().float().numpy()
    query_feats, negative_feats = text_feats[: len(args.queries)], text_feats[len(args.queries):]

    # ── Score each query ──────────────────────────────────────────────────────
    import open3d as o3d

    heatmaps: list[tuple[str, np.ndarray, np.ndarray]] = []   # (query, scores, colors)
    for query, q_feat in zip(args.queries, query_feats):
        seg_rel = relevancy_scores(mask_feats, q_feat, negative_feats,
                                   temperature=args.temperature)
        scores = aggregate_point_scores(seg_rel, bundle["obs_point"],
                                        bundle["obs_mask"], n_points)

        seen = np.isfinite(scores)
        n_hot = int((scores[seen] > 0.5).sum())
        console.print(
            f"  [bold]{query!r}[/bold]: peak relevancy "
            f"{np.nanmax(scores):.2f}, {n_hot:,} points above 0.5 "
            f"({100 * n_hot / max(seen.sum(), 1):.1f}% of observed)"
        )

        colors = score_colormap(scores)
        heatmaps.append((query, scores, colors))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        ply_path = args.run_dir / f"query_{_slug(query)}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        console.print(f"    [green]✓[/green] Heatmap → {ply_path}")

    # ── Rerun viewer: RGB cloud + one toggleable heatmap per query ───────────
    if not args.no_viewer:
        import rerun as rr
        import rerun.blueprint as rrb

        rrd_path = args.run_dir / "query.rrd"
        rr.init("video-to-3d-query")
        rr.save(str(rrd_path))
        rr.send_blueprint(rrb.Blueprint(
            rrb.Spatial3DView(name="Query Heatmaps", origin="/world"),
            rrb.SelectionPanel(expanded=False),
            rrb.TimePanel(expanded=False),
        ))

        rr.log("world/scene/rgb", rr.Points3D(positions=points, colors=rgb, radii=0.003))
        for query, _, colors in heatmaps:
            rr.log(f"world/query/{_slug(query)}",
                   rr.Points3D(positions=points, colors=colors, radii=0.003))

        console.print()
        console.print("  Opening Rerun viewer — toggle queries in the left panel…")
        subprocess.Popen(
            [sys.executable, "-m", "rerun", str(rrd_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        console.print(f"  Reopen any time with: [dim]rerun {rrd_path}[/dim]")

    console.print()


if __name__ == "__main__":
    main()
