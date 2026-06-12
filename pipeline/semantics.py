"""
Open-vocabulary semantic labeling via SAM 2.1 + OpenCLIP.

For each video frame:
  1. SAM2AutomaticMaskGenerator produces instance masks (no prompts needed)
  2. Each mask crop — background pixels greyed out so CLIP sees the object,
     not its surroundings — is encoded by OpenCLIP in batches
  3. Pre-encoded text embeddings for candidate labels are compared via cosine similarity
  4. Best-matching label assigned to every pixel in that mask

2D → 3D lifting (multi-view fusion, see _fuse_labels):
  5. Every 3D point is projected into every frame; a frame only gets a say if
     the point passes a z-buffer occlusion test against that frame's depth map
     (otherwise occluded points would inherit the label of whatever surface
     is in front of them)
  6. Each visible, labeled observation casts one vote; the per-point label is
     the majority across all views — a single bad mask in one frame can no
     longer poison a point

Gracefully degrades — if SAM2 or OpenCLIP are missing it raises ImportError
and run.py catches it.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open3d as o3d
import torch
from rich.console import Console
from rich.progress import track

from pipeline.reconstruct import ReconstructionResult
from pipeline.postprocess import SceneResult

console = Console()

DEFAULT_LABELS = [
    "chair", "table", "sofa", "desk", "bed",
    "floor", "wall", "ceiling", "door", "window",
    "lamp", "monitor", "keyboard", "shelf", "cabinet",
    "plant", "pillow", "curtain", "person", "other",
]

# Deterministic per-label RGB colors
def _label_color(label: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(label)) % (2**32))
    return rng.integers(60, 230, size=3, dtype=np.uint8)

LABEL_COLORS = {l: _label_color(l) for l in DEFAULT_LABELS}


@dataclass
class SemanticResult:
    labels:        List[str]                        # per-point label string
    colored_cloud: o3d.geometry.PointCloud          # cloud coloured by class
    label_set:     List[str]                        # all candidate labels used


# ── Public entry point ────────────────────────────────────────────────────────

def label_scene(
    result: ReconstructionResult,
    scene:  SceneResult,
    custom_labels: Optional[List[str]],
    device: torch.device,
) -> SemanticResult:
    import open_clip
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    labels = custom_labels or DEFAULT_LABELS

    # ── Load SAM 2.1 ─────────────────────────────────────────────────────────
    console.print("  Loading SAM 2.1 Hiera-Small...")
    sam_device = device.type if device.type in ("cuda", "mps") else "cpu"
    sam_model  = build_sam2_hf("facebook/sam2.1-hiera-small", device=sam_device)
    mask_gen   = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=16,          # coarser grid → faster, still good coverage
        pred_iou_thresh=0.85,
        stability_score_thresh=0.90,
    )

    # ── Load OpenCLIP ─────────────────────────────────────────────────────────
    console.print("  Loading OpenCLIP ViT-B/32...")
    # "-quickgelu" matches the activation the OpenAI weights were trained with;
    # the plain ViT-B-32 config uses standard GELU and degrades the embeddings.
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    tokenizer  = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    with torch.inference_mode():
        text_tokens   = tokenizer([f"a photo of a {l}" for l in labels]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)   # (L, D)

    # ── Per-frame segmentation + labeling ────────────────────────────────────
    S = result.images.shape[0]
    label_maps: list[np.ndarray] = []   # (H, W) int per frame; -1 = unlabeled

    for s in track(range(S), description="  Labeling frames"):
        lmap = _label_frame(
            result.images[s], mask_gen,
            clip_model, clip_preprocess, text_features,
            labels, device,
        )
        label_maps.append(lmap)

    # ── Propagate 2D labels → 3D points (multi-view fusion) ──────────────────
    pts_all = np.asarray(scene.point_cloud.points)   # (N, 3)

    best, n_votes = _fuse_labels(
        pts_all, label_maps,
        result.extrinsics, result.intrinsics, result.depth,
        n_labels=len(labels),
    )
    point_labels = np.where(n_votes > 0, best, -1)   # (N,) int; -1 = never observed

    n_labeled = int((point_labels >= 0).sum())
    console.print(
        f"  Multi-view fusion: {n_labeled:,}/{len(pts_all):,} points labeled "
        f"({100 * n_labeled / max(len(pts_all), 1):.0f}%), "
        f"avg {n_votes[n_votes > 0].mean():.1f} views/point"
        if n_labeled else
        "  [yellow]Multi-view fusion: no points received a label[/yellow]"
    )

    # Map index → label string; -1 → "other"
    str_labels = [labels[i] if i >= 0 else "other" for i in point_labels]

    # Build color-coded point cloud (LUT row len(labels) = "other" fallback)
    color_lut = np.array(
        [LABEL_COLORS.get(l, _label_color(l)) for l in labels]
        + [LABEL_COLORS.get("other", _label_color("other"))],
        dtype=np.float64,
    ) / 255.0
    sem_colors = color_lut[np.where(point_labels >= 0, point_labels, len(labels))]

    sem_cloud = o3d.geometry.PointCloud()
    sem_cloud.points = o3d.utility.Vector3dVector(pts_all)
    sem_cloud.colors = o3d.utility.Vector3dVector(sem_colors)

    return SemanticResult(
        labels=str_labels,
        colored_cloud=sem_cloud,
        label_set=labels,
    )


# ── Per-frame helpers ─────────────────────────────────────────────────────────

def _label_frame(
    frame_rgb: np.ndarray,         # (H, W, 3) uint8
    mask_gen,
    clip_model,
    clip_preprocess,
    text_features: torch.Tensor,   # (L, D)
    labels: List[str],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return (H, W) int32 array: index into labels, or -1 for unlabeled pixels."""
    from PIL import Image as PILImage

    H, W = frame_rgb.shape[:2]
    label_map = np.full((H, W), -1, dtype=np.int32)

    masks = mask_gen.generate(frame_rgb)
    if not masks:
        return label_map

    # Largest masks first so small objects can override large background later
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    # ── Prepare all crops, then encode in batches ─────────────────────────────
    crops: list[torch.Tensor] = []
    segs:  list[np.ndarray]   = []
    for mask_info in masks:
        seg  = mask_info["segmentation"]                    # (H, W) bool
        x, y, bw, bh = [int(v) for v in mask_info["bbox"]]
        if bw < 10 or bh < 10:
            continue

        # Grey out background pixels inside the bbox so CLIP classifies the
        # segmented object rather than whatever surrounds it.
        crop = frame_rgb[y : y + bh, x : x + bw].copy()
        crop[~seg[y : y + bh, x : x + bw]] = 127

        crops.append(clip_preprocess(PILImage.fromarray(crop)))
        segs.append(seg)

    if not crops:
        return label_map

    lbl_idxs: list[int] = []
    with torch.inference_mode():
        for i in range(0, len(crops), batch_size):
            batch = torch.stack(crops[i : i + batch_size]).to(device)
            feats = clip_model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)        # (B, D)
            lbl_idxs.extend((feats @ text_features.T).argmax(dim=-1).cpu().tolist())

    # Paint in sorted order: large masks first, smaller ones override on top
    for seg, lbl_idx in zip(segs, lbl_idxs):
        label_map[seg] = lbl_idx

    return label_map


def _fuse_labels(
    pts_world:     np.ndarray,        # (N, 3)
    label_maps:    List[np.ndarray],  # S × (H, W) int32; -1 = unlabeled
    extrinsics:    np.ndarray,        # (S, 3, 4) cam-from-world
    intrinsics:    np.ndarray,        # (S, 3, 3)
    depth_maps:    np.ndarray,        # (S, H, W) metric depth
    n_labels:      int,
    occlusion_tol: float = 0.05,      # relative depth tolerance for visibility
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse per-frame 2D label maps into per-point 3D labels by majority vote.

    A frame votes for a point only if the point projects inside the image AND
    its camera-space depth matches the frame's depth map at that pixel within
    `occlusion_tol` (relative) — i.e. the point is actually visible in that
    frame, not hidden behind a nearer surface.

    Returns:
        best:    (N,) int64 — winning label index per point (argmax of votes;
                 meaningless where n_votes == 0)
        n_votes: (N,) int64 — number of frames that cast a vote for the point
    """
    N = len(pts_world)
    S, H, W = depth_maps.shape

    votes = np.zeros((N, n_labels), dtype=np.uint16)

    for s in range(S):
        uvs, z = _project_points(pts_world, extrinsics[s], intrinsics[s])
        us = np.round(uvs[:, 0]).astype(np.int64)
        vs = np.round(uvs[:, 1]).astype(np.int64)

        in_frame = (z > 0.05) & (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        if not in_frame.any():
            continue
        idx = np.flatnonzero(in_frame)
        us, vs, zs = us[idx], vs[idx], z[idx]

        # Z-buffer occlusion test against this frame's depth map
        d_map   = depth_maps[s][vs, us]
        visible = (d_map > 0) & (np.abs(zs - d_map) <= occlusion_tol * d_map)
        if not visible.any():
            continue
        idx, us, vs = idx[visible], us[visible], vs[visible]

        lbl     = label_maps[s][vs, us]
        labeled = lbl >= 0
        np.add.at(votes, (idx[labeled], lbl[labeled]), 1)

    return votes.argmax(axis=1), votes.sum(axis=1, dtype=np.int64)


def _project_points(
    pts_world:  np.ndarray,   # (N, 3)
    extrinsic:  np.ndarray,   # (3, 4) cam-from-world
    intrinsic:  np.ndarray,   # (3, 3)
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D world points to 2D pixel coordinates. Returns (N, 2) and depths (N,)."""
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    cam_pts = (R @ pts_world.T).T + t        # (N, 3) camera-space
    depths  = cam_pts[:, 2]

    uv = (intrinsic @ cam_pts.T).T           # (N, 3) homogeneous image coords
    uv = uv / (uv[:, 2:3] + 1e-8)           # normalise

    return uv[:, :2], depths


