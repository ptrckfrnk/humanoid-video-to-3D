"""
Open-vocabulary semantic labeling via SAM 2.1 + OpenCLIP.

For each video frame:
  1. SAM2AutomaticMaskGenerator produces instance masks (no prompts needed)
  2. Each mask crop is encoded by OpenCLIP image encoder
  3. Pre-encoded text embeddings for candidate labels are compared via cosine similarity
  4. Best-matching label assigned to every pixel in that mask
  5. Labels propagated into 3D by projecting world points back into each frame

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
    from sam2.build_sam import build_sam2

    labels = custom_labels or DEFAULT_LABELS

    # ── Load SAM 2.1 ─────────────────────────────────────────────────────────
    console.print("  Loading SAM 2.1 Hiera-Small...")
    sam_device = device.type if device.type in ("cuda", "mps") else "cpu"
    ckpt_path  = _download_sam2()
    sam_model  = build_sam2("sam2.1_hiera_small.yaml", ckpt_path, device=sam_device)
    mask_gen   = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=16,          # coarser grid → faster, still good coverage
        pred_iou_thresh=0.85,
        stability_score_thresh=0.90,
    )

    # ── Load OpenCLIP ─────────────────────────────────────────────────────────
    console.print("  Loading OpenCLIP ViT-B/32...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    tokenizer  = open_clip.get_tokenizer("ViT-B-32")

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

    # ── Propagate 2D labels → 3D points ──────────────────────────────────────
    H, W    = result.images.shape[1:3]
    pts_all = np.asarray(scene.point_cloud.points)   # (N, 3)
    N       = len(pts_all)
    point_labels = [-1] * N    # -1 until a frame assigns a label

    for s in range(S):
        uvs, depths = _project_points(
            pts_all, result.extrinsics[s], result.intrinsics[s]
        )
        in_frame = (
            (depths > 0.05) &
            (uvs[:, 0] >= 0) & (uvs[:, 0] < W) &
            (uvs[:, 1] >= 0) & (uvs[:, 1] < H)
        )
        if not in_frame.any():
            continue

        us = uvs[in_frame, 0].astype(int).clip(0, W - 1)
        vs = uvs[in_frame, 1].astype(int).clip(0, H - 1)
        frame_label_idxs = label_maps[s][vs, us]

        idxs = np.where(in_frame)[0]
        for pt_i, lbl_idx in zip(idxs, frame_label_idxs):
            if lbl_idx >= 0 and point_labels[pt_i] == -1:
                point_labels[pt_i] = lbl_idx

    # Map index → label string; -1 → "other"
    str_labels = [labels[i] if i >= 0 else "other" for i in point_labels]

    # Build color-coded point cloud
    sem_colors = np.array([
        LABEL_COLORS.get(l, _label_color(l)) / 255.0
        for l in str_labels
    ], dtype=np.float64)

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

    for mask_info in masks:
        seg  = mask_info["segmentation"]                    # (H, W) bool
        x, y, bw, bh = [int(v) for v in mask_info["bbox"]]
        if bw < 10 or bh < 10:
            continue

        crop       = frame_rgb[y : y + bh, x : x + bw]
        pil_crop   = PILImage.fromarray(crop)
        img_tensor = clip_preprocess(pil_crop).unsqueeze(0).to(device)

        with torch.inference_mode():
            img_feat = clip_model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)    # (1, D)
            sim      = (img_feat @ text_features.T).softmax(dim=-1)      # (1, L)
            lbl_idx  = int(sim.argmax())

        label_map[seg] = lbl_idx

    return label_map


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


def _download_sam2() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="facebook/sam2.1-hiera-small",
        filename="sam2.1_hiera_small.pt",
    )
