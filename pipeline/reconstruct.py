"""
3D reconstruction via VGGT (CVPR 2025 Best Paper) or VGGT-Omega (CVPR 2026 Oral).

Hardware routing (automatic unless overridden with --model):
  CUDA  → VGGT-Omega  (bfloat16/float16, best quality)
  MPS   → VGGT-1B     (float32, Apple Silicon)
  CPU   → VGGT-1B     (float32, slow but functional)

Both backends produce the same ReconstructionResult dataclass so the rest
of the pipeline is hardware-agnostic.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

from utils.device import autocast_ctx

console = Console()


@dataclass
class ReconstructionResult:
    """Unified output from any reconstruction backend."""
    images:             np.ndarray   # (S, H, W, 3)  uint8  RGB
    world_points:       np.ndarray   # (S, H, W, 3)  float32 world-space XYZ
    world_points_conf:  np.ndarray   # (S, H, W)     float32 confidence
    depth:              np.ndarray   # (S, H, W)     float32 metric depth
    depth_conf:         np.ndarray   # (S, H, W)     float32
    extrinsics:         np.ndarray   # (S, 3, 4)     cam-from-world
    intrinsics:         np.ndarray   # (S, 3, 3)     camera intrinsics
    frame_paths:        list[Path]


# ── Public entry point ────────────────────────────────────────────────────────

def reconstruct(
    frame_paths: list[Path],
    device: torch.device,
    dtype: torch.dtype,
    model_name: str = "auto",
    image_size: int = 518,
) -> ReconstructionResult:
    model_name = _resolve_model(model_name, device)
    console.print(f"  Model : [cyan]{model_name}[/cyan]  device=[yellow]{device}[/yellow]  dtype={dtype}")

    if model_name == "vggt-omega":
        return _run_vggt_omega(frame_paths, device, dtype)
    return _run_vggt(frame_paths, device, dtype, image_size)


# ── Model selection ───────────────────────────────────────────────────────────

def _resolve_model(model_name: str, device: torch.device) -> str:
    if model_name == "auto":
        return "vggt-omega" if device.type == "cuda" else "vggt"
    if model_name == "vggt-omega" and device.type != "cuda":
        console.print("  [yellow]⚠  VGGT-Omega requires CUDA → falling back to VGGT-1B[/yellow]")
        return "vggt"
    return model_name


# ── VGGT-1B backend (MPS / CPU / CUDA) ───────────────────────────────────────

def _run_vggt(
    frame_paths: list[Path],
    device: torch.device,
    dtype: torch.dtype,
    image_size: int = 518,
) -> ReconstructionResult:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    console.print("  Loading [bold]VGGT-1B[/bold] from HuggingFace (~5 GB first run)...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).eval()

    image_names = [str(p) for p in frame_paths]
    images = load_and_preprocess_images(image_names).to(device)   # (S, 3, H, W)

    with torch.inference_mode(), autocast_ctx(device, dtype):
        predictions = model(images)

    # Decode pose encoding → extrinsics + intrinsics
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        (images.shape[-2], images.shape[-1]),
    )                                              # (1, S, 3, 4) and (1, S, 3, 3)

    extrinsics = extrinsics.squeeze(0).cpu().float().numpy()        # (S, 3, 4)
    intrinsics = intrinsics.squeeze(0).cpu().float().numpy()        # (S, 3, 3)

    world_points      = predictions["world_points"].squeeze(0).cpu().float().numpy()       # (S, H, W, 3)
    world_points_conf = predictions["world_points_conf"].squeeze(0).cpu().float().numpy()  # (S, H, W)
    depth             = predictions["depth"].squeeze(0).squeeze(-1).cpu().float().numpy()  # (S, H, W)
    depth_conf        = predictions["depth_conf"].squeeze(0).cpu().float().numpy()         # (S, H, W)

    # images tensor (1, S, 3, H, W) → (S, H, W, 3) uint8
    imgs_raw = predictions["images"].squeeze(0)     # (S, 3, H, W) float in [0, 1]
    imgs_np  = (imgs_raw.cpu().float().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)

    return ReconstructionResult(
        images=imgs_np,
        world_points=world_points,
        world_points_conf=world_points_conf,
        depth=depth,
        depth_conf=depth_conf,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame_paths=frame_paths,
    )


# ── VGGT-Omega backend (CUDA only) ────────────────────────────────────────────

def _run_vggt_omega(
    frame_paths: list[Path],
    device: torch.device,
    dtype: torch.dtype,
) -> ReconstructionResult:
    try:
        from vggt_omega.models.vggt_omega import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ImportError:
        console.print("  [yellow]VGGT-Omega package not found → falling back to VGGT-1B[/yellow]")
        return _run_vggt(frame_paths, device, dtype)

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    console.print("  Loading [bold]VGGT-Omega[/bold] (CVPR 2026) from HuggingFace...")
    ckpt_path = hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.safetensors")
    model = VGGTOmega()
    model.load_state_dict(load_file(ckpt_path), strict=False)
    model = model.to(device).eval()

    image_names = [str(p) for p in frame_paths]
    images = load_and_preprocess_images(image_names, image_resolution=512).to(device)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype):
            predictions = model(images)

    image_size_hw = (images.shape[-2], images.shape[-1])
    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"].squeeze(0),
        image_size_hw,
        build_intrinsics=True,
    )
    extrinsics = extrinsics.cpu().float().numpy()   # (S, 3, 4)
    intrinsics = intrinsics.cpu().float().numpy()   # (S, 3, 3)

    depth      = predictions["depth"].squeeze(0).squeeze(-1).cpu().float().numpy()   # (S, H, W)
    depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()           # (S, H, W)

    # VGGT-Omega has no world_points head → backproject from depth
    world_points = _backproject(depth, extrinsics, intrinsics)

    imgs_raw = predictions["images"].squeeze(0)
    imgs_np  = (imgs_raw.cpu().float().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)

    return ReconstructionResult(
        images=imgs_np,
        world_points=world_points,
        world_points_conf=depth_conf,
        depth=depth,
        depth_conf=depth_conf,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame_paths=frame_paths,
    )


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _backproject(
    depth: np.ndarray,       # (S, H, W)
    extrinsics: np.ndarray,  # (S, 3, 4) cam-from-world
    intrinsics: np.ndarray,  # (S, 3, 3)
) -> np.ndarray:
    """Back-project depth maps to metric 3D world-space point maps (S, H, W, 3)."""
    S, H, W = depth.shape
    world_points = np.zeros((S, H, W, 3), dtype=np.float32)

    ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    ones   = np.ones_like(xs)
    pixels = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3)   # (H*W, 3)

    for s in range(S):
        K_inv = np.linalg.inv(intrinsics[s])          # (3, 3)
        d     = depth[s].reshape(-1)                  # (H*W,)
        R     = extrinsics[s, :3, :3]                 # cam-from-world rotation
        t     = extrinsics[s, :3, 3]                  # cam-from-world translation

        cam_pts   = (K_inv @ pixels.T).T * d[:, None]       # (H*W, 3) in camera space
        world_pts = (R.T @ (cam_pts - t).T).T               # (H*W, 3) in world space
        world_points[s] = world_pts.reshape(H, W, 3)

    return world_points
