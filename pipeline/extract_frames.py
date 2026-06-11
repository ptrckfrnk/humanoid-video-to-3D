"""
Extract frames from a video, selecting for maximum visual diversity.

Strategy:
  Pass 1 — Read a dense candidate pool (6× the requested count) as 64×64
            grayscale thumbnails. Fast, no disk I/O.
  Select  — Farthest-point sampling (FPS) in thumbnail L2 space. Each new
            frame is chosen to be as different as possible from every frame
            already selected. Near-duplicates (camera barely moved) are
            automatically skipped; frames with real camera baseline are kept.
  Pass 2 — Write only the selected frames to disk at full resolution.

The function signature is identical to the original so run.py needs no changes.
"""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

_THUMB_PX    = 64   # thumbnail side length for diversity scoring
_POOL_FACTOR = 6    # candidate pool size = n_frames × _POOL_FACTOR


def extract_frames(
    video_path: Path,
    n_frames: int,
    output_dir: Path,
) -> list[Path]:
    """
    Sample n_frames from the video, maximising visual diversity via
    farthest-point sampling. Returns saved paths in temporal order.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video reports 0 frames: {video_path}")

    n_frames = min(n_frames, total)

    # ── Pass 1: build candidate pool as tiny thumbnails ───────────────────────
    pool_size       = min(total, n_frames * _POOL_FACTOR)
    candidate_idxs  = np.linspace(0, total - 1, pool_size, dtype=int)
    pool_vid_idxs: list[int]       = []
    thumbs:        list[np.ndarray] = []

    for idx in candidate_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (_THUMB_PX, _THUMB_PX)).ravel().astype(np.float32) / 255.0
        pool_vid_idxs.append(int(idx))
        thumbs.append(thumb)

    cap.release()

    if not thumbs:
        raise RuntimeError("No frames could be extracted from the video.")

    # ── Select diverse frames ─────────────────────────────────────────────────
    thumbnails = np.stack(thumbs)                              # (P, THUMB_PX²)
    chosen     = _fps_select(thumbnails, n_frames)             # pool positions
    selected_vid_idxs = sorted(pool_vid_idxs[p] for p in chosen)

    # ── Pass 2: write selected frames at full resolution ─────────────────────
    cap   = cv2.VideoCapture(str(video_path))
    saved: list[Path] = []

    for i, idx in enumerate(selected_vid_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        path = output_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(path), frame)
        saved.append(path)

    cap.release()

    if not saved:
        raise RuntimeError("No frames could be saved.")

    return saved


# ── Farthest-point sampling ───────────────────────────────────────────────────

def _fps_select(thumbnails: np.ndarray, n: int) -> list[int]:
    """
    Greedily pick n row-indices from thumbnails such that each new pick
    maximises its L2 distance to the nearest already-selected thumbnail.
    Returns indices in ascending (temporal) order.
    """
    N = len(thumbnails)
    if N <= n:
        return list(range(N))

    # Seed with the first frame; track each candidate's distance to its
    # nearest selected neighbour.
    selected  = [0]
    min_dists = np.linalg.norm(thumbnails - thumbnails[0], axis=1)
    min_dists[0] = -np.inf   # exclude from future picks

    for _ in range(n - 1):
        next_i = int(np.argmax(min_dists))
        selected.append(next_i)
        # Update: each candidate's min-dist can only decrease
        d = np.linalg.norm(thumbnails - thumbnails[next_i], axis=1)
        np.minimum(min_dists, d, out=min_dists)
        min_dists[next_i] = -np.inf

    selected.sort()
    return selected
