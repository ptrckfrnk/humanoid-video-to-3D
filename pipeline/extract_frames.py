"""Extract evenly-spaced frames from a video file."""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np


def extract_frames(
    video_path: Path,
    n_frames: int,
    output_dir: Path,
) -> list[Path]:
    """
    Sample n_frames evenly from the video and save them as PNG files.

    Returns a list of saved image paths in temporal order.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0:
        raise RuntimeError(f"Video reports 0 frames: {video_path}")

    n_frames = min(n_frames, total)
    indices  = np.linspace(0, total - 1, n_frames, dtype=int)

    saved: list[Path] = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        path = output_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(path), frame)
        saved.append(path)

    cap.release()

    if not saved:
        raise RuntimeError("No frames could be extracted from the video.")

    return saved
