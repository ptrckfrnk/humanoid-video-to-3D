"""
Open-vocabulary 3D querying support.

The semantic stage stores, for every SAM2 segment, its CLIP image embedding,
plus a sparse observation table linking each 3D point to the segments it was
visible in (after the z-buffer occlusion test). A point's semantic feature is
the mean of its segments' features — but instead of materialising a dense
(N, 512) per-point feature field (gigabytes at GPU-run scale), scoring stays
segment-side and exploits linearity:

    score(point) = text · mean(feat_s)  =  mean(text · feat_s)

so a text query costs one (M, D) @ (D,) product over M segments (thousands)
plus a sparse mean over the observation table — milliseconds, at ~30x less
memory than a per-point feature field, with identical results.

Relevancy follows LERF (Kerr et al., ICCV 2023): the query similarity is
contrasted against canonical negative prompts ("object", "things", ...) via
pairwise softmax, which yields far crisper heatmaps than raw cosine values.

Only numpy is required here; CLIP itself is loaded by query.py.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# LERF canonical negatives: generic prompts the query must beat
CANONICAL_NEGATIVES = ["object", "things", "stuff", "texture"]

# Coarse turbo colormap anchors (avoids a matplotlib dependency)
_TURBO = np.array([
    (48, 18, 59), (62, 86, 196), (33, 150, 245), (26, 206, 202),
    (98, 242, 121), (183, 244, 53), (249, 210, 41), (244, 121, 24),
    (122, 4, 3),
], dtype=np.float64)


# ── Feature bundle I/O ────────────────────────────────────────────────────────

def save_feature_bundle(
    path: Path,
    *,
    points: np.ndarray,         # (N, 3) float32
    colors: np.ndarray,         # (N, 3) uint8 RGB
    mask_feats: np.ndarray,     # (M, D) float16, L2-normalised
    obs_point: np.ndarray,      # (K,) int32 — observation table: point index
    obs_mask: np.ndarray,       # (K,) int32 — observation table: segment index
    point_labels: np.ndarray,   # (N,) int32 label index; -1 = unobserved
    label_set: list[str],
    clip_model: str,
    clip_pretrained: str,
) -> None:
    np.savez_compressed(
        path,
        points=points.astype(np.float32),
        colors=colors.astype(np.uint8),
        mask_feats=mask_feats.astype(np.float16),
        obs_point=obs_point.astype(np.int32),
        obs_mask=obs_mask.astype(np.int32),
        point_labels=point_labels.astype(np.int32),
        label_set=np.array(label_set),
        clip_model=np.array(clip_model),
        clip_pretrained=np.array(clip_pretrained),
    )


def load_feature_bundle(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ── Query scoring ─────────────────────────────────────────────────────────────

def relevancy_scores(
    mask_feats:     np.ndarray,   # (M, D) L2-normalised
    text_feat:      np.ndarray,   # (D,)   L2-normalised query embedding
    negative_feats: np.ndarray,   # (C, D) L2-normalised canonical negatives
    temperature:    float = 10.0,
) -> np.ndarray:
    """
    LERF-style relevancy per segment: pairwise softmax of the query similarity
    against each canonical negative, taking the worst case. 0.5 means "no more
    query-like than a generic object"; near 1.0 means a confident match.
    """
    mask_feats = mask_feats.astype(np.float32)
    s_query = mask_feats @ text_feat.astype(np.float32)            # (M,)
    s_neg   = mask_feats @ negative_feats.astype(np.float32).T     # (M, C)

    # Pairwise softmax against each negative; keep the hardest one
    e_q = np.exp(temperature * s_query)[:, None]                   # (M, 1)
    e_n = np.exp(temperature * s_neg)                              # (M, C)
    return (e_q / (e_q + e_n)).min(axis=1)


def aggregate_point_scores(
    seg_scores: np.ndarray,   # (M,) per-segment score
    obs_point:  np.ndarray,   # (K,) observation table: point index
    obs_mask:   np.ndarray,   # (K,) observation table: segment index
    n_points:   int,
) -> np.ndarray:
    """
    Sparse mean of segment scores over each point's observations.
    Returns (N,) float32; NaN for points never observed in any segment.
    """
    sums = np.bincount(obs_point, weights=seg_scores[obs_mask], minlength=n_points)
    cnts = np.bincount(obs_point, minlength=n_points)

    out = np.full(n_points, np.nan, dtype=np.float32)
    seen = cnts > 0
    out[seen] = (sums[seen] / cnts[seen]).astype(np.float32)
    return out


def score_colormap(scores: np.ndarray) -> np.ndarray:
    """
    Map scores in [0, 1] to turbo RGB; NaN (unobserved) renders dark grey.
    Returns (N, 3) uint8.
    """
    out = np.full((len(scores), 3), 60, dtype=np.uint8)
    seen = np.isfinite(scores)
    if not seen.any():
        return out

    t = np.clip(scores[seen], 0.0, 1.0) * (len(_TURBO) - 1)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(_TURBO) - 1)
    frac = (t - lo)[:, None]
    out[seen] = np.clip(_TURBO[lo] * (1 - frac) + _TURBO[hi] * frac, 0, 255).astype(np.uint8)
    return out
