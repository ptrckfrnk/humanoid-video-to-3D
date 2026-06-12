"""
Self-evaluation metrics — no ground truth required.

The reconstruction makes redundant predictions (every frame predicts depth,
and the merged cloud should agree with all of them), so internal consistency
is measurable:

  multi_view_consistency  fraction of cloud points whose depth agrees with
                          the depth map in >= 2 frames — a point only the
                          source frame believes in is likely noise
  mean_views_per_point    average number of frames that corroborate a point
  median_rel_depth_error  median |z - depth_map| / depth_map over all
                          in-frustum observations — overall geometric
                          agreement between the cloud and the depth maps

These go into run_info.json for every run, so parameter and code changes
are comparable on numbers rather than visual impressions.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from pipeline.reconstruct import ReconstructionResult
from pipeline.semantics import _project_points


def geometry_metrics(
    result: ReconstructionResult,
    pcd: o3d.geometry.PointCloud,
    n_sample: int = 50_000,
    consistency_tol: float = 0.05,   # relative depth tolerance, as in fusion
    seed: int = 0,
) -> dict:
    """Cross-view consistency of the final cloud against every depth map."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return {}
    if len(pts) > n_sample:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), n_sample, replace=False)]

    S, H, W = result.depth.shape
    n_views  = np.zeros(len(pts), dtype=np.int32)
    rel_errs: list[np.ndarray] = []

    for s in range(S):
        uvs, z = _project_points(pts, result.extrinsics[s], result.intrinsics[s])
        us = np.round(uvs[:, 0]).astype(np.int64)
        vs = np.round(uvs[:, 1]).astype(np.int64)

        in_frame = (z > 0.05) & (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        if not in_frame.any():
            continue
        idx = np.flatnonzero(in_frame)
        d_map = result.depth[s][vs[idx], us[idx]]

        valid = d_map > 0
        rel = np.abs(z[idx[valid]] - d_map[valid]) / d_map[valid]
        rel_errs.append(rel)
        n_views[idx[valid][rel <= consistency_tol]] += 1

    if not rel_errs:
        return {}

    return {
        "multi_view_consistency": round(float((n_views >= 2).mean()), 4),
        "mean_views_per_point":   round(float(n_views.mean()), 2),
        "median_rel_depth_error": round(float(np.median(np.concatenate(rel_errs))), 5),
    }
