"""
Unit tests for the self-evaluation metrics (pipeline/metrics.py).
Synthetic geometry only — no models, no GPU.

Run with:  python -m pytest tests/  (or just:  python tests/test_metrics.py)
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.reconstruct import ReconstructionResult, _backproject
from pipeline.metrics import geometry_metrics

H = W = 32
K = np.array([[50.0, 0.0, 16.0],
              [0.0, 50.0, 16.0],
              [0.0, 0.0, 1.0]])


def _make_result(depths: np.ndarray) -> tuple[ReconstructionResult, o3d.geometry.PointCloud]:
    """Two identity-pose cameras; cloud backprojected from frame 0."""
    S = len(depths)
    extrinsics = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (S, 1, 1))
    intrinsics = np.tile(K, (S, 1, 1))
    conf = np.full((S, H, W), 10.0, dtype=np.float32)

    world = _backproject(depths, extrinsics, intrinsics)
    result = ReconstructionResult(
        images=np.zeros((S, H, W, 3), np.uint8),
        world_points=world.astype(np.float32),
        world_points_conf=conf,
        depth=depths,
        depth_conf=conf,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame_paths=[Path(f"{s}.png") for s in range(S)],
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world[0].reshape(-1, 3).astype(np.float64))
    return result, pcd


def test_consistent_scene_scores_high():
    # Both frames agree on a flat plane at z=2 → every cloud point is
    # corroborated by 2 views with ~zero depth error.
    depths = np.full((2, H, W), 2.0, dtype=np.float32)
    result, pcd = _make_result(depths)

    m = geometry_metrics(result, pcd)
    assert m["multi_view_consistency"] == 1.0
    assert m["mean_views_per_point"] == 2.0
    assert m["median_rel_depth_error"] < 1e-4


def test_inconsistent_frame_lowers_consistency():
    # Frame 1 predicts a plane 25% nearer — outside the 5% tolerance, so
    # no point gets a second corroborating view.
    depths = np.stack([
        np.full((H, W), 2.0, dtype=np.float32),
        np.full((H, W), 1.5, dtype=np.float32),
    ])
    result, pcd = _make_result(depths)

    m = geometry_metrics(result, pcd)
    assert m["multi_view_consistency"] == 0.0   # nothing has >= 2 views
    assert m["mean_views_per_point"] == 1.0     # only the source frame agrees
    assert m["median_rel_depth_error"] > 0.05


def test_empty_cloud_returns_empty():
    depths = np.full((1, H, W), 2.0, dtype=np.float32)
    result, _ = _make_result(depths)
    assert geometry_metrics(result, o3d.geometry.PointCloud()) == {}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(tests)} tests passed")
