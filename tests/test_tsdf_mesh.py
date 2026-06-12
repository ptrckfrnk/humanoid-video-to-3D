"""
Unit test for TSDF mesh fusion: integrate two synthetic views of a flat
plane at z = 2 m and verify the extracted mesh lies on that plane.

Run with:  python -m pytest tests/  (or just:  python tests/test_tsdf_mesh.py)
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.reconstruct import ReconstructionResult, _backproject
from pipeline.postprocess import _build_mesh_tsdf

H = W = 64
K = np.array([[100.0, 0.0, 32.0],
              [0.0, 100.0, 32.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)
PLANE_Z = 2.0


def _make_result() -> ReconstructionResult:
    # Two cameras at x = 0 and x = 0.1, both looking straight at a
    # fronto-parallel plane → constant depth maps.
    extrinsics = np.zeros((2, 3, 4))
    for s, cam_x in enumerate([0.0, 0.1]):
        extrinsics[s, :3, :3] = np.eye(3)
        extrinsics[s, :3, 3] = [-cam_x, 0.0, 0.0]   # t = -R @ cam_center

    depth = np.full((2, H, W), PLANE_Z, dtype=np.float32)
    conf = np.full((2, H, W), 10.0, dtype=np.float32)
    conf[1, :, : W // 4] = 0.0   # low-confidence strip → must be ignored

    intrinsics = np.stack([K, K])
    world_points = _backproject(depth, extrinsics, intrinsics)

    return ReconstructionResult(
        images=np.full((2, H, W, 3), 128, dtype=np.uint8),
        world_points=world_points.astype(np.float32),
        world_points_conf=conf,
        depth=depth,
        depth_conf=conf,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame_paths=[Path("a.png"), Path("b.png")],
    )


def test_tsdf_mesh_recovers_plane():
    result = _make_result()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        result.world_points.reshape(-1, 3).astype(np.float64)
    )

    mesh = _build_mesh_tsdf(result, pcd, conf_percentile=20.0)

    assert mesh is not None
    assert len(mesh.vertices) > 100
    assert mesh.has_vertex_colors()

    # Every vertex must sit on the z = 2 plane (within a few voxels)
    diag = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
    voxel = float(np.clip(diag / 256.0, 0.004, 0.03))
    z = np.asarray(mesh.vertices)[:, 2]
    assert np.abs(z - PLANE_Z).max() < 3 * voxel, (
        f"mesh deviates {np.abs(z - PLANE_Z).max():.4f} m from the plane"
    )


if __name__ == "__main__":
    test_tsdf_mesh_recovers_plane()
    print("  ✓ test_tsdf_mesh_recovers_plane\n\n1 test passed")
