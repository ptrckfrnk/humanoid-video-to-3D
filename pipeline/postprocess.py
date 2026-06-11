"""
Convert VGGT output into a clean Open3D point cloud and optional mesh.

Steps:
  1. Flatten all frame point maps into a single cloud
  2. Filter by confidence threshold
  3. Remove NaN / inf values
  4. Statistical outlier removal (removes flying pixels / noise)
  5. Optional: Poisson surface reconstruction → watertight mesh
"""

from __future__ import annotations
import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@contextlib.contextmanager
def _silence_stderr():
    """Redirect stderr at the file-descriptor level to suppress C library noise."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved   = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)

from pipeline.reconstruct import ReconstructionResult


@dataclass
class SceneResult:
    point_cloud: o3d.geometry.PointCloud
    mesh: o3d.geometry.TriangleMesh | None


def postprocess(
    result: ReconstructionResult,
    conf_threshold: float = 1.5,
    build_mesh: bool = False,
    output_dir: Path | None = None,
) -> SceneResult:
    S, H, W, _ = result.world_points.shape

    pts    = result.world_points.reshape(-1, 3)
    confs  = result.world_points_conf.reshape(-1)
    colors = result.images.reshape(-1, 3) / 255.0   # (N, 3) float [0, 1]

    # Confidence filter
    mask = confs > conf_threshold
    pts, colors = pts[mask], colors[mask]

    # Remove non-finite coordinates
    finite = np.isfinite(pts).all(axis=1)
    pts, colors = pts[finite], colors[finite]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # Statistical outlier removal — keeps the main surface, removes noise
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Align scene so the floor is flat and Y points up
    pcd, result = _align_to_gravity(pcd, result)

    if build_mesh:
        try:
            mesh = _build_mesh(pcd)
        except Exception as e:
            print(f"  [postprocess] mesh skipped: {e}")
            mesh = None
    else:
        mesh = None

    return SceneResult(point_cloud=pcd, mesh=mesh)


def _rotation_from_vec_to_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix that rotates unit vector a onto unit vector b."""
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + Vx + Vx @ Vx * ((1 - c) / (s * s))


def _align_to_gravity(
    pcd: o3d.geometry.PointCloud,
    result: ReconstructionResult,
) -> tuple[o3d.geometry.PointCloud, ReconstructionResult]:
    """Detect floor plane via RANSAC and rotate the whole scene to Y-up."""
    if len(pcd.points) < 100:
        return pcd, result

    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=1000
        )
    except Exception:
        return pcd, result

    normal = np.array(plane_model[:3], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # Flip normal so it points away from the floor (toward the scene content)
    pts_arr = np.asarray(pcd.points)
    all_idx = set(range(len(pts_arr)))
    non_floor_idx = list(all_idx - set(inliers))
    if non_floor_idx:
        non_floor_mean = pts_arr[non_floor_idx].mean(axis=0)
        floor_mean = pts_arr[inliers].mean(axis=0)
        if np.dot(normal, non_floor_mean - floor_mean) < 0:
            normal = -normal

    R_align = _rotation_from_vec_to_vec(normal, np.array([0.0, 1.0, 0.0]))

    # Rotate point cloud in-place
    pcd.rotate(R_align, center=(0.0, 0.0, 0.0))

    # Rotate world_points tensor
    S, H, W, _ = result.world_points.shape
    wp = result.world_points.reshape(-1, 3)
    result.world_points = (R_align @ wp.T).T.reshape(S, H, W, 3)

    # For w2c transforms: rotating world by R_align means R_new = R_old @ R_align.T
    # Translation t is unchanged (it's in camera space, not world space)
    result.extrinsics[:, :3, :3] = result.extrinsics[:, :3, :3] @ R_align.T

    return pcd, result


def _build_mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh | None:
    """Poisson surface reconstruction from a dense oriented point cloud."""
    n = len(pcd.points)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    # k=20 is enough for consistent orientation and avoids hangs on noisy clouds
    pcd.orient_normals_consistent_tangent_plane(k=20)

    # Conservative depth scaling — deep Poisson is unstable on sparse clouds
    if n > 400_000:
        depth = 10
    elif n > 150_000:
        depth = 9
    else:
        depth = 8

    with _silence_stderr():
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )

    densities = np.asarray(densities)
    if len(densities) > 0 and len(mesh.vertices) > 0:
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))

    if len(mesh.vertices) == 0:
        return None

    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_vertex_normals()
    return mesh
