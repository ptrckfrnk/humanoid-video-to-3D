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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

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

    mesh = _build_mesh(pcd) if build_mesh else None

    return SceneResult(point_cloud=pcd, mesh=mesh)


def _build_mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """Poisson surface reconstruction from a dense oriented point cloud."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=100)

    mesh, _densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_vertex_normals()
    return mesh
