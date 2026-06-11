"""
Convert VGGT output into a clean Open3D point cloud and optional mesh.

Steps:
  1. Flatten all frame point maps into a single cloud
  2. Filter by confidence threshold
  3. Remove NaN / inf values
  4. Statistical outlier removal (removes flying pixels / noise)
  5. Optional meshing:
       tsdf (default) — volumetric TSDF fusion of the per-frame depth maps
         using the predicted camera poses (KinectFusion-style). Averages out
         per-frame depth noise, produces a coloured mesh via marching cubes,
         and avoids the Open3D Poisson segfault on Apple Silicon entirely.
       poisson — Poisson reconstruction on the merged point cloud (fallback)
"""

from __future__ import annotations
import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from pipeline.reconstruct import ReconstructionResult


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


@dataclass
class SceneResult:
    point_cloud: o3d.geometry.PointCloud
    mesh: o3d.geometry.TriangleMesh | None


def postprocess(
    result: ReconstructionResult,
    conf_threshold: float = 1.5,
    build_mesh: bool = False,
    mesh_method: str = "tsdf",
    output_dir: Path | None = None,
    console=None,
) -> SceneResult:
    # Open3D's C++ backend defaults to VerbosityLevel.Info which floods stdout
    # with progress lines that appear as blank output in rich terminals.
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    S, H, W, _ = result.world_points.shape

    pts    = result.world_points.reshape(-1, 3)
    confs  = result.world_points_conf.reshape(-1)
    colors = result.images.reshape(-1, 3) / 255.0

    n_raw = len(pts)

    # Confidence filter
    mask = confs > conf_threshold
    pts, colors = pts[mask], colors[mask]

    # Remove non-finite coordinates
    finite = np.isfinite(pts).all(axis=1)
    pts, colors = pts[finite], colors[finite]

    n_filtered = len(pts)
    if console:
        console.print(
            f"  Confidence filter (>{conf_threshold}): "
            f"{n_raw:,} → {n_filtered:,} points"
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # Statistical outlier removal — keeps the main surface, removes noise
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if console:
        console.print(
            f"  Outlier removal:              "
            f"{n_filtered:,} → {len(pcd.points):,} points"
        )

    # Align scene so the floor is flat and Y points up
    pcd, result = _align_to_gravity(pcd, result)

    if build_mesh:
        try:
            if mesh_method == "tsdf":
                mesh = _build_mesh_tsdf(result, pcd, conf_threshold, console=console)
            else:
                mesh = _build_mesh_poisson(pcd, console=console)
        except Exception as e:
            if console:
                console.print(f"  [yellow]Mesh skipped:[/yellow] {e}")
            else:
                print(f"  Mesh skipped: {e}")
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


def _print(console, msg: str) -> None:
    if console:
        console.print(msg)
    else:
        print(msg)


def _build_mesh_tsdf(
    result: ReconstructionResult,
    pcd: o3d.geometry.PointCloud,
    conf_threshold: float,
    console=None,
) -> o3d.geometry.TriangleMesh | None:
    """
    Volumetric TSDF fusion of the per-frame depth maps (KinectFusion-style).

    Each depth map is integrated into a voxel grid using its predicted camera
    pose; the mesh is extracted via marching cubes. Because every depth map
    contributes to the signed-distance average, per-frame depth noise cancels
    out instead of accumulating — and vertex colours come directly from the
    video frames. Works identically on CUDA / MPS / CPU (pure C++ on CPU).

    The (already aligned & filtered) point cloud is used only to size the
    voxel grid to the scene.
    """
    S, H, W = result.depth.shape

    # Scene-adaptive resolution: ~256 voxels across the scene diagonal,
    # clamped to [4 mm, 3 cm]. sdf_trunc at 4 voxels is the usual heuristic.
    diag  = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
    voxel = float(np.clip(diag / 256.0, 0.004, 0.03))

    # Truncate depth just past the furthest confident return — far outliers
    # would otherwise carve through good geometry. (Not the bbox diagonal:
    # camera-space depth can exceed the scene's own extent.)
    conf_depths = result.depth[result.depth_conf >= conf_threshold]
    depth_trunc = float(np.percentile(conf_depths, 99) * 1.2) if len(conf_depths) else diag

    _print(console, f"  TSDF fusion: {S} depth maps, voxel={voxel * 100:.1f} cm…")
    t0 = time.perf_counter()

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=4 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for s in range(S):
        depth = result.depth[s].astype(np.float32).copy()
        depth[result.depth_conf[s] < conf_threshold] = 0.0   # 0 = invalid pixel

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(result.images[s])),
            o3d.geometry.Image(depth),
            depth_scale=1.0,            # depth is already metric
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        K = result.intrinsics[s]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        extrinsic = np.eye(4)
        extrinsic[:3, :4] = result.extrinsics[s]   # cam-from-world, as expected
        volume.integrate(rgbd, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()
    _print(
        console,
        f"  [green]✓[/green] TSDF fusion done  ({time.perf_counter() - t0:.1f}s) — "
        f"{len(mesh.vertices):,} vertices  {len(mesh.triangles):,} faces",
    )
    if len(mesh.vertices) == 0:
        return None

    # Drop small floating components (isolated noise blobs)
    cluster_ids, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_ids   = np.asarray(cluster_ids)
    cluster_sizes = np.asarray(cluster_sizes)
    if len(cluster_sizes) > 1:
        min_tris = max(100, int(0.01 * cluster_sizes.max()))
        mesh.remove_triangles_by_mask(cluster_sizes[cluster_ids] < min_tris)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    return mesh


def _build_mesh_poisson(
    pcd: o3d.geometry.PointCloud,
    console=None,
) -> o3d.geometry.TriangleMesh | None:
    """Poisson surface reconstruction from a dense oriented point cloud."""
    n_in = len(pcd.points)

    # ── Downsample ──────────────────────────────────────────────────────────
    # Use random_down_sample (not voxel): voxel size must be tuned to surface
    # density, not volume — empty air inflates volume and makes the formula
    # produce nearly zero surviving points.
    _TARGET = 100_000
    if n_in > _TARGET:
        pcd = pcd.random_down_sample(float(_TARGET) / n_in)
        n   = len(pcd.points)
        _print(console, f"  Mesh downsample: {n_in:,} → {n:,} pts")
    else:
        n = n_in

    # depth=6 → 64³ = 260k octree cells → ~10–30 s on M4 Pro (single CPU core).
    # depth=7 segfaults on some Open3D builds on Apple Silicon; depth=6 is safe
    # for a debug/smoke-test run. GPU workstation can go to depth=9 or 10.
    depth = 6

    # ── Three sequential steps, each prints start + elapsed on completion ───
    # Plain console.print() — no background threads, no spinner animation.
    # Reliable on all macOS terminals; Rich Progress + blocking C++ doesn't
    # animate reliably when the GIL is released inside the C++ call.

    _print(console, "  Estimating normals…")
    t0 = time.perf_counter()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    _print(console, f"  [green]✓[/green] Normals estimated  ({time.perf_counter()-t0:.1f}s)")

    _print(console, "  Orienting normals (k=20)…")
    t0 = time.perf_counter()
    pcd.orient_normals_consistent_tangent_plane(k=20)
    _print(console, f"  [green]✓[/green] Normals oriented  ({time.perf_counter()-t0:.1f}s)")

    _print(console, f"  Poisson reconstruction  depth={depth}  (~10–30 s on MPS)…")
    _print(console, "  [dim](single CPU core — ~7 % in Activity Monitor is normal)[/dim]")
    t0 = time.perf_counter()
    # No _silence_stderr() here: suppressing fd 2 hides segfault messages and
    # makes silent crashes look like normal exits. Open3D verbosity=Error already
    # suppresses routine info logging; only genuine errors will appear.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    elapsed = time.perf_counter() - t0
    _print(
        console,
        f"  [green]✓[/green] Poisson done  ({elapsed:.1f}s) — "
        f"{len(mesh.vertices):,} vertices  {len(mesh.triangles):,} faces",
    )

    # ── Trim low-density fringe artefacts ────────────────────────────────────
    densities = np.asarray(densities)
    if len(densities) > 0 and len(mesh.vertices) > 0:
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))

    if len(mesh.vertices) == 0:
        return None

    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_vertex_normals()
    return mesh
