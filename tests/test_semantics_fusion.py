"""
Unit tests for the multi-view semantic fusion (_fuse_labels) and the
projection geometry it depends on. Pure numpy — no models, no GPU.

Run with:  python -m pytest tests/  (or just:  python tests/test_semantics_fusion.py)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.semantics import _fuse_labels, _project_points
from pipeline.reconstruct import _backproject

# Shared toy camera: 64×64 image, principal point at centre
H = W = 64
K = np.array([[100.0, 0.0, 32.0],
              [0.0, 100.0, 32.0],
              [0.0, 0.0, 1.0]])
IDENTITY_POSE = np.hstack([np.eye(3), np.zeros((3, 1))])  # cam at origin, +z forward


def _frame(depth_value: float, seg_idx: int):
    """A frame with constant depth whose every pixel belongs to one segment."""
    depth = np.full((H, W), depth_value, dtype=np.float32)
    seg = np.full((H, W), seg_idx, dtype=np.int32)
    return depth, seg


def test_occluded_point_gets_no_vote():
    # Point A sits on the observed surface (z=2); point B hides behind it (z=5).
    # Both project to the image centre, but only A should receive the label.
    pts = np.array([[0.0, 0.0, 2.0],
                    [0.0, 0.0, 5.0]])
    depth, seg = _frame(depth_value=2.0, seg_idx=0)
    mask_labels = np.array([1], dtype=np.int32)   # segment 0 → label 1

    r = _fuse_labels(
        pts, [seg], mask_labels,
        IDENTITY_POSE[None], K[None], depth[None],
        n_labels=3,
    )
    assert r.n_votes[0] == 1 and r.best[0] == 1   # visible → labeled
    assert r.n_votes[1] == 0                       # occluded → no vote
    assert list(r.obs_point) == [0] and list(r.obs_mask) == [0]


def test_majority_vote_overrides_single_bad_frame():
    # Three frames see the same point; segments labeled 2, 0, 2.
    pts = np.array([[0.0, 0.0, 2.0]])
    d, seg0 = _frame(2.0, 0)
    _, seg1 = _frame(2.0, 1)
    _, seg2 = _frame(2.0, 2)
    mask_labels = np.array([2, 0, 2], dtype=np.int32)

    r = _fuse_labels(
        pts, [seg0, seg1, seg2], mask_labels,
        np.repeat(IDENTITY_POSE[None], 3, axis=0),
        np.repeat(K[None], 3, axis=0),
        np.repeat(d[None], 3, axis=0),
        n_labels=3,
    )
    assert r.n_votes[0] == 3
    assert r.best[0] == 2
    assert r.n_agree[0] == 2   # 2 of 3 views agreed with the majority


def test_unsegmented_pixels_cast_no_vote():
    pts = np.array([[0.0, 0.0, 2.0]])
    depth, seg = _frame(2.0, -1)   # SAM found nothing here

    r = _fuse_labels(
        pts, [seg], np.zeros((0,), np.int32),
        IDENTITY_POSE[None], K[None], depth[None],
        n_labels=3,
    )
    assert r.n_votes[0] == 0
    assert len(r.obs_point) == 0 and len(r.obs_mask) == 0


def test_point_outside_frustum_gets_no_vote():
    # Behind the camera and far off to the side — neither may vote.
    pts = np.array([[0.0, 0.0, -2.0],
                    [50.0, 0.0, 2.0]])
    depth, seg = _frame(2.0, 0)

    r = _fuse_labels(
        pts, [seg], np.array([1], np.int32),
        IDENTITY_POSE[None], K[None], depth[None],
        n_labels=3,
    )
    assert (r.n_votes == 0).all()


def test_occlusion_tolerance_is_relative():
    # 3% depth error at z=2 (6 cm) passes the default 5% tolerance;
    # 10% (20 cm) fails.
    depth, seg = _frame(2.0, 0)
    pts = np.array([[0.0, 0.0, 2.0 * 1.03],
                    [0.0, 0.0, 2.0 * 1.10]])

    r = _fuse_labels(
        pts, [seg], np.array([1], np.int32),
        IDENTITY_POSE[None], K[None], depth[None],
        n_labels=3,
    )
    assert r.n_votes[0] == 1
    assert r.n_votes[1] == 0


def test_observation_table_matches_visibility():
    # Two frames: the point is visible in frame 0 (depth matches) but
    # occluded in frame 1 (depth map says something nearer is in front).
    pts = np.array([[0.0, 0.0, 2.0]])
    d_match, seg0 = _frame(2.0, 0)
    d_occl, seg1 = _frame(1.0, 1)
    mask_labels = np.array([1, 2], dtype=np.int32)

    r = _fuse_labels(
        pts, [seg0, seg1], mask_labels,
        np.repeat(IDENTITY_POSE[None], 2, axis=0),
        np.repeat(K[None], 2, axis=0),
        np.stack([d_match, d_occl]),
        n_labels=3,
    )
    assert r.n_votes[0] == 1
    assert list(r.obs_point) == [0]
    assert list(r.obs_mask) == [0]   # only the frame-0 segment observed it


def test_backproject_project_roundtrip():
    # Backproject a depth map through a non-trivial pose, then reproject:
    # every world point must land back on its source pixel.
    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    t = np.array([0.1, -0.2, 0.3])
    extrinsic = np.hstack([R, t[:, None]])

    depth = np.full((1, H, W), 2.0, dtype=np.float32)
    world = _backproject(depth, extrinsic[None], K[None])   # (1, H, W, 3)

    uvs, z = _project_points(world.reshape(-1, 3), extrinsic, K)
    us_expected, vs_expected = np.meshgrid(np.arange(W), np.arange(H))

    np.testing.assert_allclose(uvs[:, 0], us_expected.ravel(), atol=1e-3)
    np.testing.assert_allclose(uvs[:, 1], vs_expected.ravel(), atol=1e-3)
    np.testing.assert_allclose(z, 2.0, atol=1e-5)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(tests)} tests passed")
