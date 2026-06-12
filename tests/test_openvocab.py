"""
Unit tests for open-vocabulary query scoring (pipeline/openvocab.py).
Pure numpy — no CLIP, no GPU.

Run with:  python -m pytest tests/  (or just:  python tests/test_openvocab.py)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.openvocab import (
    aggregate_point_scores,
    load_feature_bundle,
    relevancy_scores,
    save_feature_bundle,
    score_colormap,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_aggregate_is_mean_over_observations():
    # Point 0 observed by segments 0 (score 1.0) and 1 (score 0.0) → mean 0.5.
    # Point 1 observed once by segment 1 → 0.0. Point 2 never observed → NaN.
    seg_scores = np.array([1.0, 0.0])
    obs_point = np.array([0, 0, 1], dtype=np.int32)
    obs_mask = np.array([0, 1, 1], dtype=np.int32)

    scores = aggregate_point_scores(seg_scores, obs_point, obs_mask, n_points=3)
    assert scores[0] == 0.5
    assert scores[1] == 0.0
    assert np.isnan(scores[2])


def test_relevancy_separates_match_from_generic():
    # Segment A aligned with the query, segment B aligned with a negative.
    q = _unit([1.0, 0.0, 0.0])
    neg = _unit([0.0, 1.0, 0.0])
    mask_feats = np.stack([q, neg]).astype(np.float16)

    rel = relevancy_scores(mask_feats, q, neg[None])
    assert rel[0] > 0.95          # confident match
    assert rel[1] < 0.05          # confidently not the query
    assert np.all((rel >= 0) & (rel <= 1))


def test_relevancy_uses_hardest_negative():
    # A segment equally similar to query and one negative scores 0.5 against
    # it; adding an easier negative must not raise the relevancy.
    feat = _unit([1.0, 1.0, 0.0])
    q = _unit([1.0, 0.0, 0.0])
    hard = _unit([0.0, 1.0, 0.0])
    easy = _unit([0.0, 0.0, 1.0])

    rel_hard_only = relevancy_scores(feat[None], q, hard[None])
    rel_both = relevancy_scores(feat[None], q, np.stack([hard, easy]))
    np.testing.assert_allclose(rel_hard_only, 0.5, atol=1e-6)
    np.testing.assert_allclose(rel_both, rel_hard_only)


def test_colormap_handles_nan_and_range():
    scores = np.array([0.0, 0.5, 1.0, np.nan])
    colors = score_colormap(scores)
    assert colors.shape == (4, 3) and colors.dtype == np.uint8
    assert tuple(colors[3]) == (60, 60, 60)          # unobserved → grey
    assert not np.array_equal(colors[0], colors[2])  # ends of the scale differ


def test_feature_bundle_roundtrip():
    rng = np.random.default_rng(0)
    kwargs = dict(
        points=rng.normal(size=(10, 3)).astype(np.float32),
        colors=rng.integers(0, 255, (10, 3), dtype=np.uint8),
        mask_feats=rng.normal(size=(4, 8)).astype(np.float16),
        obs_point=np.array([0, 1, 2], dtype=np.int32),
        obs_mask=np.array([0, 1, 3], dtype=np.int32),
        point_labels=np.full(10, -1, dtype=np.int32),
        label_set=["chair", "table"],
        clip_model="ViT-B-32-quickgelu",
        clip_pretrained="openai",
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bundle.npz"
        save_feature_bundle(path, **kwargs)
        loaded = load_feature_bundle(path)

    for key in ("points", "colors", "mask_feats", "obs_point", "obs_mask", "point_labels"):
        np.testing.assert_array_equal(loaded[key], kwargs[key])
    assert list(loaded["label_set"]) == ["chair", "table"]
    assert str(loaded["clip_model"]) == "ViT-B-32-quickgelu"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(tests)} tests passed")
