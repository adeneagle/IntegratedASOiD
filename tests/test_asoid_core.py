"""Tests for the vendored A-SOID numeric core.

The index-mapping tests are the load-bearing ones: if `row_to_frames` or
`downsample_targets` drift, nothing raises — the uncertainty heatmap just points
at the wrong part of the video and the user relabels the wrong frames.
"""

import numpy as np
import pytest

from backend.asoid_core import features as feat
from backend.asoid_core import indexing as idx
from backend.asoid_core import pose_io, predict
from backend.asoid_core.active_learning import RFActiveLearner, fingerprint_bodyparts

from tests.conftest import BODYPARTS, FPS, make_behavior_series, make_pose_array


# ── indexing ──────────────────────────────────────────────────────────────

def test_frames2integ_matches_upstream_formula():
    # B_extract_features.py:60 — round(framerate * (min_duration / 0.1))
    assert idx.frames2integ_from(30, 0.1) == 30
    assert idx.frames2integ_from(60, 0.1) == 60
    assert idx.frames2integ_from(30, 0.2) == 60
    assert idx.frames2integ_from(120, 0.4) == 480


def test_bin_width_is_frames2integ_over_ten():
    assert idx.bin_width_from(30) == 3     # 100 ms at 30 fps
    assert idx.bin_width_from(60) == 6
    assert idx.bin_width_from(5) == 1      # never zero


def test_row_to_frames_is_contiguous_and_non_overlapping():
    bw = 3
    seen = []
    for row in range(10):
        frames = list(idx.row_to_frames(row, bw))
        assert len(frames) == bw
        seen.extend(frames)
    assert seen == list(range(30)), "bins must tile the frame axis without gaps"


def test_frame_to_row_inverts_row_to_frames():
    bw = 4
    for row in range(25):
        for f in idx.row_to_frames(row, bw):
            assert idx.frame_to_row(f, bw) == row


def test_expand_rows_to_frames_pads_the_tail():
    # Feature extraction drops the trailing partial bin, so expansion is short;
    # the tail must edge-pad rather than leave the array misaligned.
    out = idx.expand_rows_to_frames(np.array([5, 6, 7]), 3, n_frames=11)
    assert out.tolist() == [5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7]


def test_expand_rows_to_frames_truncates_when_too_long():
    out = idx.expand_rows_to_frames(np.array([1, 2, 3]), 3, n_frames=5)
    assert out.tolist() == [1, 1, 1, 2, 2]


def test_expand_rows_handles_2d_probabilities():
    proba = np.array([[0.1, 0.9], [0.8, 0.2]])
    out = idx.expand_rows_to_frames(proba, 2, n_frames=4)
    assert out.shape == (4, 2)
    assert out[0].tolist() == [0.1, 0.9]
    assert out[3].tolist() == [0.8, 0.2]


def test_downsample_targets_takes_the_bin_mode():
    # Bin straddling a boundary gets the label that dominates it.
    targets = np.array([0, 0, 1,   1, 1, 1,   2, 2, 0])
    assert idx.downsample_targets(targets, 3, 3).tolist() == [0, 1, 2]


def test_downsample_targets_respects_requested_row_count():
    targets = np.zeros(300, dtype=int)
    assert idx.downsample_targets(targets, 3, 50).shape == (50,)


# ── features ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_nchoose2_pair_count(n):
    pairs = feat.fast_nchoose2(n, 2)
    assert pairs.shape == (2, n * (n - 1) // 2)


@pytest.mark.parametrize("n,expected", [(2, 4), (3, 9), (4, 16), (5, 25)])
def test_feature_width_formula(n, expected):
    # nchoose2(n)*2 (length + angle per pair) + n (displacement per part)
    assert feat.feature_width(n) == expected


def test_extract_features_shape_matches_feature_width(pose_array):
    f = feat.extract_features([pose_array], idx.frames2integ_from(FPS, 0.1))
    assert f.ndim == 2
    assert f.shape[1] == feat.feature_width(len(BODYPARTS))
    # One row per whole bin.
    assert f.shape[0] == pytest.approx(pose_array.shape[0] / 3, rel=0.02)


def test_extract_features_all_shifts_returns_bin_width_variants(pose_array):
    frames2integ = idx.frames2integ_from(FPS, 0.1)
    per_rec = feat.extract_features([pose_array], frames2integ, all_shifts=True)
    assert len(per_rec) == 1
    assert len(per_rec[0]) == idx.bin_width_from(frames2integ)


def test_extract_features_is_deterministic(pose_array):
    f2i = idx.frames2integ_from(FPS, 0.1)
    a = feat.extract_features([pose_array], f2i)
    b = feat.extract_features([pose_array], f2i)
    np.testing.assert_array_equal(a, b)


def test_features_separate_the_two_behaviors(pose_array, behavior_series):
    """The synthetic task must actually be encoded in the features."""
    f2i = idx.frames2integ_from(FPS, 0.1)
    f = feat.extract_features([pose_array], f2i)
    y = idx.downsample_targets(behavior_series, idx.bin_width_from(f2i), f.shape[0])
    n = min(len(f), len(y))
    f, y = f[:n], y[:n]
    # Mean pairwise length (first block of columns) must differ by class.
    n_pairs = len(BODYPARTS) * (len(BODYPARTS) - 1) // 2
    mean_len = f[:, :n_pairs].mean(axis=1)
    assert mean_len[y == 1].mean() > mean_len[y == 0].mean() * 1.5


# ── pose_io ───────────────────────────────────────────────────────────────

def test_load_dlc_and_inspect(dlc_csv):
    info = pose_io.inspect_pose(dlc_csv, "deeplabcut")
    assert info["n_frames"] == 3000
    assert sorted(info["bodyparts"]) == sorted(BODYPARTS)
    assert info["is_3d"] is False
    assert info["has_likelihood"] is True


def test_load_sleap_and_inspect(sleap_h5):
    info = pose_io.inspect_pose(sleap_h5, "sleap")
    assert info["n_frames"] == 3000
    assert sorted(info["bodyparts"]) == sorted(BODYPARTS)
    assert info["has_likelihood"] is True


def test_select_columns_excludes_likelihood(dlc_csv):
    df = pose_io.load_pose(dlc_csv, "deeplabcut")
    idx_sel, idx_llh, is_3d = pose_io.select_columns(df, BODYPARTS, "deeplabcut")
    assert len(idx_sel) == len(BODYPARTS) * 2, "x/y only"
    assert len(idx_llh) == len(BODYPARTS)
    assert not set(idx_sel) & set(idx_llh)
    assert is_3d is False


def test_select_columns_rejects_unknown_bodypart(dlc_csv):
    df = pose_io.load_pose(dlc_csv, "deeplabcut")
    with pytest.raises(ValueError, match="not found"):
        pose_io.select_columns(df, ["nonexistent"], "deeplabcut")


def test_select_columns_follows_file_order_not_caller_order(dlc_csv):
    """Column order must depend only on the file, so a checkpoint's feature
    layout is reproducible however the user ticked the boxes."""
    df = pose_io.load_pose(dlc_csv, "deeplabcut")
    a, _, _ = pose_io.select_columns(df, BODYPARTS, "deeplabcut")
    b, _, _ = pose_io.select_columns(df, list(reversed(BODYPARTS)), "deeplabcut")
    assert a == b


def test_adp_filt_holds_last_confident_position(dlc_csv_lowconf):
    arr, is_3d, perc = pose_io.load_and_filter(
        dlc_csv_lowconf, "deeplabcut", BODYPARTS, llh_value=0.6
    )
    assert arr.shape == (3000, len(BODYPARTS) * 2)
    assert not np.isnan(arr).any()
    # 'nose' lost tracking for frames 100..199 -> its coords must be frozen.
    nose_col = BODYPARTS.index("nose") * 2
    frozen = arr[100:200, nose_col]
    assert np.allclose(frozen, frozen[0]), "low-likelihood run should hold one position"
    # And it should be reported as rejected.
    assert max(perc) > 0.03


def test_load_and_filter_reports_no_rejection_on_clean_data(dlc_csv):
    _arr, _is3d, perc = pose_io.load_and_filter(
        dlc_csv, "deeplabcut", BODYPARTS, llh_value=0.6
    )
    assert max(perc) < 0.05


def test_keypoints_for_overlay_is_unfiltered(dlc_csv_lowconf):
    names, arr = pose_io.keypoints_for_overlay(dlc_csv_lowconf, "deeplabcut", BODYPARTS)
    assert len(names) == len(BODYPARTS)
    assert arr.shape == (3000, len(BODYPARTS), 3)
    nose = names.index("nose")
    # The overlay must still show the LOW likelihood, so bad tracking is visible.
    assert arr[150, nose, 2] < 0.6


def test_unsupported_origin_rejected(dlc_csv):
    with pytest.raises(ValueError, match="not supported"):
        pose_io.load_pose(dlc_csv, "not_a_real_tracker")


# ── predict ───────────────────────────────────────────────────────────────

def test_normalized_entropy_bounds():
    confident = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert predict.normalized_entropy(confident)[0] == pytest.approx(0.0)
    assert predict.normalized_entropy(uniform)[0] == pytest.approx(1.0)


def test_normalized_entropy_is_monotonic():
    probs = np.array([
        [0.99, 0.01],
        [0.75, 0.25],
        [0.50, 0.50],
    ])
    unc = predict.normalized_entropy(probs)
    assert unc[0] < unc[1] < unc[2]


def test_normalized_entropy_single_class_is_zero():
    assert predict.normalized_entropy(np.array([[1.0]]))[0] == 0.0


# ── active learning ───────────────────────────────────────────────────────

@pytest.fixture
def learnable_dataset():
    """Features/targets for a task that is solvable but not trivially so."""
    behavior = make_behavior_series(4000, block=100)
    pose = make_pose_array(4000, behavior, seed=2, noise=6.0)
    f2i = idx.frames2integ_from(FPS, 0.1)
    X = feat.extract_features([pose], f2i)
    y = idx.downsample_targets(behavior, idx.bin_width_from(f2i), X.shape[0])
    n = min(len(X), len(y))
    return X[:n], y[:n]


def test_active_learning_improves_and_converges(learnable_dataset):
    X, y = learnable_dataset
    learner = RFActiveLearner(
        X, y, class_names=["restA", "restB", "other"],
        init_ratio=0.01, max_iter=10, max_samples_iter=15, conf_threshold=0.6,
    )
    events = []
    result = learner.run(progress_cb=events.append)

    iterations = [e for e in events if e["event"] == "iteration"]
    assert len(iterations) >= 2, "loop should refine at least once"
    assert result.full_macro_f1 > 0.7, "ceiling model must learn the task"
    # Active learning should not degrade the seed model.
    assert iterations[-1]["macro_f1"] >= iterations[0]["macro_f1"] - 1e-9
    # Low-confidence pool must shrink as the model improves.
    assert iterations[-1]["n_lowconf_remaining"] <= iterations[0]["n_lowconf_remaining"]
    # And it must get there on a fraction of the data.
    assert iterations[-1]["n_train"] < X.shape[0] * 0.8


def test_active_learning_emits_start_and_ceiling(learnable_dataset):
    X, y = learnable_dataset
    events = []
    RFActiveLearner(X, y, class_names=["a", "b", "other"],
                    init_ratio=0.05, max_iter=2).run(progress_cb=events.append)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start"
    assert "ceiling" in kinds
    assert kinds[-1] == "done"
    start = events[0]
    assert start["n_samples"] == X.shape[0]
    assert start["n_features"] == X.shape[1]


def test_active_learning_rejects_single_class():
    X = np.random.RandomState(0).rand(200, 16)
    y = np.zeros(200, dtype=int)
    learner = RFActiveLearner(X, y, class_names=["only", "other"], init_ratio=0.5)
    with pytest.raises(ValueError, match="at least two"):
        learner.run()


def test_tiny_init_ratio_still_seeds_every_class(learnable_dataset):
    """int(n * init_ratio) rounds to 0 for small classes; the seed must still
    contain every class rather than dropping it (or raising)."""
    X, y = learnable_dataset
    events = []
    RFActiveLearner(X, y, class_names=["a", "b", "other"],
                    init_ratio=1e-9, max_iter=1).run(progress_cb=events.append)
    seed = next(e for e in events if e["event"] == "iteration")
    assert seed["n_train"] == np.unique(y).size  # one sample per class


def _sparse_labels(y, per_class=40):
    """A few dozen labeled rows per behavior, everything else "other" (code 2)."""
    rng = np.random.RandomState(0)
    sparse = np.full_like(y, 2)
    for c in (0, 1):
        sparse[rng.choice(np.where(y == c)[0], per_class, replace=False)] = c
    return sparse


def test_sparse_labels_train_at_the_default_init_ratio(learnable_dataset):
    """Real projects look like this. At the default init_ratio of 0.01 every
    behavior used to round to zero seed samples, and training with
    exclude_other failed outright."""
    X, y = learnable_dataset
    events = []
    RFActiveLearner(X, _sparse_labels(y), class_names=["a", "b", "other"],
                    init_ratio=0.01, conf_threshold=0.6, exclude_other=True,
                    max_iter=30).run(progress_cb=events.append)
    iterations = [e for e in events if e["event"] == "iteration"]
    assert len(iterations) >= 2, "loop should refine, not stop on the seed"
    assert iterations[-1]["macro_f1"] > 0.9


def test_sparse_labels_with_other_seed_every_behavior(learnable_dataset):
    """Without exclude_other this used to raise "seeds only one class"."""
    X, y = learnable_dataset
    result = RFActiveLearner(X, _sparse_labels(y), class_names=["a", "b", "other"],
                             init_ratio=0.01, max_iter=1).run()
    assert set(result.model.classes_) == {0, 1, 2}


def test_exclude_other_drops_the_last_class(learnable_dataset):
    X, y = learnable_dataset
    # Mark a chunk as "other" (class 2).
    y = y.copy()
    y[:200] = 2
    learner = RFActiveLearner(X, y, class_names=["a", "b", "other"],
                              init_ratio=0.1, max_iter=2, exclude_other=True)
    result = learner.run()
    assert 2 not in result.model.classes_


def test_cancellation_propagates_out_of_the_loop(learnable_dataset):
    X, y = learnable_dataset

    class Stop(Exception):
        pass

    def cb(payload):
        if payload.get("event") == "iteration":
            raise Stop()

    learner = RFActiveLearner(X, y, class_names=["a", "b", "other"],
                              init_ratio=0.05, max_iter=50)
    with pytest.raises(Stop):
        learner.run(progress_cb=cb)


# ── body-part fingerprint ────────────────────────────────────────────────

def test_fingerprint_is_stable_and_order_sensitive():
    a = fingerprint_bodyparts(["nose", "tail"], False)
    assert a == fingerprint_bodyparts(["nose", "tail"], False)
    assert a != fingerprint_bodyparts(["tail", "nose"], False)
    assert a != fingerprint_bodyparts(["nose", "tail"], True)
    assert a != fingerprint_bodyparts(["nose", "tail", "ear"], False)
