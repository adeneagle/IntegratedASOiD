"""Tests for the pose service: label alignment and feature extraction."""

import os

import numpy as np
import pytest

from backend.services import pose_service
from tests.conftest import BODYPARTS, FPS, wait_for, write_sparse_labels


# ── Label parsing ─────────────────────────────────────────────────────────

def test_read_label_csv_parses_sparse_one_hot(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("time,walk,groom\n0,1,0\n1,1,0\n5,0,1\n", encoding="utf-8")
    names, frame_to_col = pose_service.read_label_csv(str(path))
    assert names == ["walk", "groom"]
    assert frame_to_col == {0: 0, 1: 0, 5: 1}


def test_read_label_csv_handles_header_only(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("time\n", encoding="utf-8")
    names, frame_to_col = pose_service.read_label_csv(str(path))
    assert names == []
    assert frame_to_col == {}


def test_read_label_csv_ignores_all_zero_rows(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("time,walk,groom\n0,0,0\n1,0,1\n", encoding="utf-8")
    _names, frame_to_col = pose_service.read_label_csv(str(path))
    assert 0 not in frame_to_col, "a row with no positive column is not a label"
    assert frame_to_col[1] == 1


def test_build_class_names_puts_other_last(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("time,walk\n0,1\n", encoding="utf-8")
    b.write_text("time,groom\n0,1\n", encoding="utf-8")
    names = pose_service.build_class_names([str(a), str(b)])
    assert names[-1] == "other"
    assert set(names[:-1]) == {"walk", "groom"}


def test_build_class_names_dedupes_across_files(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("time,walk,groom\n0,1,0\n", encoding="utf-8")
    b.write_text("time,walk\n0,1\n", encoding="utf-8")
    names = pose_service.build_class_names([str(a), str(b)])
    assert names.count("walk") == 1


def test_build_class_names_skips_missing_files(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("time,walk\n0,1\n", encoding="utf-8")
    names = pose_service.build_class_names([str(a), str(tmp_path / "nope.csv"), None])
    assert names == ["walk", "other"]


# ── Sparse -> dense ───────────────────────────────────────────────────────

def test_dense_targets_fills_unlabeled_with_other(tmp_path):
    path = tmp_path / "l.csv"
    # Only frames 2 and 3 are labeled; everything else is unlabeled.
    path.write_text("time,walk\n2,1\n3,1\n", encoding="utf-8")
    class_names = ["walk", "other"]
    dense = pose_service.dense_targets_from_csv(str(path), 6, class_names)
    other = len(class_names) - 1
    assert dense.tolist() == [other, other, 0, 0, other, other]


def test_dense_targets_maps_columns_onto_global_class_order(tmp_path):
    """A file listing behaviors in a different order must still map correctly."""
    path = tmp_path / "l.csv"
    path.write_text("time,groom,walk\n0,1,0\n1,0,1\n", encoding="utf-8")
    class_names = ["walk", "groom", "other"]  # global order differs from the file's
    dense = pose_service.dense_targets_from_csv(str(path), 3, class_names)
    assert dense[0] == class_names.index("groom")
    assert dense[1] == class_names.index("walk")
    assert dense[2] == class_names.index("other")


def test_dense_targets_ignores_out_of_range_frames(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("time,walk\n0,1\n999,1\n", encoding="utf-8")
    dense = pose_service.dense_targets_from_csv(str(path), 3, ["walk", "other"])
    assert dense.shape == (3,)
    assert dense[0] == 0


def test_dense_targets_with_no_label_file_is_all_other():
    dense = pose_service.dense_targets_from_csv(None, 5, ["walk", "other"])
    assert (dense == 1).all()


# ── Extraction ────────────────────────────────────────────────────────────

def test_extract_one_aligns_features_and_targets(dlc_csv, label_csv):
    out = pose_service.extract_one(
        pose_path=dlc_csv, label_path=label_csv, origin="deeplabcut",
        bodyparts=BODYPARTS, class_names=["rearing", "grooming", "other"],
        frames2integ=30,
    )
    assert out["features"].shape[0] == out["targets"].shape[0]
    assert out["features"].shape[1] == 16   # 4 body parts
    assert out["bin_width"] == 3
    assert out["n_frames"] == 3000
    assert out["is_3d"] is False
    # All three classes should be represented (the sparse fill produced "other").
    assert set(np.unique(out["targets"])) == {0, 1, 2}


def test_extract_one_stops_after_the_last_label(tmp_path, dlc_csv):
    """A pose file tracking far past the reviewed footage must not flood the
    training set with unreviewed 'other' rows (the 4-hour-pose / 6-minute-video
    case, which produced a 99.8%-other dataset)."""
    # Labels cover only the first 600 of 3000 frames.
    label_path = tmp_path / "short_labels.csv"
    lines = ["time,rearing"] + [f"{i},1" for i in range(0, 600)]
    label_path.write_text("\n".join(lines), encoding="utf-8")

    out = pose_service.extract_one(
        pose_path=dlc_csv, label_path=str(label_path), origin="deeplabcut",
        bodyparts=BODYPARTS, class_names=["rearing", "other"], frames2integ=30,
    )
    assert out["n_frames_total"] == 3000
    assert out["n_frames"] <= 600 + out["bin_width"]
    # Almost everything kept is a real label, not unreviewed filler.
    other_code = 1
    assert (out["targets"] == other_code).mean() < 0.1


def test_extract_one_can_keep_the_unlabeled_tail(tmp_path, dlc_csv):
    label_path = tmp_path / "short_labels.csv"
    lines = ["time,rearing"] + [f"{i},1" for i in range(0, 600)]
    label_path.write_text("\n".join(lines), encoding="utf-8")

    out = pose_service.extract_one(
        pose_path=dlc_csv, label_path=str(label_path), origin="deeplabcut",
        bodyparts=BODYPARTS, class_names=["rearing", "other"], frames2integ=30,
        limit_to_labeled=False,
    )
    assert out["n_frames"] == 3000
    assert (out["targets"] == 1).mean() > 0.7   # dominated by unreviewed "other"


def test_extract_one_without_labels_is_all_other(dlc_csv):
    out = pose_service.extract_one(
        pose_path=dlc_csv, label_path=None, origin="deeplabcut",
        bodyparts=BODYPARTS, class_names=["rearing", "other"], frames2integ=30,
    )
    assert (out["targets"] == 1).all()


def test_extract_one_works_on_sleap(sleap_h5, label_csv):
    out = pose_service.extract_one(
        pose_path=sleap_h5, label_path=label_csv, origin="sleap",
        bodyparts=BODYPARTS, class_names=["rearing", "grooming", "other"],
        frames2integ=30,
    )
    assert out["features"].shape[1] == 16
    assert out["features"].shape[0] > 0


def test_extraction_task_writes_npz_and_summary(tmp_path, dlc_csv, label_csv):
    project = tmp_path / "proj"
    project.mkdir()
    task_id = pose_service.start_extraction(
        project_dir=str(project),
        videos=[{"id": "vid1", "pose_path": dlc_csv, "label_path": label_csv}],
        origin="deeplabcut", bodyparts=BODYPARTS, framerate=FPS,
    )
    status = wait_for(lambda: pose_service.get_task(task_id)["status"])
    task = pose_service.get_task(task_id)
    assert status == "completed", task.get("error")

    npz = project / "iteration-0" / "features" / "vid1.npz"
    assert npz.exists()
    with np.load(npz, allow_pickle=True) as data:
        assert data["features"].shape[1] == 16
        assert int(data["frames2integ"]) == 30
        assert int(data["bin_width"]) == 3
        assert [str(x) for x in data["bodyparts"]] == BODYPARTS
        assert [str(x) for x in data["class_names"]][-1] == "other"

    results = task["results"]
    assert results["class_names"][-1] == "other"
    assert results["videos"][0]["n_rows"] > 0
    assert results["videos"][0]["class_counts"]["other"] > 0


def test_extraction_task_uses_requested_iteration_dir(tmp_path, dlc_csv, label_csv):
    project = tmp_path / "proj"
    project.mkdir()
    task_id = pose_service.start_extraction(
        project_dir=str(project),
        videos=[{"id": "vid1", "pose_path": dlc_csv, "label_path": label_csv}],
        origin="deeplabcut", bodyparts=BODYPARTS, framerate=FPS,
        iteration_dir="iteration-3",
    )
    wait_for(lambda: pose_service.get_task(task_id)["status"])
    assert (project / "iteration-3" / "features" / "vid1.npz").exists()


def test_extraction_errors_without_any_labels(tmp_path, dlc_csv):
    project = tmp_path / "proj"
    project.mkdir()
    task_id = pose_service.start_extraction(
        project_dir=str(project),
        videos=[{"id": "vid1", "pose_path": dlc_csv, "label_path": None}],
        origin="deeplabcut", bodyparts=BODYPARTS, framerate=FPS,
    )
    status = wait_for(lambda: pose_service.get_task(task_id)["status"])
    assert status == "error"
    assert "No labeled behaviors" in pose_service.get_task(task_id)["error"]


# ── load_features ─────────────────────────────────────────────────────────

def test_load_features_stacks_videos(tmp_path, dlc_csv, label_csv):
    project = tmp_path / "proj"
    project.mkdir()
    task_id = pose_service.start_extraction(
        project_dir=str(project),
        videos=[
            {"id": "v1", "pose_path": dlc_csv, "label_path": label_csv},
            {"id": "v2", "pose_path": dlc_csv, "label_path": label_csv},
        ],
        origin="deeplabcut", bodyparts=BODYPARTS, framerate=FPS,
    )
    wait_for(lambda: pose_service.get_task(task_id)["status"])

    X1, y1, _ = pose_service.load_features(str(project), ["v1"])
    X2, y2, meta = pose_service.load_features(str(project), ["v1", "v2"])
    assert X2.shape[0] == 2 * X1.shape[0]
    assert y2.shape[0] == 2 * y1.shape[0]
    assert meta["bodyparts"] == BODYPARTS
    assert meta["frames2integ"] == 30


def test_load_features_missing_video_raises(tmp_path):
    project = tmp_path / "proj"
    (project / "iteration-0" / "features").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Run feature extraction"):
        pose_service.load_features(str(project), ["nope"])


def test_load_features_rejects_mismatched_widths(tmp_path):
    """Videos extracted with different body-part selections must not be stacked."""
    fdir = tmp_path / "proj" / "iteration-0" / "features"
    fdir.mkdir(parents=True)
    common = dict(frames2integ=30, bin_width=3, n_frames=100, is_3d=False,
                  bodyparts=np.array(["a", "b"], dtype=object),
                  class_names=np.array(["x", "other"], dtype=object))
    np.savez(fdir / "v1.npz", features=np.zeros((5, 16)),
             targets=np.zeros(5, dtype=int), **common)
    np.savez(fdir / "v2.npz", features=np.zeros((5, 25)),
             targets=np.zeros(5, dtype=int), **common)
    with pytest.raises(ValueError, match="inconsistent feature widths"):
        pose_service.load_features(str(tmp_path / "proj"), ["v1", "v2"])


# ── Keypoints for the overlay ─────────────────────────────────────────────

def test_keypoints_returns_column_major_window(dlc_csv):
    out = pose_service.keypoints(dlc_csv, "deeplabcut", BODYPARTS, start=10, end=60)
    assert out["n_frames"] == 50
    assert out["start"] == 10
    assert len(out["points"]) == len(BODYPARTS)
    p = out["points"][0]
    assert len(p["x"]) == len(p["y"]) == len(p["likelihood"]) == 50


def test_keypoints_clamps_end_to_file_length(dlc_csv):
    out = pose_service.keypoints(dlc_csv, "deeplabcut", BODYPARTS, start=0, end=999999)
    assert out["n_frames"] == 3000


def test_inspect_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pose_service.inspect("/no/such/pose.csv", "deeplabcut")
