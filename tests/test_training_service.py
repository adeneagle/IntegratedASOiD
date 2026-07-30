"""Tests for the training service: the AL loop, checkpoints, and cancellation."""

import os

import joblib
import numpy as np
import pytest

from backend.services import pose_service, training_service
from tests.conftest import BODYPARTS, FPS, wait_for


@pytest.fixture
def extracted_project(tmp_path, dlc_csv, label_csv):
    """A project with features already extracted for one video."""
    project = tmp_path / "proj"
    project.mkdir()
    task_id = pose_service.start_extraction(
        project_dir=str(project),
        videos=[{"id": "vid1", "pose_path": dlc_csv, "label_path": label_csv}],
        origin="deeplabcut", bodyparts=BODYPARTS, framerate=FPS,
    )
    status = wait_for(lambda: pose_service.get_task(task_id)["status"])
    assert status == "completed", pose_service.get_task(task_id).get("error")
    return str(project)


def _train(project, **overrides):
    config = {
        "project_dir": project,
        "video_ids": ["vid1"],
        "iteration_dir": "iteration-0",
        "init_ratio": 0.05,
        "max_iter": 4,
        "max_samples_iter": 20,
        "conf_threshold": 0.6,
        "exclude_other": False,
        "checkpoint_dir": os.path.join(project, "models", "checkpoints"),
        "model_name": "test_model",
    }
    config.update(overrides)
    task_id = training_service.start_training(config)
    status = wait_for(lambda: training_service.get_task(task_id)["status"])
    return task_id, status


def test_training_completes_and_writes_a_checkpoint(extracted_project):
    task_id, status = _train(extracted_project)
    task = training_service.get_task(task_id)
    assert status == "completed", task.get("error")

    ckpt = task["checkpoint_path"]
    assert ckpt and os.path.exists(ckpt)
    assert ckpt.endswith(".joblib")
    assert "test_model" in os.path.basename(ckpt)


def test_checkpoint_records_everything_inference_needs(extracted_project):
    task_id, status = _train(extracted_project)
    assert status == "completed"
    bundle = joblib.load(training_service.get_task(task_id)["checkpoint_path"])

    # Without these, inference cannot reproduce the training feature space.
    assert bundle["bodyparts"] == BODYPARTS
    assert bundle["bodypart_fingerprint"]
    assert bundle["frames2integ"] == 30
    assert bundle["bin_width"] == 3
    assert bundle["is_3d"] is False
    assert bundle["class_names"][-1] == "other"
    assert bundle["n_features"] == 16
    assert hasattr(bundle["model"], "predict_proba")


def test_metrics_stream_has_ordered_startup_then_iterations(extracted_project):
    task_id, status = _train(extracted_project)
    assert status == "completed"
    events = training_service.read_all_metrics(task_id)

    assert events, "metrics file should not be empty"
    kinds = [e.get("event") for e in events]
    assert "start" in kinds
    assert "ceiling" in kinds
    assert kinds.count("iteration") >= 1
    # Startup messages must arrive before the first iteration.
    assert kinds.index("start") < kinds.index("iteration")

    iters = [e for e in events if e.get("event") == "iteration"]
    for e in iters:
        assert {"macro_f1", "f1_per_class", "n_train", "n_lowconf_remaining"} <= set(e)
    # Iteration numbers are contiguous from 0.
    assert [e["iteration"] for e in iters] == list(range(len(iters)))


def test_active_learning_uses_a_fraction_of_the_data(extracted_project):
    task_id, status = _train(extracted_project)
    assert status == "completed"
    events = training_service.read_all_metrics(task_id)
    start = next(e for e in events if e.get("event") == "start")
    last = [e for e in events if e.get("event") == "iteration"][-1]
    assert last["n_train"] < start["n_train"], "AL should train on a subset"


def test_read_latest_metrics_reports_terminal_status(extracted_project):
    task_id, _ = _train(extracted_project)
    latest = training_service.read_latest_metrics(task_id)
    assert latest["status"] == "completed"
    assert "checkpoint_path" in latest


def test_binarize_collapses_the_label_space(extracted_project):
    task_id, status = _train(extracted_project, binarize="rearing")
    task = training_service.get_task(task_id)
    assert status == "completed", task.get("error")
    bundle = joblib.load(task["checkpoint_path"])
    assert bundle["class_names"] == ["other", "rearing"]


def test_label_union_merges_behaviors_into_one_positive_class(extracted_project):
    task_id, status = _train(extracted_project, label_union=["rearing", "grooming"])
    task = training_service.get_task(task_id)
    assert status == "completed", task.get("error")
    bundle = joblib.load(task["checkpoint_path"])
    assert bundle["class_names"] == ["other", "positive"]


def test_training_errors_on_missing_features(tmp_path):
    project = tmp_path / "empty"
    (project / "iteration-0" / "features").mkdir(parents=True)
    task_id, status = _train(str(project))
    assert status == "error"
    assert "feature" in training_service.get_task(task_id)["error"].lower()


def test_stop_training_sets_the_cancel_flag(extracted_project):
    task_id = training_service.start_training({
        "project_dir": extracted_project,
        "video_ids": ["vid1"],
        "init_ratio": 0.05,
        "max_iter": 500,          # long enough to still be running
        "max_samples_iter": 1,
        "conf_threshold": 0.9,
        "checkpoint_dir": os.path.join(extracted_project, "models", "checkpoints"),
    })
    assert training_service.stop_training(task_id) is True
    status = wait_for(lambda: training_service.get_task(task_id)["status"])
    assert status == "cancelled"


def test_stop_training_unknown_task_returns_false():
    assert training_service.stop_training("not-a-task") is False


def test_list_checkpoints_surfaces_metadata(extracted_project):
    task_id, status = _train(extracted_project)
    assert status == "completed"
    found = training_service.list_checkpoints(
        [os.path.join(extracted_project, "models")]
    )
    assert len(found) == 1
    entry = found[0]
    assert entry["filename"].endswith(".joblib")
    assert entry["bodyparts"] == BODYPARTS
    assert entry["class_names"][-1] == "other"
    assert entry["frames2integ"] == 30


def test_list_checkpoints_ignores_missing_dirs():
    assert training_service.list_checkpoints(["/no/such/dir"]) == []


def test_collapse_labels_is_a_noop_without_binarize():
    y = np.array([0, 1, 2])
    names = ["a", "b", "other"]
    out_y, out_names = training_service._collapse_labels(y, names, None, None)
    assert out_names == names
    np.testing.assert_array_equal(out_y, y)


def test_collapse_labels_ignores_unknown_behavior_names():
    y = np.array([0, 1, 2])
    names = ["a", "b", "other"]
    _out_y, out_names = training_service._collapse_labels(y, names, "nonexistent", None)
    assert out_names == names
