"""Tests for the inference service.

The response-shape tests matter most: the renderer's Active Learning panel was
carried over from the TSM app untouched, so it consumes exactly
`{frames: [{index, predicted_class, uncertainty}], class_names}`. Any drift here
silently breaks the uncertainty heatmap.
"""

import os

import joblib
import numpy as np
import pytest

from backend.services import inference_service, pose_service, training_service
from tests.conftest import BODYPARTS, FPS, wait_for


@pytest.fixture
def trained_project(tmp_path, dlc_csv, label_csv):
    """A project with features extracted and a model trained. Returns (dir, ckpt)."""
    project = tmp_path / "proj"
    project.mkdir()
    task_id = pose_service.start_extraction(
        project_dir=str(project),
        videos=[{"id": "vid1", "pose_path": dlc_csv, "label_path": label_csv}],
        origin="deeplabcut", bodyparts=BODYPARTS, framerate=FPS,
    )
    assert wait_for(lambda: pose_service.get_task(task_id)["status"]) == "completed"

    train_id = training_service.start_training({
        "project_dir": str(project),
        "video_ids": ["vid1"],
        "init_ratio": 0.1,
        "max_iter": 3,
        "max_samples_iter": 20,
        "conf_threshold": 0.6,
        "checkpoint_dir": os.path.join(str(project), "models", "checkpoints"),
        "model_name": "m",
    })
    status = wait_for(lambda: training_service.get_task(train_id)["status"])
    assert status == "completed", training_service.get_task(train_id).get("error")
    return str(project), training_service.get_task(train_id)["checkpoint_path"]


def _infer(ckpt, pose_path, **kwargs):
    task_id = inference_service.start_inference(
        checkpoint_path=ckpt, pose_path=pose_path, origin="deeplabcut", **kwargs
    )
    status = wait_for(lambda: inference_service.get_task(task_id)["status"])
    return inference_service.get_task(task_id), status


def test_results_match_the_renderer_contract(trained_project, dlc_csv):
    _project, ckpt = trained_project
    task, status = _infer(ckpt, dlc_csv)
    assert status == "completed", task.get("error")

    results = task["results"]
    assert set(results) >= {"frames", "class_names"}
    assert isinstance(results["class_names"], list)

    frames = results["frames"]
    assert len(frames) == 3000, "one entry per native video frame"
    assert set(frames[0]) >= {"index", "predicted_class", "uncertainty"}
    assert isinstance(frames[0]["index"], int)
    assert isinstance(frames[0]["predicted_class"], int)
    assert isinstance(frames[0]["uncertainty"], float)


def test_frames_are_native_indexed_and_contiguous(trained_project, dlc_csv):
    _project, ckpt = trained_project
    task, _ = _infer(ckpt, dlc_csv)
    indices = [f["index"] for f in task["results"]["frames"]]
    assert indices == list(range(3000))


def test_uncertainty_is_normalized(trained_project, dlc_csv):
    _project, ckpt = trained_project
    task, _ = _infer(ckpt, dlc_csv)
    unc = np.array([f["uncertainty"] for f in task["results"]["frames"]])
    assert unc.min() >= 0.0
    assert unc.max() <= 1.0


def test_predicted_classes_index_into_class_names(trained_project, dlc_csv):
    _project, ckpt = trained_project
    task, _ = _infer(ckpt, dlc_csv)
    results = task["results"]
    n = len(results["class_names"])
    for f in results["frames"]:
        assert 0 <= f["predicted_class"] < n


def test_predictions_recover_the_ground_truth(trained_project, dlc_csv, behavior_series):
    """A sanity check that the whole pipeline actually learned something."""
    _project, ckpt = trained_project
    task, _ = _infer(ckpt, dlc_csv)
    results = task["results"]
    pred = np.array([f["predicted_class"] for f in results["frames"]])
    rearing = results["class_names"].index("rearing")
    truth_mask = behavior_series == 1
    accuracy = (pred[truth_mask] == rearing).mean()
    assert accuracy > 0.8, f"accuracy on labeled frames was only {accuracy:.2f}"


def test_frame_ranges_scope_the_output(trained_project, dlc_csv):
    _project, ckpt = trained_project
    task, status = _infer(ckpt, dlc_csv, frame_ranges=[[100, 199], [500, 549]])
    assert status == "completed", task.get("error")
    frames = task["results"]["frames"]
    assert len(frames) == 150
    assert frames[0]["index"] == 100
    assert frames[99]["index"] == 199
    assert frames[100]["index"] == 500


def test_frame_ranges_normalize_reversed_pairs(trained_project, dlc_csv):
    _project, ckpt = trained_project
    task, _ = _infer(ckpt, dlc_csv, frame_ranges=[[199, 100]])
    assert len(task["results"]["frames"]) == 100


def test_bodypart_mismatch_is_rejected(trained_project, dlc_csv):
    """The upstream 1710d15 failure: a different selection must not silently run."""
    _project, ckpt = trained_project
    task, status = _infer(ckpt, dlc_csv, bodyparts=["nose", "tailbase"])
    assert status == "error"
    assert "body-part selection" in task["error"]


def test_bodyparts_default_to_the_checkpoint(trained_project, dlc_csv):
    _project, ckpt = trained_project
    task, status = _infer(ckpt, dlc_csv, bodyparts=None)
    assert status == "completed", task.get("error")


def test_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        inference_service.load_model("/no/such/model.joblib")


def test_non_checkpoint_file_rejected(tmp_path):
    bad = tmp_path / "notamodel.joblib"
    joblib.dump([1, 2, 3], bad)
    with pytest.raises(ValueError, match="not an A-SOID Desktop checkpoint"):
        inference_service.load_model(str(bad))


def test_wanted_set_none_means_whole_video():
    assert inference_service._wanted_set(None) is None
    assert inference_service._wanted_set([]) is None


def test_wanted_set_is_inclusive():
    assert inference_service._wanted_set([[2, 4]]) == {2, 3, 4}


def test_cancel_stops_inference(trained_project, dlc_csv):
    _project, ckpt = trained_project
    task_id = inference_service.start_inference(
        checkpoint_path=ckpt, pose_path=dlc_csv, origin="deeplabcut"
    )
    inference_service.get_task(task_id)["cancel"] = True
    status = wait_for(lambda: inference_service.get_task(task_id)["status"])
    assert status in ("cancelled", "completed")  # may finish before the flag is read


# ── Batch prediction ──────────────────────────────────────────────────────

def test_batch_writes_importable_one_hot_csv(trained_project, dlc_csv, tmp_path):
    _project, ckpt = trained_project
    outdir = tmp_path / "batch_out"
    task_id = inference_service.start_batch_inference(
        checkpoint_path=ckpt, pose_paths=[dlc_csv],
        output_dir=str(outdir), origin="deeplabcut",
    )
    status = wait_for(lambda: inference_service.get_task(task_id)["status"])
    task = inference_service.get_task(task_id)
    assert status == "completed", task.get("error")

    files = task["results"]["files"]
    assert len(files) == 1
    out_csv = files[0]["output_path"]
    assert os.path.exists(out_csv)

    with open(out_csv, encoding="utf-8") as f:
        header = f.readline().strip()
        first = f.readline().strip()
    # Must be the same layout the labeling UI imports.
    assert header.startswith("time,")
    assert header.split(",")[1:] == list(files[0]["summary"].keys())
    assert first.split(",")[0] == "0"
    # Exactly one class flagged per row.
    assert sum(int(v) for v in first.split(",")[1:]) == 1


def test_batch_csv_round_trips_through_the_label_parser(trained_project, dlc_csv, tmp_path):
    _project, ckpt = trained_project
    outdir = tmp_path / "batch_out"
    task_id = inference_service.start_batch_inference(
        checkpoint_path=ckpt, pose_paths=[dlc_csv],
        output_dir=str(outdir), origin="deeplabcut",
    )
    wait_for(lambda: inference_service.get_task(task_id)["status"])
    out_csv = inference_service.get_task(task_id)["results"]["files"][0]["output_path"]

    names, frame_to_col = pose_service.read_label_csv(out_csv)
    assert names, "predictions must parse as labels"
    assert len(frame_to_col) == 3000


def test_batch_summary_reports_per_class_counts(trained_project, dlc_csv, tmp_path):
    _project, ckpt = trained_project
    task_id = inference_service.start_batch_inference(
        checkpoint_path=ckpt, pose_paths=[dlc_csv],
        output_dir=str(tmp_path / "out"), origin="deeplabcut",
    )
    wait_for(lambda: inference_service.get_task(task_id)["status"])
    summary = inference_service.get_task(task_id)["results"]["files"][0]["summary"]
    assert sum(s["frames"] for s in summary.values()) == 3000
    for s in summary.values():
        assert 0.0 <= s["mean_uncertainty"] <= 1.0
