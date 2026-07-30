"""Endpoint-level tests: schemas, validation, and the task/SSE lifecycle."""

import os

import pytest

from backend.services import inference_service, pose_service, training_service
from tests.conftest import BODYPARTS, FPS, wait_for


# ── Health ────────────────────────────────────────────────────────────────

async def test_health_reports_the_numeric_stack(client):
    async with client as ac:
        resp = await ac.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    runtime = body["runtime"]
    assert "numpy" in runtime
    assert "sklearn" in runtime
    assert "numba" in runtime


async def test_health_does_not_mention_torch(client):
    """The torch stack is gone; nothing should still be probing for it."""
    async with client as ac:
        resp = await ac.get("/api/health")
    assert "torch" not in resp.text.lower()
    assert "cuda" not in resp.text.lower()


# ── Pose router ───────────────────────────────────────────────────────────

async def test_inspect_pose_endpoint(client, dlc_csv):
    async with client as ac:
        resp = await ac.post("/api/pose/inspect",
                             json={"pose_path": dlc_csv, "origin": "deeplabcut"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_frames"] == 3000
    assert sorted(body["bodyparts"]) == sorted(BODYPARTS)


async def test_inspect_pose_missing_file_is_404(client):
    async with client as ac:
        resp = await ac.post("/api/pose/inspect",
                             json={"pose_path": "/no/such.csv", "origin": "deeplabcut"})
    assert resp.status_code == 404


async def test_inspect_pose_bad_origin_is_400(client, dlc_csv):
    async with client as ac:
        resp = await ac.post("/api/pose/inspect",
                             json={"pose_path": dlc_csv, "origin": "nonsense"})
    assert resp.status_code == 400


async def test_keypoints_endpoint(client, dlc_csv):
    async with client as ac:
        resp = await ac.post("/api/pose/keypoints", json={
            "pose_path": dlc_csv, "origin": "deeplabcut",
            "bodyparts": BODYPARTS, "start": 0, "end": 25,
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_frames"] == 25
    assert len(body["points"]) == len(BODYPARTS)


async def test_extract_features_rejects_fewer_than_two_bodyparts(client, tmp_path, dlc_csv):
    async with client as ac:
        resp = await ac.post("/api/pose/extract-features", json={
            "project_dir": str(tmp_path), "origin": "deeplabcut",
            "videos": [{"id": "v", "pose_path": dlc_csv}],
            "bodyparts": ["nose"], "framerate": FPS,
        })
    assert resp.status_code == 422
    assert "at least two" in resp.json()["detail"]


async def test_extract_features_rejects_empty_video_list(client, tmp_path):
    async with client as ac:
        resp = await ac.post("/api/pose/extract-features", json={
            "project_dir": str(tmp_path), "origin": "deeplabcut",
            "videos": [], "bodyparts": BODYPARTS, "framerate": FPS,
        })
    assert resp.status_code == 422


async def test_extract_features_lifecycle(client, tmp_path, dlc_csv, label_csv):
    project = tmp_path / "proj"
    project.mkdir()
    async with client as ac:
        resp = await ac.post("/api/pose/extract-features", json={
            "project_dir": str(project), "origin": "deeplabcut",
            "videos": [{"id": "v1", "pose_path": dlc_csv, "label_path": label_csv}],
            "bodyparts": BODYPARTS, "framerate": FPS,
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        wait_for(lambda: pose_service.get_task(task_id)["status"])

        results = await ac.get(f"/api/pose/results/{task_id}")
        assert results.status_code == 200
        body = results.json()
        assert body["class_names"][-1] == "other"
        assert body["bin_width"] == 3
        assert body["videos"][0]["n_features"] == 16


async def test_pose_results_before_completion_is_400(client, tmp_path, dlc_csv):
    project = tmp_path / "proj"
    project.mkdir()
    async with client as ac:
        resp = await ac.post("/api/pose/extract-features", json={
            "project_dir": str(project), "origin": "deeplabcut",
            "videos": [{"id": "v1", "pose_path": dlc_csv}],
            "bodyparts": BODYPARTS, "framerate": FPS,
        })
        task_id = resp.json()["task_id"]
        # The task errors (no labels); results must still refuse to serve.
        wait_for(lambda: pose_service.get_task(task_id)["status"])
        results = await ac.get(f"/api/pose/results/{task_id}")
    assert results.status_code == 400


async def test_pose_status_unknown_task_is_404(client):
    async with client as ac:
        resp = await ac.get("/api/pose/status/nope")
    assert resp.status_code == 404


# ── Training router ───────────────────────────────────────────────────────

async def test_training_rejects_empty_video_list(client, tmp_path):
    async with client as ac:
        resp = await ac.post("/api/training/start", json={
            "project_dir": str(tmp_path), "video_ids": [],
        })
    assert resp.status_code == 422


async def test_training_start_returns_a_task_id(client, tmp_path):
    async with client as ac:
        resp = await ac.post("/api/training/start", json={
            "project_dir": str(tmp_path), "video_ids": ["nope"],
        })
    assert resp.status_code == 200
    assert "task_id" in resp.json()


async def test_training_stop_unknown_task_is_404(client):
    async with client as ac:
        resp = await ac.post("/api/training/stop/nope")
    assert resp.status_code == 404


async def test_checkpoints_endpoint_returns_a_list(client, tmp_path):
    async with client as ac:
        resp = await ac.get("/api/training/checkpoints",
                            params={"search_dirs": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json()["checkpoints"] == []


# ── Inference router ──────────────────────────────────────────────────────

async def test_inference_batch_rejects_empty_file_list(client, tmp_path):
    async with client as ac:
        resp = await ac.post("/api/inference/batch", json={
            "checkpoint_path": "x.joblib", "pose_paths": [],
            "output_dir": str(tmp_path),
        })
    assert resp.status_code == 422


async def test_inference_results_unknown_task_is_404(client):
    async with client as ac:
        resp = await ac.get("/api/inference/results/nope")
    assert resp.status_code == 404


async def test_inference_stop_unknown_task_is_404(client):
    async with client as ac:
        resp = await ac.post("/api/inference/stop/nope")
    assert resp.status_code == 404


async def test_inference_run_requires_a_pose_path(client):
    async with client as ac:
        resp = await ac.post("/api/inference/run",
                             json={"checkpoint_path": "x.joblib"})
    assert resp.status_code == 422, "pose_path is required by the schema"


# ── Project router ────────────────────────────────────────────────────────

async def test_export_and_inspect_rlt_over_http(client, sample_project, tmp_path):
    project_dir, _ = sample_project
    out = str(tmp_path / "api.rlt")
    async with client as ac:
        resp = await ac.post("/api/project/export-rlt", json={
            "project_dir": project_dir, "output_path": out, "package": False,
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["summary"]["labels"] == 2

        info = await ac.post("/api/project/inspect-rlt", json={"rlt_path": out})
    assert info.status_code == 200
    assert info.json()["labels"] == 2


async def test_export_rlt_missing_project_is_404(client, tmp_path):
    async with client as ac:
        resp = await ac.post("/api/project/export-rlt", json={
            "project_dir": str(tmp_path / "nope"),
            "output_path": str(tmp_path / "x.rlt"),
        })
    assert resp.status_code == 404


# ── Route surface ─────────────────────────────────────────────────────────

async def test_removed_tsm_routes_are_gone(client):
    """The pixel pipeline's endpoints must not still be mounted."""
    async with client as ac:
        for path in ("/api/preprocess/background", "/api/preprocess/run",
                     "/api/training/export-torchscript", "/api/training/benchmark-inference",
                     "/api/training/check-coverage"):
            resp = await ac.post(path, json={})
            assert resp.status_code == 404, f"{path} still responds ({resp.status_code})"
